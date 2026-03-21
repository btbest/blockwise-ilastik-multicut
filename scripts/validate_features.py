#!/usr/bin/env python
"""Compare blimp's feature computation against ilastik's cached features.

Reads an ilastik .ilp project file, loads the input data and cached
superpixels for each training lane, recomputes features using blimp's
``compute_ilastikrag_features()``, and compares them against the
EdgeFeatures that ilastik cached inside the .ilp.

This helps diagnose whether segmentation differences between blimp and
ilastik originate in the feature computation step.

Usage
-----
    python scripts/validate_features.py project.ilp
    python scripts/validate_features.py project.ilp --lanes 0 1
    python scripts/validate_features.py project.ilp --rf-comparison

The script resolves file paths stored in the .ilp (relative to the
project directory).  Lanes whose data files cannot be found on disk
are skipped with a warning.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path, PureWindowsPath

import h5py
import numpy as np
import pandas as pd

# Allow running from the repo root or from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ilp_reader import (
    _open_ilp_file,
    _dataframe_from_hdf5,
    APPLET_GROUP,
    WSDT_GROUP,
    discover_lanes,
    read_edge_features,
    read_feature_names,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _resolve_ilp_filepath(ilp_path: str, raw_path: str) -> tuple[Path | None, str | None]:
    """Resolve a filePath from the .ilp's Input Data to a local path.

    ilastik stores paths with Windows backslashes and may embed an HDF5
    internal path after the file extension (e.g.
    ``Downloads\\foo.h5\\exported_data``).

    Returns
    -------
    (file_path, internal_path)
        *file_path* is a resolved ``Path`` (or None if not found).
        *internal_path* is the HDF5 dataset path (or None for non-HDF5).
    """
    ilp_dir = Path(ilp_path).resolve().parent

    # Normalise Windows backslashes to forward slashes.
    raw_path = raw_path.replace("\\", "/")

    # Split off HDF5 internal path.  Convention: everything after the first
    # path component that ends with .h5 / .hdf5 is the internal path.
    parts = raw_path.split("/")
    file_parts: list[str] = []
    internal_path: str | None = None
    for i, p in enumerate(parts):
        file_parts.append(p)
        lower = p.lower()
        if lower.endswith(".h5") or lower.endswith(".hdf5"):
            internal_path = "/".join(parts[i + 1:]) or None
            break

    file_rel = "/".join(file_parts)

    # Try relative to ilp directory first, then absolute.
    candidate = ilp_dir / file_rel
    if candidate.exists():
        return candidate, internal_path
    candidate = Path(file_rel)
    if candidate.exists():
        return candidate, internal_path

    return None, internal_path


def _load_volume(path: Path, internal_path: str | None = None) -> np.ndarray:
    """Load a volume from HDF5, TIFF, or zarr.

    A trailing singleton channel axis (shape …×1) is automatically squeezed.
    """
    suffix = path.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        with h5py.File(str(path), "r") as f:
            if internal_path:
                data = f[internal_path][()]
            else:
                # First dataset.
                datasets: list[str] = []
                f.visititems(
                    lambda name, obj: datasets.append(name)
                    if isinstance(obj, h5py.Dataset)
                    else None
                )
                if not datasets:
                    raise ValueError(f"No datasets in {path}")
                data = f[datasets[0]][()]

    elif suffix in (".tif", ".tiff"):
        try:
            import tifffile
            data = tifffile.imread(str(path))
        except ImportError:
            raise ImportError(
                "tifffile is required to read .tif files.  "
                "Install with: pip install tifffile"
            )

    elif ".zarr" in str(path):
        import zarr
        store = zarr.open(str(path), mode="r")
        if hasattr(store, "arrays"):
            arrays = list(store.arrays())
            data = arrays[0][1][()] if arrays else store[()]
        else:
            data = store[()]

    else:
        raise ValueError(f"Unsupported file format: {path}")

    # Squeeze trailing singleton channel.
    if data.ndim >= 2 and data.shape[-1] == 1:
        data = data[..., 0]

    return data


# ---------------------------------------------------------------------------
# Read lane input data from .ilp
# ---------------------------------------------------------------------------


def _read_lane_info(ilp_path: str, lane: int) -> dict:
    """Read Input Data metadata for one lane.

    Returns
    -------
    dict with keys:
        'raw_data_path', 'raw_data_internal'   – Raw Data file
        'probs_path', 'probs_internal'         – Probabilities file
        'channel_selection'                    – int, which prob channel to use
    """
    lane_key = f"lane{lane:04d}"
    with _open_ilp_file(ilp_path) as f:
        infos = f["Input Data/infos"]
        if lane_key not in infos:
            raise KeyError(f"Lane {lane} not found in Input Data/infos")

        lane_g = infos[lane_key]
        result = {}

        # Raw Data
        if "Raw Data" in lane_g and "filePath" in lane_g["Raw Data"]:
            raw_fp = lane_g["Raw Data/filePath"][()]
            if isinstance(raw_fp, bytes):
                raw_fp = raw_fp.decode()
            result["raw_data_path"], result["raw_data_internal"] = (
                _resolve_ilp_filepath(ilp_path, raw_fp)
            )
            result["raw_data_orig"] = raw_fp
        else:
            result["raw_data_path"] = None
            result["raw_data_internal"] = None
            result["raw_data_orig"] = None

        # Probabilities
        if "Probabilities" in lane_g and "filePath" in lane_g["Probabilities"]:
            prob_fp = lane_g["Probabilities/filePath"][()]
            if isinstance(prob_fp, bytes):
                prob_fp = prob_fp.decode()
            result["probs_path"], result["probs_internal"] = (
                _resolve_ilp_filepath(ilp_path, prob_fp)
            )
            result["probs_orig"] = prob_fp
        else:
            result["probs_path"] = None
            result["probs_internal"] = None
            result["probs_orig"] = None

        # Channel selection (which probability channel is the boundary channel)
        if WSDT_GROUP in f and "ChannelSelections" in f[WSDT_GROUP]:
            cs = f[WSDT_GROUP]["ChannelSelections"][()]
            result["channel_selection"] = int(cs[0]) if len(cs) > 0 else 0
        else:
            result["channel_selection"] = 0

    return result


def _read_cached_superpixels(ilp_path: str, lane: int) -> np.ndarray | None:
    """Load the cached DT Watershed superpixels for one lane.

    Returns a 3-D uint32 array, or None if not cached.
    """
    sp_key = f"superpixels{lane:03d}"
    with _open_ilp_file(ilp_path) as f:
        if WSDT_GROUP not in f:
            return None
        ws_g = f[WSDT_GROUP]
        if "Superpixels" not in ws_g:
            return None
        sp_g = ws_g["Superpixels"]
        if sp_key not in sp_g:
            return None

        block_g = sp_g[sp_key]
        block_keys = sorted(block_g.keys())
        if len(block_keys) == 1:
            data = block_g[block_keys[0]][()]
            # Squeeze trailing singleton channel.
            if data.ndim == 4 and data.shape[-1] == 1:
                data = data[..., 0]
            return data.astype(np.uint32)
        else:
            # Multi-block: need to reassemble.  ilastik stores blocks as a
            # flat list; the spatial layout depends on the crop shape and the
            # block size (128^3).  For now, only single-block crops are
            # supported.
            print(
                f"  WARNING: lane {lane} has {len(block_keys)} watershed "
                f"blocks; multi-block reassembly is not yet implemented.  "
                f"Skipping."
            )
            return None


def _build_channel_map(
    feature_names: dict[str, list[str]],
    raw_data: np.ndarray,
    probs_data: np.ndarray,
    channel_selection: int,
) -> dict[str, np.ndarray]:
    """Map feature channel names to data arrays.

    In ilastik's multicut workflow:
    - "Raw Data" channel → raw intensity data
    - Any other channel name → the selected boundary probability channel
    """
    channel_data = {}
    for ch_name in feature_names:
        if ch_name == "Raw Data":
            channel_data[ch_name] = raw_data.astype(np.float32)
        else:
            # Boundary / probabilities channel.
            if probs_data.ndim == 4:
                # Multi-channel probabilities: select the boundary channel.
                channel_data[ch_name] = probs_data[..., channel_selection].astype(
                    np.float32
                )
            elif probs_data.ndim == 3:
                channel_data[ch_name] = probs_data.astype(np.float32)
            else:
                raise ValueError(
                    f"Unexpected probabilities shape {probs_data.shape}"
                )
    return channel_data


# ---------------------------------------------------------------------------
# Feature comparison
# ---------------------------------------------------------------------------


def _compare_features(
    ilp_df: pd.DataFrame,
    blimp_features: np.ndarray,
    blimp_edge_ids: np.ndarray,
    feature_names: dict[str, list[str]],
    lane: int,
) -> dict:
    """Compare ilastik's cached features against blimp's recomputed features.

    Returns a summary dict with comparison statistics.
    """
    # Build blimp DataFrame with the same column structure.
    # compute_ilastikrag_features returns features in the order:
    #   [channel1_feat1, channel1_feat2, ..., channel2_feat1, ...]
    # with column names like "Raw Data standard_sp_mean_sum".
    # We need to reconstruct the column names from ilastikrag output.

    # Get ilastik's feature columns (everything except sp1, sp2).
    ilp_feat_cols = [c for c in ilp_df.columns if c not in ("sp1", "sp2")]
    ilp_sp = ilp_df[["sp1", "sp2"]].values.astype(np.uint64)

    # Sort both by edge (sp1, sp2) for alignment.
    # Normalise edge ordering: smaller ID first.
    ilp_edges = np.sort(ilp_sp, axis=1)
    blimp_edges = np.sort(blimp_edge_ids, axis=1)

    # Build edge-keyed lookup for ilastik features.
    ilp_edge_to_idx = {}
    for i, (a, b) in enumerate(ilp_edges):
        ilp_edge_to_idx[(int(a), int(b))] = i

    blimp_edge_to_idx = {}
    for i, (a, b) in enumerate(blimp_edges):
        blimp_edge_to_idx[(int(a), int(b))] = i

    ilp_set = set(ilp_edge_to_idx.keys())
    blimp_set = set(blimp_edge_to_idx.keys())

    common = ilp_set & blimp_set
    only_ilp = ilp_set - blimp_set
    only_blimp = blimp_set - ilp_set

    result = {
        "lane": lane,
        "n_edges_ilastik": len(ilp_set),
        "n_edges_blimp": len(blimp_set),
        "n_common": len(common),
        "n_only_ilastik": len(only_ilp),
        "n_only_blimp": len(only_blimp),
        "ilp_feature_cols": ilp_feat_cols,
        "blimp_n_features": blimp_features.shape[1] if blimp_features.ndim == 2 else 0,
    }

    if not common:
        result["error"] = "No common edges — superpixels likely differ completely."
        return result

    # Align features on common edges.
    ilp_indices = [ilp_edge_to_idx[e] for e in sorted(common)]
    blimp_indices = [blimp_edge_to_idx[e] for e in sorted(common)]

    ilp_vals = ilp_df.iloc[ilp_indices][ilp_feat_cols].values.astype(np.float32)
    blimp_vals = blimp_features[blimp_indices]

    # Column count check.
    if ilp_vals.shape[1] != blimp_vals.shape[1]:
        result["column_mismatch"] = {
            "ilastik_columns": ilp_feat_cols,
            "blimp_n_columns": blimp_vals.shape[1],
            "note": (
                "Column count differs!  This likely means ilastikrag "
                "produces different columns than what ilastik cached.  "
                "Check whether edgeregion features are included/excluded."
            ),
        }
        # Try to match by truncating to the shorter set.
        n_cols = min(ilp_vals.shape[1], blimp_vals.shape[1])
        ilp_vals = ilp_vals[:, :n_cols]
        blimp_vals = blimp_vals[:, :n_cols]
        cols_to_compare = ilp_feat_cols[:n_cols]
    else:
        cols_to_compare = ilp_feat_cols

    # Per-feature comparison.
    per_feature = []
    for i, col in enumerate(cols_to_compare):
        diff = np.abs(ilp_vals[:, i] - blimp_vals[:, i])
        # Relative diff (avoid div-by-zero).
        scale = np.maximum(np.abs(ilp_vals[:, i]), np.abs(blimp_vals[:, i]))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.where(scale > 0, diff / scale, 0.0)

        corr = float(np.corrcoef(ilp_vals[:, i], blimp_vals[:, i])[0, 1]) if len(common) > 1 else float("nan")

        per_feature.append({
            "column": col,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "median_abs_diff": float(np.median(diff)),
            "max_rel_diff": float(rel_diff.max()),
            "mean_rel_diff": float(rel_diff.mean()),
            "correlation": corr,
            "ilastik_range": (float(ilp_vals[:, i].min()), float(ilp_vals[:, i].max())),
            "blimp_range": (float(blimp_vals[:, i].min()), float(blimp_vals[:, i].max())),
            "exact_match": bool(np.array_equal(ilp_vals[:, i], blimp_vals[:, i])),
        })

    result["per_feature"] = per_feature
    result["all_exact"] = all(f["exact_match"] for f in per_feature)

    return result


def _print_comparison(result: dict) -> None:
    """Pretty-print the comparison result for one lane."""
    lane = result["lane"]
    print(f"\n{'='*72}")
    print(f"Lane {lane}")
    print(f"{'='*72}")

    print(
        f"  Edges:  ilastik={result['n_edges_ilastik']}  "
        f"blimp={result['n_edges_blimp']}  "
        f"common={result['n_common']}  "
        f"only-ilastik={result['n_only_ilastik']}  "
        f"only-blimp={result['n_only_blimp']}"
    )

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    if "column_mismatch" in result:
        cm = result["column_mismatch"]
        print(f"\n  COLUMN MISMATCH:")
        print(f"    ilastik columns ({len(cm['ilastik_columns'])}):")
        for c in cm["ilastik_columns"]:
            print(f"      {c}")
        print(f"    blimp columns: {cm['blimp_n_columns']}")
        print(f"    {cm['note']}")

    if result.get("all_exact"):
        print(f"\n  All {len(result['per_feature'])} features match EXACTLY.")
        return

    print(f"\n  Feature columns ({result['blimp_n_features']} blimp, "
          f"{len(result['ilp_feature_cols'])} ilastik):")
    print()

    # Table header.
    hdr = f"  {'Column':<50s} {'MaxAbsDiff':>10s} {'MeanAbsDiff':>11s} {'Corr':>6s} {'Exact':>5s}"
    print(hdr)
    print(f"  {'-'*50} {'-'*10} {'-'*11} {'-'*6} {'-'*5}")

    for f in result["per_feature"]:
        exact = "YES" if f["exact_match"] else "no"
        corr = f"{f['correlation']:.4f}" if not np.isnan(f["correlation"]) else "n/a"
        print(
            f"  {f['column']:<50s} "
            f"{f['max_abs_diff']:>10.6f} "
            f"{f['mean_abs_diff']:>11.6f} "
            f"{corr:>6s} "
            f"{exact:>5s}"
        )

    # Show ranges for non-exact features.
    mismatches = [f for f in result["per_feature"] if not f["exact_match"]]
    if mismatches:
        print(f"\n  Ranges for mismatched features:")
        for f in mismatches:
            il_lo, il_hi = f["ilastik_range"]
            bl_lo, bl_hi = f["blimp_range"]
            print(
                f"    {f['column']:<50s}  "
                f"ilastik=[{il_lo:.4f}, {il_hi:.4f}]  "
                f"blimp=[{bl_lo:.4f}, {bl_hi:.4f}]"
            )


# ---------------------------------------------------------------------------
# RF comparison (optional)
# ---------------------------------------------------------------------------


def _compare_rf_predictions(ilp_path: str, feature_names: dict) -> None:
    """Compare vigra RF predictions on the cached training features.

    Extracts the vigra RF from the .ilp, predicts on the cached
    EdgeFeatures, and reports the prediction statistics.  This verifies
    whether the RF wrapper produces the expected probabilities.
    """
    from fit_classifier import extract_vigra_rf_from_ilp

    print(f"\n{'='*72}")
    print("Random Forest comparison")
    print(f"{'='*72}")

    rf = extract_vigra_rf_from_ilp(ilp_path)
    print(f"  {rf}")

    # Read the RF's expected feature names from the .ilp.
    with _open_ilp_file(ilp_path) as f:
        out_g = f[f"{APPLET_GROUP}/Output"]
        if "feature_names" in out_g:
            rf_feat_names = [
                v.decode() if isinstance(v, bytes) else v
                for v in out_g["feature_names"][()]
            ]
        else:
            rf_feat_names = None

    if rf_feat_names:
        print(f"\n  RF expects {len(rf_feat_names)} features:")
        for fn in rf_feat_names:
            print(f"    {fn}")

    # Predict on each lane's cached features.
    lanes = discover_lanes(ilp_path)
    for lane in lanes:
        try:
            df = read_edge_features(ilp_path, lane=lane)
        except KeyError:
            continue

        feat_cols = [c for c in df.columns if c not in ("sp1", "sp2")]
        X = df[feat_cols].values.astype(np.float32)

        print(f"\n  Lane {lane}: {X.shape[0]} edges, {X.shape[1]} features")
        print(f"    Feature columns: {feat_cols}")

        if rf_feat_names and feat_cols != rf_feat_names:
            print("    WARNING: feature column order differs from RF's expected order!")
            print(f"    RF expects:   {rf_feat_names}")
            print(f"    EdgeFeatures: {feat_cols}")

        if X.shape[1] != len(rf_feat_names or feat_cols):
            print(f"    ERROR: feature count mismatch ({X.shape[1]} vs {len(rf_feat_names or feat_cols)})")
            continue

        probs = rf.predict_proba(X)
        split_col = int(np.argmax(rf.classes_))
        p_split = probs[:, split_col]
        print(f"    P(split) stats:  min={p_split.min():.4f}  "
              f"max={p_split.max():.4f}  mean={p_split.mean():.4f}  "
              f"std={p_split.std():.4f}")
        print(f"    P(split) quartiles:  "
              f"25%={np.percentile(p_split, 25):.4f}  "
              f"50%={np.percentile(p_split, 50):.4f}  "
              f"75%={np.percentile(p_split, 75):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate_lane(ilp_path: str, lane: int, feature_names: dict) -> dict | None:
    """Validate features for a single lane.  Returns comparison dict or None."""
    print(f"\n--- Lane {lane} ---")

    # 1. Load cached superpixels.
    superpixels = _read_cached_superpixels(ilp_path, lane)
    if superpixels is None:
        print(f"  No cached superpixels for lane {lane}; skipping.")
        return None
    print(f"  Superpixels shape: {superpixels.shape}, "
          f"range: [{superpixels.min()}, {superpixels.max()}], "
          f"n_labels: {len(np.unique(superpixels))}")

    # 2. Load input data.
    info = _read_lane_info(ilp_path, lane)

    if info["raw_data_path"] is None:
        print(f"  Raw Data file not found: {info['raw_data_orig']}")
        print(f"  Skipping lane {lane}.")
        return None
    if info["probs_path"] is None:
        print(f"  Probabilities file not found: {info['probs_orig']}")
        print(f"  Skipping lane {lane}.")
        return None

    print(f"  Raw Data:      {info['raw_data_path']}")
    print(f"  Probabilities: {info['probs_path']}"
          f" (channel {info['channel_selection']})")

    raw_data = _load_volume(info["raw_data_path"], info["raw_data_internal"])
    probs_data = _load_volume(info["probs_path"], info["probs_internal"])
    print(f"  Raw Data shape: {raw_data.shape}  dtype: {raw_data.dtype}")
    print(f"  Probabilities shape: {probs_data.shape}  dtype: {probs_data.dtype}")

    # Sanity check shapes.
    sp_shape = superpixels.shape
    raw_shape = raw_data.shape[:3]
    if sp_shape != raw_shape:
        print(f"  WARNING: shape mismatch: superpixels {sp_shape} vs raw {raw_shape}")

    # 3. Build channel data map.
    channel_data = _build_channel_map(
        feature_names, raw_data, probs_data, info["channel_selection"]
    )

    # 4. Compute features with blimp.
    from multicut_from_ilp import compute_ilastikrag_features

    print(f"  Computing features with blimp ...")
    features, edge_ids = compute_ilastikrag_features(
        superpixels, channel_data, feature_names
    )
    if features is None:
        print(f"  compute_ilastikrag_features returned None (< 2 superpixels?)")
        return None
    print(f"  Blimp features: {features.shape[0]} edges, {features.shape[1]} columns")

    # 5. Load cached EdgeFeatures.
    try:
        ilp_df = read_edge_features(ilp_path, lane=lane)
    except KeyError:
        print(f"  No cached EdgeFeatures for lane {lane}; skipping comparison.")
        return None
    ilp_feat_cols = [c for c in ilp_df.columns if c not in ("sp1", "sp2")]
    print(f"  ilastik features: {len(ilp_df)} edges, {len(ilp_feat_cols)} columns")

    # 6. Compare.
    result = _compare_features(ilp_df, features, edge_ids, feature_names, lane)
    _print_comparison(result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate blimp feature computation against ilastik cached features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
                python scripts/validate_features.py project.ilp
                python scripts/validate_features.py project.ilp --lanes 0 1
                python scripts/validate_features.py project.ilp --rf-comparison
        """),
    )
    parser.add_argument("ilp", help="Path to the ilastik .ilp project file")
    parser.add_argument(
        "--lanes", type=int, nargs="*", default=None,
        help="Lane indices to validate (default: all lanes with labels)",
    )
    parser.add_argument(
        "--rf-comparison", action="store_true",
        help="Also compare RF predictions on cached features",
    )
    args = parser.parse_args()

    ilp_path = args.ilp
    print(f"ilastik project: {ilp_path}")

    feature_names = read_feature_names(ilp_path)
    print(f"\nFeature channels:")
    for ch, feats in feature_names.items():
        print(f"  {ch}: {feats}")

    # Discover available lanes.
    labeled_lanes = discover_lanes(ilp_path)
    print(f"\nLanes with labels: {labeled_lanes}")

    # Also discover lanes in Input Data (may differ from labeled lanes).
    with _open_ilp_file(ilp_path) as f:
        input_lanes = sorted(
            int(k[len("lane"):])
            for k in f["Input Data/infos"].keys()
            if k.startswith("lane")
        )
    print(f"Lanes in Input Data: {input_lanes}")

    lanes_to_check = args.lanes if args.lanes is not None else labeled_lanes
    if not lanes_to_check:
        print("No lanes to validate.")
        return 1

    # Validate each lane.
    results = []
    n_ok = 0
    n_skip = 0
    for lane in lanes_to_check:
        result = validate_lane(ilp_path, lane, feature_names)
        if result is None:
            n_skip += 1
        else:
            results.append(result)
            if result.get("all_exact"):
                n_ok += 1

    # Summary.
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  Lanes checked: {len(lanes_to_check)}")
    print(f"  Skipped (missing data): {n_skip}")
    print(f"  Exact match: {n_ok}")
    print(f"  With differences: {len(results) - n_ok}")

    for r in results:
        if "column_mismatch" in r:
            print(f"\n  Lane {r['lane']}: COLUMN MISMATCH — "
                  f"ilastik has {len(r['ilp_feature_cols'])} cols, "
                  f"blimp has {r['blimp_n_features']} cols")
        elif "error" in r:
            print(f"\n  Lane {r['lane']}: ERROR — {r['error']}")
        elif not r.get("all_exact"):
            mismatched = [
                f["column"] for f in r["per_feature"] if not f["exact_match"]
            ]
            print(f"\n  Lane {r['lane']}: {len(mismatched)} features differ:")
            for col in mismatched:
                print(f"    - {col}")

    # Optional RF comparison.
    if args.rf_comparison:
        _compare_rf_predictions(ilp_path, feature_names)

    any_diff = any(not r.get("all_exact") for r in results)
    return 1 if any_diff else 0


if __name__ == "__main__":
    sys.exit(main())
