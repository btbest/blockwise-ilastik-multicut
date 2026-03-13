"""
ilp_mc_block  –  single-command ilastik multicut pipeline

Fits the sklearn classifier from the .ilp training data, then immediately runs
the blockwise lazy multicut on the provided raw data and boundary probabilities.
All outputs land in --output-dir:

    rf.pkl                          sklearn classifier (reusable)
    <raw_stem>_segmentation.zarr    final segmentation (uint64, zyx)
    params.json                     exact call parameters for reproducibility

    ws_tmp.dat is written to --output-dir during the run and deleted on success.

Usage
-----
    ilp-mc-block \\
        --ilp my_project.ilp \\
        --raw raw.zarr \\
        --probabilities boundary.zarr \\
        --output-dir results/ \\
        [--block-shape 256 256 256] [--halo 32 32 32] \\
        [--beta 0.5] [--threads 8] [--n-estimators 200] \\
        [--use-2dws] [--ws-threshold 0.5] [--ws-sigma 2.0] \\
        [--solver kernighan-lin]

Input formats
-------------
Both --raw and --probabilities accept local zarr stores and HDF5 files:

    /path/to/file.zarr           local zarr store
    /path/to/file.h5             HDF5 file (must contain exactly one dataset)
    C:\\Users\\...\\file.h5      Windows absolute paths are also supported

Volumes must be in zyx axis order.  Both inputs must have the same shape.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

from fit_classifier import fit_rf_from_ilp
from ilp_reader import read_feature_names
from multicut_from_ilp import _find_boundary_channel, _run_lazy


_URL_SCHEMES = ("http://", "https://", "s3://", "gs://", "ftp://")


def _parse_data_path(spec: str):
    """
    Return (path, None) for any data path spec.

    HDF5 dataset keys are no longer specified via colon notation; the single
    dataset inside the file is auto-detected when it is opened.  This also
    avoids misinterpreting Windows drive-letter colons (e.g. C:\\...) as key
    separators.
    """
    return spec, None


def _find_raw_channel(feature_names: dict) -> str:
    """Return the name of the raw data channel (contains 'raw', case-insensitive)."""
    for name in feature_names:
        if "raw" in name.lower():
            return name
    raise ValueError(
        f"Cannot identify raw data channel in: {list(feature_names)}. "
        "Expected a channel name containing 'raw' (case-insensitive)."
    )


def _build_channel_spec(channel_name: str, path: str, key) -> str:
    return f"{channel_name}:{path}"


def main():
    parser = argparse.ArgumentParser(
        prog="ilp-mc-block",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--ilp", required=True, metavar="PATH",
        help="Ilastik .ilp project file",
    )
    parser.add_argument(
        "--raw", required=True, metavar="PATH",
        help="Raw data volume (zarr or h5 with a single dataset), zyx axis order",
    )
    parser.add_argument(
        "--probabilities", required=True, metavar="PATH",
        help="Boundary probability volume (zarr or h5 with a single dataset), zyx axis order",
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory for all outputs (created if it does not exist)",
    )

    # Blockwise / multicut parameters
    parser.add_argument(
        "--block-shape", type=int, nargs=3, default=[256, 256, 256],
        metavar=("Z", "Y", "X"),
        help="Block shape for blockwise processing (default: 256 256 256)",
    )
    parser.add_argument(
        "--halo", type=int, nargs=3, default=[32, 32, 32],
        metavar=("Z", "Y", "X"),
        help="Halo (overlap) around each block (default: 32 32 32)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="Multicut edge-cost bias: <0.5 merges more, >0.5 splits more (default: 0.5)",
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Number of parallel threads for watershed and multicut (default: 8)",
    )

    # Classifier parameters
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="Number of trees in the random forest (default: 200)",
    )

    # Watershed parameters
    parser.add_argument(
        "--use-2dws", action="store_true",
        help="Use stacked 2D watersheds (recommended for strongly anisotropic data)",
    )
    parser.add_argument("--ws-threshold", type=float, default=0.5)
    parser.add_argument("--ws-sigma", type=float, default=2.0)
    parser.add_argument(
        "--solver", default="kernighan-lin",
        choices=["kernighan-lin", "greedy-additive", "greedy-fixation"],
        help="Multicut internal solver (default: kernighan-lin)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup output directory and output paths
    # -----------------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_path, raw_key = _parse_data_path(args.raw)
    raw_stem = Path(raw_path).stem          # e.g. "my_raw" from "my_raw.zarr"

    seg_zarr = str(out / f"{raw_stem}_segmentation.zarr")
    rf_pkl   = str(out / "rf.pkl")
    ws_tmp   = str(out / "ws_tmp.dat")

    # -----------------------------------------------------------------------
    # Save call parameters for reproducibility
    # -----------------------------------------------------------------------
    params = {
        "ilp":          args.ilp,
        "raw":          args.raw,
        "probabilities": args.probabilities,
        "output_dir":   str(out.resolve()),
        "block_shape":  args.block_shape,
        "halo":         args.halo,
        "beta":         args.beta,
        "threads":      args.threads,
        "n_estimators": args.n_estimators,
        "use_2dws":     args.use_2dws,
        "ws_threshold": args.ws_threshold,
        "ws_sigma":     args.ws_sigma,
        "solver":       args.solver,
    }
    params_file = out / "params.json"
    params_file.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Parameters written to {params_file}")

    # -----------------------------------------------------------------------
    # Step 1: Fit sklearn classifier from the .ilp training data
    # -----------------------------------------------------------------------
    print("\n=== Step 1/3: Fitting classifier ===")
    rf = fit_rf_from_ilp(
        args.ilp,
        n_estimators=args.n_estimators,
        n_jobs=args.threads,
        verbose=True,
    )
    with open(rf_pkl, "wb") as fh:
        pickle.dump(rf, fh)
    print(f"Classifier saved to {rf_pkl}")

    # -----------------------------------------------------------------------
    # Step 2: Map --raw / --probabilities to the ILP channel names
    # -----------------------------------------------------------------------
    print("\n=== Step 2/3: Mapping channels ===")
    feature_names = read_feature_names(args.ilp)
    raw_channel      = _find_raw_channel(feature_names)
    boundary_channel = _find_boundary_channel(feature_names)
    print(f"  Raw channel      : {raw_channel!r}  →  {args.raw}")
    print(f"  Boundary channel : {boundary_channel!r}  →  {args.probabilities}")

    probs_path, probs_key = _parse_data_path(args.probabilities)

    # Boundary channel must appear first in channel_specs so _run_lazy finds it
    # via _find_boundary_channel (order in the specs list does not matter for
    # feature computation, but placing it first is conventional).
    channel_specs = [
        _build_channel_spec(boundary_channel, probs_path, probs_key),
        _build_channel_spec(raw_channel,      raw_path,   raw_key),
    ]

    # -----------------------------------------------------------------------
    # Step 3: Run blockwise lazy multicut
    # -----------------------------------------------------------------------
    print("\n=== Step 3/3: Running blockwise multicut ===")
    _run_lazy(
        ilp_path=args.ilp,
        rf=rf,
        channel_specs=channel_specs,
        output_zarr_path=seg_zarr,
        output_zarr_key="seg",
        beta=args.beta,
        block_shape=tuple(args.block_shape),
        halo=tuple(args.halo),
        internal_solver=args.solver,
        n_threads=args.threads,
        use_2dws=args.use_2dws,
        ws_threshold=args.ws_threshold,
        ws_sigma=args.ws_sigma,
        ws_tmp_path=ws_tmp,
        verbose=True,
    )

    print("\n=== Done ===")
    print(f"Segmentation : {seg_zarr}")
    print(f"Params       : {params_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
