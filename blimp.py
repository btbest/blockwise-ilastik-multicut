"""
blimp  –  single-command ilastik multicut pipeline

Loads the edge classifier from the .ilp (or re-fits one from training data),
then immediately runs the blockwise lazy multicut on the provided raw data
and boundary probabilities.  All outputs land in --output-dir:

    rf.pkl                          classifier pickle (only if --classifier-source sklearn)
    <raw_stem>_segmentation.zarr    final segmentation (uint64, zyx)
    <raw_stem>_watershed.zarr       watershed superpixels
    params.json                     exact call parameters for reproducibility

Pass --no-keep-watershed to delete the watershed after the run.
Or pass ws-zarr to point to a precomputed watershed zarr.

Usage
-----
    # Minimal: run on all lanes from the .ilp project
    blimp --ilp my_project.ilp

    # Explicit data paths (single lane):
    blimp \\
        --ilp my_project.ilp \\
        --raw raw.zarr \\
        --probabilities boundaries.zarr \\
        --output-dir results/

Input formats
-------------
Both --raw and --probabilities accept local zarr stores and HDF5 files:

    /path/to/file.zarr           local zarr store
    /path/to/file.h5             HDF5 file (must contain exactly one dataset)
    C:\\Users\\...\\file.h5      Windows absolute paths are also supported

Input axes are read from vigra axistags when present.  Use --input-axes to
override or provide missing metadata, and --channel-index to select one channel
from a multi-channel input.  Internally, data is presented as zyx.  Both inputs
must resolve to the same zyx shape.

When --raw and --probabilities are omitted, all Raw Data + Probabilities
lane pairs are read from the .ilp project file's Input Data group.
Output defaults to a ``blimp-output/`` directory next to the .ilp file.
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path


def _run_one_lane(
    *,
    ilp_path,
    raw_path,
    prob_path,
    out,
    rf,
    args,
    ws,
    mc,
    lane_index,
    n_lanes,
    read_feature_names,
    _find_raw_channel,
    _find_boundary_channel,
    _build_channel_spec,
    _run_lazy,
):
    """Run the full multicut pipeline on a single raw+probabilities pair."""
    from _cli_helpers import data_stem

    prefix = f"[lane {lane_index + 1}/{n_lanes}] " if n_lanes > 1 else ""

    raw_stem = data_stem(raw_path)

    seg_zarr = str(out / f"{raw_stem}_segmentation.zarr")
    default_ws = str(out / f"{raw_stem}_watershed.zarr")

    if args.ws_zarr:
        ws_zarr_path = args.ws_zarr
        keep_watershed = True
    else:
        ws_zarr_path = default_ws
        keep_watershed = args.keep_watershed

    # --- Save call parameters for reproducibility ---
    params = {
        "ilp":            ilp_path,
        "raw":            raw_path,
        "probabilities":  prob_path,
        "input_axes":     args.input_axes,
        "channel_index":  args.channel_index,
        "output_dir":     str(out.resolve()),
        "max_block_shape": args.max_block_shape,
        "halo":           args.halo,
        "threads":        args.threads,
        "classifier_source": args.classifier_source,
        "n_estimators":   args.n_estimators,
        "ws_method":      ws["ws_method"],
        "ws_threshold":   ws["ws_threshold"],
        "ws_sigma":       ws["ws_sigma"],
        "ws_min_size":    ws["ws_min_size"],
        "ws_alpha":       ws["ws_alpha"],
        "ws_pixel_pitch": ws["ws_pixel_pitch"],
        "ws_apply_nonmax": ws["ws_apply_nonmax"],
        "ws_invert":      ws["ws_invert"],
        "solver":         args.solver,
        "ws_zarr":        ws_zarr_path,
        "keep_watershed": keep_watershed,
        "mc_beta":        mc["mc_beta"],
        "mc_threshold":   mc["mc_threshold"],
    }
    params_file = out / f"{raw_stem}_params.json" if n_lanes > 1 else out / "params.json"
    params_file.write_text(json.dumps(params, indent=2) + "\n")
    print(f"{prefix}Parameters written to {params_file}")

    # --- Map channels ---
    print(f"\n{prefix}=== Mapping channels ===")
    feature_names = read_feature_names(ilp_path)
    raw_channel = _find_raw_channel(feature_names)
    boundary_channel = _find_boundary_channel(feature_names)
    print(f"  Raw channel      : {raw_channel!r}  →  {raw_path}")
    print(f"  Boundary channel : {boundary_channel!r}  →  {prob_path}")

    channel_specs = [
        _build_channel_spec(boundary_channel, prob_path),
        _build_channel_spec(raw_channel, raw_path),
    ]

    # --- Run blockwise lazy multicut ---
    print(f"\n{prefix}=== Running blockwise multicut ===")
    print(f"  Watershed method   : {ws['ws_method']}")
    print(f"  Watershed parameters (from .ilp unless overridden):")
    print(f"    threshold   : {ws['ws_threshold']}")
    print(f"    sigma       : {ws['ws_sigma']}")
    print(f"    min_size    : {ws['ws_min_size']}")
    print(f"    alpha       : {ws['ws_alpha']}")
    print(f"    pixel_pitch : {ws['ws_pixel_pitch']}")
    print(f"    invert      : {ws['ws_invert']}")
    print(f"  Multicut parameters (from .ilp unless overridden):")
    print(f"    beta        : {mc['mc_beta']}")
    print(f"    threshold   : {mc['mc_threshold']}")
    _run_lazy(
        ilp_path=ilp_path,
        rf=rf,
        channel_specs=channel_specs,
        output_zarr_path=seg_zarr,
        output_zarr_key="seg",
        block_shape=tuple(args.max_block_shape),
        halo=list(args.halo),
        internal_solver=args.solver,
        n_threads=args.threads,
        ws_method=ws["ws_method"],
        ws_threshold=ws["ws_threshold"],
        ws_sigma=ws["ws_sigma"],
        ws_min_size=ws["ws_min_size"],
        ws_alpha=ws["ws_alpha"],
        ws_pixel_pitch=ws["ws_pixel_pitch"],
        ws_apply_nonmax=ws["ws_apply_nonmax"],
        ws_invert=ws["ws_invert"],
        ws_zarr_path=ws_zarr_path,
        keep_watershed=keep_watershed,
        mc_beta=mc["mc_beta"],
        mc_threshold=mc["mc_threshold"],
        input_axes=args.input_axes,
        channel_index=args.channel_index,
    )

    print(f"\n{prefix}Segmentation : {seg_zarr}")
    print(f"{prefix}Params       : {params_file}")


def main():
    parser = argparse.ArgumentParser(
        prog="blimp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared watershed + blockwise parameters
    from _cli_params import add_blockwise_args, add_watershed_args
    add_watershed_args(parser)
    add_blockwise_args(parser)

    # .ilp is required for the full multicut pipeline
    parser.add_argument(
        "--ilp", required=True, metavar="PATH",
        help="Ilastik .ilp project file",
    )

    # Multicut-specific parameters
    parser.add_argument(
        "--mc-beta", type=float, default=0.5,
        help="Multicut edge-cost bias: <0.5 merges more, >0.5 splits more (default: 0.5)",
    )
    parser.add_argument(
        "--mc-threshold", type=float, default=0.5,
        help="Multicut edge probability threshold: Edges above threshold are cut (default: 0.5)",
    )
    parser.add_argument(
        "--classifier-source",
        choices=["ilp", "sklearn"],
        default="ilp",
        help=(
            "Where to get the edge classifier.  'ilp' (default) extracts the "
            "already-trained vigra RF from the .ilp (identical to ilastik's own "
            "predictions).  'sklearn' re-fits a new sklearn RF from the cached "
            "training data."
        ),
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of trees in the random forest (only used with --classifier-source sklearn; default: 100)",
    )
    parser.add_argument(
        "--solver", default="kernighan-lin",
        choices=["kernighan-lin", "greedy-additive", "greedy-fixation"],
        help="Multicut internal solver (default: kernighan-lin)",
    )

    args = parser.parse_args()

    # Lazy imports: only load heavy modules after argument parsing succeeds
    from _cli_helpers import resolve_lane_pairs, resolve_watershed_params, resolve_mc_params
    from fit_classifier import extract_vigra_rf_from_ilp, fit_rf_from_ilp
    from ilp_reader import read_feature_names
    from multicut_from_ilp import _find_boundary_channel, _find_raw_channel, _build_channel_spec, _run_lazy

    # -----------------------------------------------------------------------
    # Resolve lane pairs and output directory
    # -----------------------------------------------------------------------
    pairs, out = resolve_lane_pairs(args, ilp_path=args.ilp)
    out.mkdir(parents=True, exist_ok=True)

    n_lanes = len(pairs)
    if n_lanes > 1:
        print(f"Found {n_lanes} lane pairs in the .ilp project file:")
        for i, p in enumerate(pairs):
            print(f"  Lane {i + 1}: raw={p['raw']}")
            print(f"           prob={p['probabilities']}")

    # -----------------------------------------------------------------------
    # Resolve watershed parameters (CLI flags override .ilp values)
    # -----------------------------------------------------------------------
    ws = resolve_watershed_params(args, ilp_path=args.ilp)

    # -----------------------------------------------------------------------
    # Step 1: Load or fit the edge classifier (once for all lanes)
    # -----------------------------------------------------------------------
    print("\n=== Step 1: Loading classifier ===")
    if args.classifier_source == "ilp":
        rf = extract_vigra_rf_from_ilp(args.ilp)
    else:
        rf = fit_rf_from_ilp(
            args.ilp,
            n_estimators=args.n_estimators,
            n_jobs=args.threads,
        )
        rf_pkl = str(out / "rf.pkl")
        with open(rf_pkl, "wb") as fh:
            pickle.dump(rf, fh)
        print(f"Classifier saved to {rf_pkl}")

    # -----------------------------------------------------------------------
    # Resolve multicut parameters (CLI flags override .ilp values)
    # -----------------------------------------------------------------------
    mc = resolve_mc_params(args, ilp_path=args.ilp)

    # -----------------------------------------------------------------------
    # Step 2+3: For each lane pair, map channels and run multicut
    # -----------------------------------------------------------------------
    for i, pair in enumerate(pairs):
        _run_one_lane(
            ilp_path=args.ilp,
            raw_path=pair["raw"],
            prob_path=pair["probabilities"],
            out=out,
            rf=rf,
            args=args,
            ws=ws,
            mc=mc,
            lane_index=i,
            n_lanes=n_lanes,
            read_feature_names=read_feature_names,
            _find_raw_channel=_find_raw_channel,
            _find_boundary_channel=_find_boundary_channel,
            _build_channel_spec=_build_channel_spec,
            _run_lazy=_run_lazy,
        )

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
