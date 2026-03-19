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

Volumes must be in zyx(c) axis order.  Both inputs must have the same shape.
Singleton channel axis is accepted (ignored).
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path


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
        "--beta", type=float, default=0.5,
        help="Multicut edge-cost bias: <0.5 merges more, >0.5 splits more (default: 0.5)",
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
    from _cli_helpers import resolve_watershed_params
    from fit_classifier import extract_vigra_rf_from_ilp, fit_rf_from_ilp
    from ilp_reader import read_feature_names
    from multicut_from_ilp import _find_boundary_channel, _find_raw_channel, _build_channel_spec, _run_lazy

    # -----------------------------------------------------------------------
    # Resolve watershed parameters (CLI flags override .ilp values)
    # -----------------------------------------------------------------------
    ws = resolve_watershed_params(args, ilp_path=args.ilp)

    # -----------------------------------------------------------------------
    # Setup output directory and output paths
    # -----------------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_stem = Path(args.raw).stem          # e.g. "my_raw" from "my_raw.zarr"

    seg_zarr     = str(out / f"{raw_stem}_segmentation.zarr")
    rf_pkl       = str(out / "rf.pkl")
    default_ws   = str(out / f"{raw_stem}_watershed.zarr")

    # If the user supplied --ws-zarr, use it as-is and never delete it.
    # Otherwise use the default path and honour --keep-watershed.
    if args.ws_zarr:
        ws_zarr_path    = args.ws_zarr
        keep_watershed  = True   # never delete a user-supplied watershed
    else:
        ws_zarr_path    = default_ws
        keep_watershed  = args.keep_watershed

    # -----------------------------------------------------------------------
    # Save call parameters for reproducibility
    # -----------------------------------------------------------------------
    params = {
        "ilp":            args.ilp,
        "raw":            args.raw,
        "probabilities":  args.probabilities,
        "output_dir":     str(out.resolve()),
        "max_block_shape": args.max_block_shape,
        "halo":           args.halo,
        "beta":           args.beta,
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
    }
    params_file = out / "params.json"
    params_file.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Parameters written to {params_file}")

    # -----------------------------------------------------------------------
    # Step 1: Load or fit the edge classifier
    # -----------------------------------------------------------------------
    print("\n=== Step 1/3: Loading classifier ===")
    if args.classifier_source == "ilp":
        rf = extract_vigra_rf_from_ilp(args.ilp)
    else:
        rf = fit_rf_from_ilp(
            args.ilp,
            n_estimators=args.n_estimators,
            n_jobs=args.threads,
        )
        # Only pickle the sklearn classifier; vigra one is already in the .ilp
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

    # Boundary channel must appear first in channel_specs so _run_lazy finds it
    # via _find_boundary_channel (order in the specs list does not matter for
    # feature computation, but placing it first is conventional).
    channel_specs = [
        _build_channel_spec(boundary_channel, args.probabilities),
        _build_channel_spec(raw_channel,      args.raw),
    ]

    # -----------------------------------------------------------------------
    # Step 3: Run blockwise lazy multicut
    # -----------------------------------------------------------------------
    print("\n=== Step 3/3: Running blockwise multicut ===")
    print(f"  Watershed method   : {ws['ws_method']}")
    print(f"  Watershed parameters (from .ilp unless overridden):")
    print(f"    threshold   : {ws['ws_threshold']}")
    print(f"    sigma       : {ws['ws_sigma']}")
    print(f"    min_size    : {ws['ws_min_size']}")
    print(f"    alpha       : {ws['ws_alpha']}")
    print(f"    pixel_pitch : {ws['ws_pixel_pitch']}")
    print(f"    invert      : {ws['ws_invert']}")
    _run_lazy(
        ilp_path=args.ilp,
        rf=rf,
        channel_specs=channel_specs,
        output_zarr_path=seg_zarr,
        output_zarr_key="seg",
        beta=args.beta,
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
    )

    print("\n=== Done ===")
    print(f"Segmentation : {seg_zarr}")
    print(f"Params       : {params_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
