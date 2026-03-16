"""
ilp-mc-block  –  single-command ilastik multicut pipeline

Fits the sklearn classifier from the .ilp training data, then immediately runs
the blockwise lazy multicut on the provided raw data and boundary probabilities.
All outputs land in --output-dir:

    rf.pkl                          sklearn random forest classifier
    <raw_stem>_segmentation.zarr    final segmentation (uint64, zyx)
    <raw_stem>_watershed.zarr       watershed superpixels
    params.json                     exact call parameters for reproducibility

Pass --no-keep-watershed to delete the watershed after the run.
Or pass ws-zarr to point to a precomputed watershed zarr.

Usage
-----
    ilp-mc-block \\
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

from fit_classifier import fit_rf_from_ilp
from ilp_reader import read_feature_names, read_wsdt_params
from multicut_from_ilp import _find_boundary_channel, _find_raw_channel, _build_channel_spec, _run_lazy


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
        "--max-block-shape", type=int, nargs=3, default=[256, 256, 256],
        metavar=("Z", "Y", "X"),
        help="Maximum block shape; actual shape may be slightly smaller to satisfy "
             "checkerboard requirements (default: 256 256 256)",
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
        "--n-estimators", type=int, default=100,
        help="Number of trees in the random forest (default: 100)",
    )

    # Watershed method
    parser.add_argument(
        "--ws-method",
        choices=["ilastik", "two-pass", "2d"],
        default=None,
        help=(
            "Watershed algorithm to use.  "
            "``ilastik`` (default): mirrors ilastik's parallel_watershed — "
            "128³ blocks with 10-voxel halo, hard block boundaries, "
            "vigra.labelMultiArray per block, cumulative offsets.  Produces "
            "pixel-identical superpixels to ilastik when the same boundary map "
            "and parameters are used.  "
            "``two-pass``: elf checkerboard two-pass watershed (previous default; "
            "uses --max-block-shape and --halo).  "
            "``2d``: stacked 2-D watershed, recommended for strongly anisotropic data.  "
            "When omitted, ``ilastik`` is used for projects with BlockwiseWatershed=True "
            "(all recent projects), ``two-pass`` for older projects that stored "
            "BlockwiseWatershed=False."
        ),
    )
    parser.add_argument(
        "--ws-threshold", type=float, default=None,
        help="Boundary probability threshold for watershed seeds.  "
             "Defaults to the value stored in the .ilp (or 0.5 if absent).",
    )
    parser.add_argument(
        "--ws-sigma", type=float, default=None,
        help="Gaussian smoothing sigma applied to the watershed seed and weight maps.  "
             "Defaults to the value stored in the .ilp (or 3.0 if absent).",
    )
    parser.add_argument(
        "--ws-min-size", type=int, default=None,
        help="Minimum superpixel size in pixels; smaller segments are merged.  "
             "Defaults to the value stored in the .ilp (or 100 if absent).",
    )
    parser.add_argument(
        "--ws-alpha", type=float, default=None,
        help="Blend factor (0–1) between the boundary map and the distance transform.  "
             "Defaults to the value stored in the .ilp (or 0.9 if absent).",
    )
    parser.add_argument(
        "--ws-invert", action="store_true", default=False,
        help="Invert the boundary probability map (p → 1-p) before running the watershed.  "
             "Equivalent to the 'Invert pixel probabilities' checkbox in ilastik's DT "
             "Watershed applet.  This setting is not stored in the .ilp file, so it must "
             "be set explicitly when needed.",
    )
    parser.add_argument(
        "--solver", default="kernighan-lin",
        choices=["kernighan-lin", "greedy-additive", "greedy-fixation"],
        help="Multicut internal solver (default: kernighan-lin)",
    )

    # Watershed reuse
    parser.add_argument(
        "--ws-zarr", default=None, metavar="PATH",
        help=(
            "Path to a pre-computed watershed zarr.  If supplied and valid the "
            "watershed step is skipped entirely — useful for re-running only the "
            "multicut with different parameters.  Implies --keep-watershed."
        ),
    )
    parser.add_argument(
        "--keep-watershed", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Keep the watershed zarr after the run (default: keep it).  "
            "Pass --no-keep-watershed to delete it.  "
            "The zarr is written to <output-dir>/<raw-stem>_watershed.zarr "
            "and can be passed to --ws-zarr on a subsequent run."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Read DT Watershed parameters from the .ilp; CLI flags override them.
    # -----------------------------------------------------------------------
    ilp_ws = read_wsdt_params(args.ilp)
    ws_threshold    = args.ws_threshold if args.ws_threshold is not None else ilp_ws["threshold"]
    ws_sigma        = args.ws_sigma     if args.ws_sigma     is not None else ilp_ws["sigma"]
    ws_min_size     = args.ws_min_size  if args.ws_min_size  is not None else ilp_ws["min_size"]
    ws_alpha        = args.ws_alpha     if args.ws_alpha     is not None else ilp_ws["alpha"]
    ws_invert       = args.ws_invert    # not stored in .ilp; always explicit
    ws_pixel_pitch  = ilp_ws["pixel_pitch"]   # not overridable via CLI for now
    ws_apply_nonmax = False                    # ApplyNonmaxSuppression; not serialised

    # Choose watershed method.  If the user didn't specify one explicitly,
    # default to "ilastik" for modern projects (BlockwiseWatershed=True) and
    # warn + fall back to "two-pass" for old ones (BlockwiseWatershed=False),
    # because those ran on the full crop at once — something we can't replicate
    # blockwise without loading the entire volume into RAM.
    if args.ws_method is not None:
        ws_method = args.ws_method
    elif ilp_ws["blockwise"]:
        ws_method = "ilastik"
    else:
        warnings.warn(
            "The .ilp was saved with BlockwiseWatershed=False (an old project). "
            "ilastik ran the watershed on the full training crop at once, which we "
            "cannot replicate blockwise.  Falling back to 'two-pass'.  Pass "
            "--ws-method explicitly to suppress this warning.",
            stacklevel=1,
        )
        ws_method = "two-pass"

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
        "n_estimators":   args.n_estimators,
        "ws_method":      ws_method,
        "ws_threshold":   ws_threshold,
        "ws_sigma":       ws_sigma,
        "ws_min_size":    ws_min_size,
        "ws_alpha":       ws_alpha,
        "ws_pixel_pitch": ws_pixel_pitch,
        "ws_apply_nonmax": ws_apply_nonmax,
        "ws_invert":      ws_invert,
        "solver":         args.solver,
        "ws_zarr":        ws_zarr_path,
        "keep_watershed": keep_watershed,
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
    print(f"  Watershed method   : {ws_method}")
    print(f"  Watershed parameters (from .ilp unless overridden):")
    print(f"    threshold   : {ws_threshold}")
    print(f"    sigma       : {ws_sigma}")
    print(f"    min_size    : {ws_min_size}")
    print(f"    alpha       : {ws_alpha}")
    print(f"    pixel_pitch : {ws_pixel_pitch}")
    print(f"    invert      : {ws_invert}")
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
        ws_method=ws_method,
        ws_threshold=ws_threshold,
        ws_sigma=ws_sigma,
        ws_min_size=ws_min_size,
        ws_alpha=ws_alpha,
        ws_pixel_pitch=ws_pixel_pitch,
        ws_apply_nonmax=ws_apply_nonmax,
        ws_invert=ws_invert,
        ws_zarr_path=ws_zarr_path,
        keep_watershed=keep_watershed,
    )

    print("\n=== Done ===")
    print(f"Segmentation : {seg_zarr}")
    print(f"Params       : {params_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
