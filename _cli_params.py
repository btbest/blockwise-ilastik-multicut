"""
Shared CLI parameter definitions and resolution helpers for blimp commands.

Used by both blimp.py (full multicut pipeline) and blimp_watershed.py
(watershed-only computation) to avoid duplication of argument definitions
and parameter merging logic.
"""

import argparse
import warnings


def add_blockwise_args(parser: argparse.ArgumentParser):
    """Add blockwise computation parameters (block shape, halo, threads).

    These are shared by both the main blimp command (for multicut) and the
    blimp-watershed command (for watershed only).
    """
    parser.add_argument(
        "--max-block-shape", type=int, nargs=3, default=[256, 256, 256],
        metavar=("Z", "Y", "X"),
        help="Maximum block shape; ws-method=two-pass may reduce this to satisfy "
             "checkerboard requirements (default: 256 256 256)",
    )
    parser.add_argument(
        "--halo", type=int, nargs=3, default=[32, 32, 32],
        metavar=("Z", "Y", "X"),
        help="Halo (overlap) around each block (default: 32 32 32)",
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Number of parallel threads for watershed and multicut (default: 8)",
    )


def add_ws_args(parser: argparse.ArgumentParser):
    """Add watershed-specific parameters.

    Includes watershed method selection, threshold/sigma/min-size/alpha,
    probability inversion, and zarr reuse options.

    For the watershed-only command (blimp-watershed), the threshold/sigma/
    min-size/alpha parameters must be supplied either via the ILP file OR
    explicitly via CLI. For the main command (blimp), they have ILP defaults.
    """
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
            "``two-pass (experimental)``: elf checkerboard two-pass watershed (uses --max-block-shape and --halo).  "
            "``2d (experimental)``: stacked 2-D watershed, for strongly anisotropic data.  "
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
             "Use this when the probability file stores P(background) / P(interior) "
             "(high = interior) instead of P(boundary) (high = membrane).  "
             "elf's distance_transform_watershed expects high = boundary; this flag "
             "flips the values so that convention is met.  "
             "Equivalent to the 'Invert pixel probabilities' checkbox in ilastik's DT "
             "Watershed applet.  This setting is not stored in the .ilp file, so it must "
             "be set explicitly when needed.",
    )
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


def resolve_ws_params(args, ilp_ws: dict):
    """
    Merge CLI arguments with ILP watershed defaults.

    Returns a SimpleNamespace with resolved watershed parameters:
        ws_threshold, ws_sigma, ws_min_size, ws_alpha,
        ws_pixel_pitch, ws_apply_nonmax, ws_invert, ws_method

    Also handles the old-project warning for BlockwiseWatershed=False.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (must include ws_threshold, ws_sigma,
        ws_min_size, ws_alpha, ws_invert, ws_method from add_ws_args).
    ilp_ws : dict
        Watershed parameters from read_wsdt_params() or hardcoded defaults:
        {
            "threshold": float,
            "min_size": int,
            "sigma": float,
            "alpha": float,
            "pixel_pitch": list | None,
            "blockwise": bool,
        }

    Returns
    -------
    argparse.Namespace with fields:
        ws_threshold, ws_sigma, ws_min_size, ws_alpha,
        ws_pixel_pitch, ws_apply_nonmax, ws_invert, ws_method
    """
    # CLI args override ILP defaults
    ws_threshold = args.ws_threshold if args.ws_threshold is not None else ilp_ws["threshold"]
    ws_sigma = args.ws_sigma if args.ws_sigma is not None else ilp_ws["sigma"]
    ws_min_size = args.ws_min_size if args.ws_min_size is not None else ilp_ws["min_size"]
    ws_alpha = args.ws_alpha if args.ws_alpha is not None else ilp_ws["alpha"]
    ws_invert = args.ws_invert  # not stored in .ilp; always explicit
    ws_pixel_pitch = ilp_ws["pixel_pitch"]  # not overridable via CLI for now
    ws_apply_nonmax = False  # ApplyNonmaxSuppression; not serialised

    # Choose watershed method. If the user didn't specify one explicitly,
    # default to "ilastik" for modern projects (BlockwiseWatershed=True) and
    # warn + fall back to "two-pass" for old ones (BlockwiseWatershed=False).
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
            stacklevel=2,
        )
        ws_method = "two-pass"

    return argparse.Namespace(
        ws_threshold=ws_threshold,
        ws_sigma=ws_sigma,
        ws_min_size=ws_min_size,
        ws_alpha=ws_alpha,
        ws_pixel_pitch=ws_pixel_pitch,
        ws_apply_nonmax=ws_apply_nonmax,
        ws_invert=ws_invert,
        ws_method=ws_method,
    )
