"""Shared CLI parameter definitions for blimp commands.

Parameter groups are lists of (flag, kwargs) tuples that can be added to any
argparse.ArgumentParser via the ``add_*_args`` helpers.
"""

import argparse

# ---------------------------------------------------------------------------
# Watershed parameters (needed by both blimp and blimp-watershed)
# ---------------------------------------------------------------------------
WATERSHED_PARAMS = [
    ("--raw", dict(
        default=None, metavar="PATH",
        help="Raw data volume (zarr or h5 with a single dataset).  "
             "When omitted, all Raw Data lanes are read from the .ilp project file.",
    )),
    ("--probabilities", dict(
        default=None, metavar="PATH",
        help="Boundary probability volume (zarr or h5 with a single dataset).  "
             "When omitted, all Probabilities lanes are read from the .ilp project file.",
    )),
    ("--axes", dict(
        default=None, metavar="AXES",
        help=(
            "Hard override for input array axes, e.g. 'zyx', 'zyxc', or 'cxyz'.  "
            "When omitted, vigra axistags from the array/dataset attrs are used "
            "if present; otherwise 3-D inputs are treated as zyx and 4-D inputs "
            "as zyxc for backwards compatibility."
        ),
    )),
    (("--channel-index", "--channel_index"), dict(
        dest="channel_index", type=int, default=None, metavar="N",
        help=(
            "Select channel N from the input channel axis.  Requires --axes or "
            "vigra axistags metadata so the channel axis can be identified.  "
            "Without this option, inputs with a channel axis must have exactly "
            "one channel."
        ),
    )),
    ("--output-dir", dict(
        default=None, metavar="DIR",
        help="Directory for all outputs (created if it does not exist).  "
             "Defaults to a 'blimp-output' directory next to the .ilp file.",
    )),
    ("--ws-method", dict(
        choices=["ilastik", "two-pass", "2d"],
        default=None,
        help=(
            "Watershed algorithm to use.  "
            "``ilastik`` (default): mirrors ilastik's parallel_watershed — "
            "128³ blocks with 10-voxel halo, hard block boundaries, "
            "vigra.labelMultiArray per block, cumulative offsets.  Produces "
            "pixel-identical superpixels to ilastik when the same boundary map "
            "and parameters are used.  "
            "``two-pass (experimental)``: elf checkerboard two-pass watershed "
            "(uses --max-block-shape and --halo).  "
            "``2d (experimental)``: stacked 2-D watershed, for strongly "
            "anisotropic data.  "
            "When omitted and --ilp is given, ``ilastik`` is used for projects "
            "with BlockwiseWatershed=True (all recent projects), ``two-pass`` "
            "for older projects.  Without --ilp, defaults to ``ilastik``."
        ),
    )),
    ("--ws-threshold", dict(
        type=float, default=None,
        help="Boundary probability threshold for watershed seeds.  "
             "Defaults to the value stored in the .ilp (or 0.5 if absent).",
    )),
    ("--ws-sigma", dict(
        type=float, default=None,
        help="Gaussian smoothing sigma applied to the watershed seed and weight maps.  "
             "Defaults to the value stored in the .ilp (or 3.0 if absent).",
    )),
    ("--ws-min-size", dict(
        type=int, default=None,
        help="Minimum superpixel size in pixels; smaller segments are merged.  "
             "Defaults to the value stored in the .ilp (or 100 if absent).",
    )),
    ("--ws-alpha", dict(
        type=float, default=None,
        help="Blend factor (0–1) between the boundary map and the distance transform.  "
             "Defaults to the value stored in the .ilp (or 0.9 if absent).",
    )),
    ("--ws-invert", dict(
        action="store_true", default=False,
        help="Invert the boundary probability map (p → 1-p) before running the "
             "watershed.  Use this when the probability file stores "
             "P(background) / P(interior) (high = interior) instead of "
             "P(boundary) (high = membrane).",
    )),
    ("--ws-zarr", dict(
        default=None, metavar="PATH",
        help=(
            "Path to a pre-computed watershed zarr.  If supplied and valid the "
            "watershed step is skipped entirely — useful for re-running with "
            "different parameters.  Implies --keep-watershed."
        ),
    )),
    ("--keep-watershed", dict(
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Keep the watershed zarr after the run (default: keep it).  "
            "Pass --no-keep-watershed to delete it.  "
            "The zarr is written to <output-dir>/<raw-stem>_watershed.zarr "
            "and can be passed to --ws-zarr on a subsequent run."
        ),
    )),
]

# ---------------------------------------------------------------------------
# Blockwise parameters (shared block-shape, halo, threads)
# ---------------------------------------------------------------------------
BLOCKWISE_PARAMS = [
    ("--max-block-shape", dict(
        type=int, nargs=3, default=[256, 256, 256],
        metavar=("Z", "Y", "X"),
        help="Maximum block shape; ws-method=two-pass may reduce this to satisfy "
             "checkerboard requirements (default: 256 256 256)",
    )),
    ("--halo", dict(
        type=int, nargs=3, default=[32, 32, 32],
        metavar=("Z", "Y", "X"),
        help="Halo (overlap) around each block (default: 32 32 32)",
    )),
    ("--threads", dict(
        type=int, default=8,
        help="Number of parallel threads for watershed and multicut (default: 8)",
    )),
]


def add_watershed_args(parser: argparse.ArgumentParser) -> None:
    """Add all watershed-related arguments to *parser*."""
    for flag, kwargs in WATERSHED_PARAMS:
        if isinstance(flag, (tuple, list)):
            parser.add_argument(*flag, **kwargs)
        else:
            parser.add_argument(flag, **kwargs)


def add_blockwise_args(parser: argparse.ArgumentParser) -> None:
    """Add blockwise processing arguments to *parser*."""
    for flag, kwargs in BLOCKWISE_PARAMS:
        parser.add_argument(flag, **kwargs)
