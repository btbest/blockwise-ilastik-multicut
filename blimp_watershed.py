"""
blimp-watershed  –  compute only the blockwise watershed (no multicut)

Power-user entrypoint that runs only the watershed step from the blimp
pipeline.  Produces a watershed zarr that can later be fed to ``blimp``
via ``--ws-zarr`` for multicut with different parameters.

An .ilp file is **optional**: when provided, watershed parameters
(threshold, sigma, min-size, alpha, pixel-pitch, ws-method) are read
from the project as defaults and can be overridden on the command line.
Without --ilp, the four core parameters (--ws-threshold, --ws-sigma,
--ws-min-size, --ws-alpha) must be given explicitly, along with --raw,
--probabilities, and --output-dir.

Usage
-----
    # Minimal: run watershed on all lanes from the .ilp project
    blimp-watershed --ilp my_project.ilp

    # With .ilp (parameters from project, overridable):
    blimp-watershed \\
        --ilp my_project.ilp \\
        --raw raw.zarr \\
        --probabilities boundaries.zarr \\
        --output-dir results/

    # Without .ilp (all parameters explicit):
    blimp-watershed \\
        --raw raw.zarr \\
        --probabilities boundaries.zarr \\
        --output-dir results/ \\
        --ws-threshold 0.5 --ws-sigma 3.0 \\
        --ws-min-size 100 --ws-alpha 0.9

Input formats
-------------
Both --raw and --probabilities accept local zarr stores and HDF5 files:

    /path/to/file.zarr           local zarr store
    /path/to/file.h5             HDF5 file (must contain exactly one dataset)

Input axes are read from vigra axistags when present.  Use --input-axes to
override or provide missing metadata, and --probability-channel-index to select
one boundary probability channel from a multi-channel probability input.
Internally, data is presented as zyx.

When --raw and --probabilities are omitted and --ilp is given, all
Raw Data + Probabilities lane pairs are read from the .ilp project
file's Input Data group.

Output
------
    <output-dir>/<raw-stem>_watershed.zarr    watershed superpixels (uint64, zyx)
    <output-dir>/watershed_params.json        exact call parameters for reproducibility
"""

import argparse
import json
import sys
from pathlib import Path


def _run_one_watershed(
    *,
    raw_path,
    prob_path,
    out,
    args,
    ws,
    ws_zarr_override,
    keep_watershed,
    lane_index,
    n_lanes,
    _open_channel_lazy,
    _Float32LazyArray,
    _InvertedLazyArray,
    _as_zyx_lazy_array,
    _open_or_compute_watershed_zarr,
):
    """Run watershed on a single raw+probabilities pair."""
    from _cli_helpers import data_stem

    prefix = f"[lane {lane_index + 1}/{n_lanes}] " if n_lanes > 1 else ""

    raw_stem = data_stem(raw_path)
    default_ws = str(out / f"{raw_stem}_watershed.zarr")

    if ws_zarr_override:
        ws_zarr_path = ws_zarr_override
        kw = True
    else:
        ws_zarr_path = default_ws
        kw = keep_watershed

    # --- Save call parameters for reproducibility ---
    params = {
        "raw":             raw_path,
        "probabilities":   prob_path,
        "input_axes":      args.input_axes,
        "probability_channel_index": args.probability_channel_index,
        "output_dir":      str(out.resolve()),
        "ilp":             args.ilp,
        "max_block_shape": args.max_block_shape,
        "halo":            args.halo,
        "threads":         args.threads,
        "ws_method":       ws["ws_method"],
        "ws_threshold":    ws["ws_threshold"],
        "ws_sigma":        ws["ws_sigma"],
        "ws_min_size":     ws["ws_min_size"],
        "ws_alpha":        ws["ws_alpha"],
        "ws_pixel_pitch":  ws["ws_pixel_pitch"],
        "ws_apply_nonmax": ws["ws_apply_nonmax"],
        "ws_invert":       ws["ws_invert"],
        "ws_zarr":         ws_zarr_path,
        "keep_watershed":  kw,
    }
    params_file = (
        out / f"{raw_stem}_watershed_params.json" if n_lanes > 1
        else out / "watershed_params.json"
    )
    params_file.write_text(json.dumps(params, indent=2) + "\n")
    print(f"{prefix}Parameters written to {params_file}")

    # --- Open boundary probabilities lazily ---
    print(f"\n{prefix}=== Computing watershed ===")
    print(f"  Watershed method   : {ws['ws_method']}")
    print(f"  Watershed parameters:")
    print(f"    threshold   : {ws['ws_threshold']}")
    print(f"    sigma       : {ws['ws_sigma']}")
    print(f"    min_size    : {ws['ws_min_size']}")
    print(f"    alpha       : {ws['ws_alpha']}")
    print(f"    pixel_pitch : {ws['ws_pixel_pitch']}")
    print(f"    invert      : {ws['ws_invert']}")

    boundary_arr, boundary_fh = _open_channel_lazy(prob_path, None)
    try:
        boundary_arr = _as_zyx_lazy_array(
            boundary_arr,
            input_axes=args.input_axes,
            channel_index=args.probability_channel_index,
            source=prob_path,
        )
        boundary_lazy = _Float32LazyArray(boundary_arr)
        vol_shape = tuple(boundary_lazy.shape)
        print(f"  Volume shape: {vol_shape}")

        ws_input = (
            _InvertedLazyArray(boundary_lazy) if ws["ws_invert"]
            else boundary_lazy
        )

        _, n_superpixels = _open_or_compute_watershed_zarr(
            ws_zarr_path=ws_zarr_path,
            boundary_lazy=ws_input,
            vol_shape=vol_shape,
            block_shape=tuple(args.max_block_shape),
            halo=list(args.halo),
            ws_method=ws["ws_method"],
            ws_threshold=ws["ws_threshold"],
            ws_sigma=ws["ws_sigma"],
            ws_min_size=ws["ws_min_size"],
            ws_alpha=ws["ws_alpha"],
            ws_pixel_pitch=ws["ws_pixel_pitch"],
            ws_apply_nonmax=ws["ws_apply_nonmax"],
            n_threads=args.threads,
        )

        print(f"\n{prefix}Watershed    : {ws_zarr_path}")
        print(f"{prefix}Superpixels  : {n_superpixels or 'Unknown'}")
        print(f"{prefix}Params       : {params_file}")
        print(f"\n{prefix}To use this watershed in a multicut run:")
        ilp_arg = f"--ilp {args.ilp}" if args.ilp else "--ilp <your_project.ilp>"
        print(f"  blimp {ilp_arg} --raw {raw_path} --probabilities {prob_path} --output-dir {args.output_dir or out} --ws-zarr {ws_zarr_path}")

    finally:
        if boundary_fh is not None:
            boundary_fh.close()


def main():
    parser = argparse.ArgumentParser(
        prog="blimp-watershed",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared watershed + blockwise parameters
    from _cli_params import add_blockwise_args, add_watershed_args
    add_watershed_args(parser)
    add_blockwise_args(parser)

    # .ilp is optional for watershed-only mode
    parser.add_argument(
        "--ilp", required=False, default=None, metavar="PATH",
        help="Ilastik .ilp project file (optional).  When provided, watershed "
             "parameters are read from the project as defaults.  Without --ilp "
             "you must supply --raw, --probabilities, --ws-threshold, --ws-sigma, "
             "--ws-min-size, and --ws-alpha explicitly.",
    )

    args = parser.parse_args()

    # --- Validation: without .ilp, core ws params + data paths must be explicit ---
    from _cli_helpers import resolve_watershed_params, validate_watershed_params
    validate_watershed_params(args)

    # --- Resolve lane pairs and output directory ---
    if args.ilp is not None:
        from _cli_helpers import resolve_lane_pairs
        pairs, out = resolve_lane_pairs(args, ilp_path=args.ilp)
    else:
        # No .ilp — --raw, --probabilities already validated
        if args.output_dir is None:
            print(
                "error: --output-dir is required when --ilp is not given.",
                file=sys.stderr,
            )
            sys.exit(2)
        pairs = [{"raw": args.raw, "probabilities": args.probabilities}]
        out = Path(args.output_dir)

    out.mkdir(parents=True, exist_ok=True)
    n_lanes = len(pairs)

    if n_lanes > 1:
        print(f"Found {n_lanes} lane pairs in the .ilp project file:")
        for i, p in enumerate(pairs):
            print(f"  Lane {i + 1}: raw={p['raw']}")
            print(f"           prob={p['probabilities']}")

    # --- Resolve final parameter values ---
    ws = resolve_watershed_params(args, ilp_path=args.ilp)

    # --- Lazy imports ---
    from multicut_from_ilp import (
        _Float32LazyArray,
        _InvertedLazyArray,
        _as_zyx_lazy_array,
        _open_channel_lazy,
        _open_or_compute_watershed_zarr,
    )

    # --- Process each lane ---
    ws_zarr_override = args.ws_zarr
    keep_watershed = args.keep_watershed

    for i, pair in enumerate(pairs):
        _run_one_watershed(
            raw_path=pair["raw"],
            prob_path=pair["probabilities"],
            out=out,
            args=args,
            ws=ws,
            ws_zarr_override=ws_zarr_override if n_lanes == 1 else None,
            keep_watershed=keep_watershed,
            lane_index=i,
            n_lanes=n_lanes,
            _open_channel_lazy=_open_channel_lazy,
            _Float32LazyArray=_Float32LazyArray,
            _InvertedLazyArray=_InvertedLazyArray,
            _as_zyx_lazy_array=_as_zyx_lazy_array,
            _open_or_compute_watershed_zarr=_open_or_compute_watershed_zarr,
        )

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
