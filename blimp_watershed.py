"""
blimp-watershed  –  compute only the blockwise watershed (no multicut)

Power-user entrypoint that runs only the watershed step from the blimp
pipeline.  Produces a watershed zarr that can later be fed to ``blimp``
via ``--ws-zarr`` for multicut with different parameters.

An .ilp file is **optional**: when provided, watershed parameters
(threshold, sigma, min-size, alpha, pixel-pitch, ws-method) are read
from the project as defaults and can be overridden on the command line.
Without --ilp, the four core parameters (--ws-threshold, --ws-sigma,
--ws-min-size, --ws-alpha) must be given explicitly.

Usage
-----
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

Volumes must be in zyx(c) axis order.  Both inputs must have the same shape.
Singleton channel axis is accepted (ignored).

Output
------
    <output-dir>/<raw-stem>_watershed.zarr    watershed superpixels (uint64, zyx)
    <output-dir>/watershed_params.json        exact call parameters for reproducibility
"""

import argparse
import json
import sys
from pathlib import Path


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
             "you must supply --ws-threshold, --ws-sigma, --ws-min-size, and "
             "--ws-alpha explicitly.",
    )

    args = parser.parse_args()

    # --- Validation: without .ilp, core ws params must be explicit ---
    from _cli_helpers import resolve_watershed_params, validate_watershed_params
    validate_watershed_params(args)

    # --- Resolve final parameter values ---
    ws = resolve_watershed_params(args, ilp_path=args.ilp)

    # --- Lazy imports ---
    from multicut_from_ilp import (
        _Float32LazyArray,
        _InvertedLazyArray,
        _open_channel_lazy,
        _open_or_compute_watershed_zarr,
    )

    # --- Output paths ---
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_stem = Path(args.raw).stem
    default_ws = str(out / f"{raw_stem}_watershed.zarr")

    if args.ws_zarr:
        ws_zarr_path = args.ws_zarr
        keep_watershed = True
    else:
        ws_zarr_path = default_ws
        keep_watershed = args.keep_watershed

    # --- Save call parameters for reproducibility ---
    params = {
        "raw":             args.raw,
        "probabilities":   args.probabilities,
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
        "keep_watershed":  keep_watershed,
    }
    params_file = out / "watershed_params.json"
    params_file.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Parameters written to {params_file}")

    # --- Open boundary probabilities lazily ---
    print("\n=== Computing watershed ===")
    print(f"  Watershed method   : {ws['ws_method']}")
    print(f"  Watershed parameters:")
    print(f"    threshold   : {ws['ws_threshold']}")
    print(f"    sigma       : {ws['ws_sigma']}")
    print(f"    min_size    : {ws['ws_min_size']}")
    print(f"    alpha       : {ws['ws_alpha']}")
    print(f"    pixel_pitch : {ws['ws_pixel_pitch']}")
    print(f"    invert      : {ws['ws_invert']}")

    boundary_arr, boundary_fh = _open_channel_lazy(args.probabilities, None)
    try:
        boundary_lazy = _Float32LazyArray(boundary_arr)
        vol_shape = tuple(boundary_lazy.shape)
        print(f"Volume shape: {vol_shape}")

        ws_input = (
            _InvertedLazyArray(boundary_lazy) if ws["ws_invert"]
            else boundary_lazy
        )

        ws_zarr_arr, n_superpixels = _open_or_compute_watershed_zarr(
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

        print(f"\n=== Done ===")
        print(f"Watershed    : {ws_zarr_path}")
        print(f"Superpixels  : {n_superpixels}")
        print(f"Params       : {params_file}")
        print(f"\nTo use this watershed in a multicut run:")
        ilp_arg = f"--ilp {args.ilp}" if args.ilp else "--ilp <your_project.ilp>"
        print(f"  blimp {ilp_arg} --raw {args.raw} --probabilities {args.probabilities} --output-dir {args.output_dir} --ws-zarr {ws_zarr_path}")

    finally:
        if boundary_fh is not None:
            boundary_fh.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
