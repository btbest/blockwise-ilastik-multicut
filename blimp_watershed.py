"""
blimp-watershed – compute watershed superpixels only

Power-user CLI for computing the watershed segmentation independently of the
full multicut pipeline. Useful for:
  - Computing once, then re-running multicut with different beta/solver params
  - Testing different watershed algorithms or parameters quickly
  - Reusing a watershed from an earlier run with --ws-zarr

The watershed is saved to a zarr file in --output-dir along with a params.json
file documenting all parameters used.

Usage
-----
    blimp-watershed \\
        --raw raw.zarr \\
        --probabilities boundaries.zarr \\
        --output-dir ws_out/ \\
        --ilp model.ilp

Or with explicit watershed parameters (no --ilp required):
    blimp-watershed \\
        --raw raw.zarr \\
        --probabilities boundaries.zarr \\
        --output-dir ws_out/ \\
        --ws-threshold 0.5 --ws-sigma 3.0 --ws-min-size 100 --ws-alpha 0.9

Input formats
-------------
Both --raw and --probabilities accept local zarr stores and HDF5 files:

    /path/to/file.zarr           local zarr store
    /path/to/file.h5             HDF5 file (must contain exactly one dataset)
    C:\\Users\\...\\file.h5      Windows absolute paths are also supported

Volumes must be in zyx(c) axis order. Both inputs must have the same shape.
Singleton channel axis is accepted (ignored).

Watershed reuse
---------------
The output zarr can be reused with the full blimp command:

    blimp \\
        --ilp model.ilp \\
        --raw raw.zarr \\
        --probabilities boundaries.zarr \\
        --output-dir full_out/ \\
        --ws-zarr ws_out/<raw_stem>_watershed.zarr
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

from _cli_params import add_blockwise_args, add_ws_args, resolve_ws_params


def main():
    parser = argparse.ArgumentParser(
        prog="blimp-watershed",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
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

    # Optional ILP project file (provides watershed defaults)
    parser.add_argument(
        "--ilp", required=False, metavar="PATH",
        help="Ilastik .ilp project file (optional; provides watershed parameter defaults)",
    )

    # Shared blockwise and watershed parameters
    add_blockwise_args(parser)
    add_ws_args(parser)

    args = parser.parse_args()

    # Lazy imports: only load heavy modules after argument parsing succeeds
    from ilp_reader import read_wsdt_params
    from multicut_from_ilp import (
        _open_channel_lazy,
        _Float32LazyArray,
        _InvertedLazyArray,
        _open_or_compute_watershed_zarr,
        _create_ome_zarr,
    )

    # -----------------------------------------------------------------------
    # XOR validation: either --ilp or all four ws-* params must be provided
    # -----------------------------------------------------------------------
    if args.ilp is None:
        # No ILP file: user must explicitly provide all watershed parameters
        missing = [
            name
            for name, val in {
                "--ws-threshold": args.ws_threshold,
                "--ws-sigma": args.ws_sigma,
                "--ws-min-size": args.ws_min_size,
                "--ws-alpha": args.ws_alpha,
            }.items()
            if val is None
        ]
        if missing:
            parser.error(
                "When --ilp is not provided you must supply all four watershed "
                f"parameters explicitly. Missing: {', '.join(missing)}"
            )
        # Use hardcoded defaults for fields that come only from ILP
        ilp_ws = {
            "threshold": args.ws_threshold,
            "min_size": args.ws_min_size,
            "sigma": args.ws_sigma,
            "alpha": args.ws_alpha,
            "pixel_pitch": None,  # not settable via CLI
            "blockwise": True,     # assume modern project
        }
    else:
        # Read watershed defaults from ILP (CLI args will override these)
        ilp_ws = read_wsdt_params(args.ilp)

    # -----------------------------------------------------------------------
    # Resolve watershed parameters (CLI overrides ILP defaults)
    # -----------------------------------------------------------------------
    ws = resolve_ws_params(args, ilp_ws)

    # -----------------------------------------------------------------------
    # Setup output directory and output paths
    # -----------------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_stem = Path(args.raw).stem  # e.g. "my_raw" from "my_raw.zarr"
    ws_zarr_path = str(out / f"{raw_stem}_watershed.zarr")

    # -----------------------------------------------------------------------
    # Save call parameters for reproducibility
    # -----------------------------------------------------------------------
    params = {
        "raw": args.raw,
        "probabilities": args.probabilities,
        "output_dir": str(out.resolve()),
        "ilp": args.ilp,
        "max_block_shape": args.max_block_shape,
        "halo": args.halo,
        "threads": args.threads,
        "ws_method": ws.ws_method,
        "ws_threshold": ws.ws_threshold,
        "ws_sigma": ws.ws_sigma,
        "ws_min_size": ws.ws_min_size,
        "ws_alpha": ws.ws_alpha,
        "ws_pixel_pitch": ws.ws_pixel_pitch,
        "ws_apply_nonmax": ws.ws_apply_nonmax,
        "ws_invert": ws.ws_invert,
        "ws_zarr": ws_zarr_path,
    }
    params_file = out / "params.json"
    params_file.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Parameters written to {params_file}")

    # -----------------------------------------------------------------------
    # Open probabilities file and prepare watershed input
    # -----------------------------------------------------------------------
    print("\n=== Opening probabilities file ===")
    boundary_lazy, boundary_handle = _open_channel_lazy(args.probabilities, key=None)
    try:
        # Cast to float32 (lazy) and optionally invert
        boundary_float32 = _Float32LazyArray(boundary_lazy)
        ws_input = (
            _InvertedLazyArray(boundary_float32)
            if ws.ws_invert
            else boundary_float32
        )
        vol_shape = tuple(ws_input.shape)
        print(f"Volume shape: {vol_shape}")

        # Diagnostic: sample a small central patch to verify probability convention
        _diag_shape = tuple(min(s, 64) for s in vol_shape)
        _diag_start = tuple((s - d) // 2 for s, d in zip(vol_shape, _diag_shape))
        _diag_sl = tuple(
            slice(a, a + d) for a, d in zip(_diag_start, _diag_shape)
        )
        import numpy as np

        _diag_patch = np.asarray(ws_input[_diag_sl], dtype=np.float32)
        print(f"  Boundary probability sample (central {_diag_shape} patch):")
        print(
            f"    min={_diag_patch.min():.4f}  max={_diag_patch.max():.4f}  "
            f"mean={_diag_patch.mean():.4f}  "
            f"fraction>{ws.ws_threshold}={(_diag_patch > ws.ws_threshold).mean():.3f}"
        )
        if _diag_patch.mean() > 0.5:
            print(
                f"  WARNING: mean probability > 0.5 — if most of the volume is "
                f"interior, this may indicate the file stores P(background) "
                f"rather than P(boundary).  Consider passing --ws-invert or "
                f"re-exporting the boundary channel."
            )
        del _diag_patch

        # -----------------------------------------------------------------------
        # Run watershed computation (or reuse existing zarr)
        # -----------------------------------------------------------------------
        print("\n=== Computing watershed ===")
        print(f"  Watershed method   : {ws.ws_method}")
        print(f"  Watershed parameters:")
        print(f"    threshold   : {ws.ws_threshold}")
        print(f"    sigma       : {ws.ws_sigma}")
        print(f"    min_size    : {ws.ws_min_size}")
        print(f"    alpha       : {ws.ws_alpha}")
        print(f"    pixel_pitch : {ws.ws_pixel_pitch}")
        print(f"    invert      : {ws.ws_invert}")

        ws_zarr_arr, n_superpixels = _open_or_compute_watershed_zarr(
            ws_zarr_path=ws_zarr_path,
            boundary_lazy=ws_input,
            vol_shape=vol_shape,
            block_shape=tuple(args.max_block_shape),
            halo=list(args.halo),
            ws_method=ws.ws_method,
            ws_threshold=ws.ws_threshold,
            ws_sigma=ws.ws_sigma,
            ws_min_size=ws.ws_min_size,
            ws_alpha=ws.ws_alpha,
            ws_pixel_pitch=ws.ws_pixel_pitch,
            ws_apply_nonmax=ws.ws_apply_nonmax,
            n_threads=args.threads,
        )

        print("\n=== Done ===")
        print(f"Watershed zarr : {ws_zarr_path} ({n_superpixels} superpixels)")
        print(f"Params         : {params_file}")
        return 0

    finally:
        # Close the boundary file handle if it exists
        if boundary_handle is not None:
            try:
                boundary_handle.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
