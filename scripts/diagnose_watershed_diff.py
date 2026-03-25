#!/usr/bin/env python
"""Diagnose where blimp vs ilastik watershed divergence originates.

Runs the watershed pipeline step-by-step on a single block and compares
intermediate results to find where the first floating-point divergence
appears.  This pinpoints whether the issue is in the input probabilities,
the distance transform, the Gaussian smoothing, or the seed detection.

This script is written to work on probabilities extracted mid-run
from ilastik and blimp respectively using something like:

```
# in multicut_from_ilp._ilastik_parallel_watershed:
export_dir = "C:/Users/root/EM/plasmamem/probs-debug"
import zarr
from pathlib import Path
zarr_path = str(Path(export_dir) / "blimp_pmap.zarr")
z = zarr.open(zarr_path, mode="w", shape=boundary_lazy.shape, dtype=boundary_lazy.dtype,
              chunks=(128, 128, 128, 1) if boundary_lazy.ndim == 4 else (128, 128, 128))
z[:] = boundary_lazy
```

```
# in ilastik OpWsdt.execute:
export_dir = "C:/Users/root/EM/plasmamem/probs-debug"
import zarr
from pathlib import Path
roi_str = "_".join(f"{s}_{e}" for s, e in zip(roi.start, roi.stop))
zarr_path = str(Path(export_dir) / f"ilastik_pmap_{roi_str}.zarr")
z = zarr.open(zarr_path, mode="w", shape=pmap.shape, dtype=pmap.dtype,
              chunks=(128, 128, 128, 1) if pmap.ndim == 4 else (128, 128, 128))
z[:] = pmap
```

Usage
-----
    python scripts/diagnose_watershed_diff.py \
        --blimp-probs  boundaries.zarr \
        --ilastik-probs exported_from_ilastik.h5 \
        --ws-threshold 0.5 --ws-sigma 3.0 --ws-alpha 0.9 \
        --ws-min-size 100 \
        [--block-id 0] [--pixel-pitch 1.0 1.0 1.0]

If only --blimp-probs is given (no --ilastik-probs), the script runs the
full block-by-block watershed twice on the same data to confirm
determinism, and prints intermediate statistics useful for debugging.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_block(arr, blocking, block_id, halo):
    """Extract one outer block (inner + halo) from an array-like."""
    block = blocking.getBlockWithHalo(block_id, halo)
    outer_bb = tuple(
        slice(s, e)
        for s, e in zip(block.outerBlock.begin, block.outerBlock.end)
    )
    return np.asarray(arr[outer_bb], dtype=np.float32), block


def _step_by_step_watershed(data, threshold, sigma, alpha, min_size,
                            pixel_pitch, label=""):
    """Run distance_transform_watershed step-by-step, returning intermediates."""
    import vigra

    try:
        import fastfilters as ff
        smoother = "fastfilters"
    except ImportError:
        import vigra.filters as ff
        smoother = "vigra.filters"

    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}Gaussian smoother: {smoother}")
    print(f"{prefix}Input: dtype={data.dtype} shape={data.shape} "
          f"min={data.min():.8f} max={data.max():.8f} mean={data.mean():.8f}")

    # Step 1: threshold
    thresholded = (data > threshold).astype("uint32")
    n_above = int(thresholded.sum())
    print(f"{prefix}After threshold ({threshold}): "
          f"{n_above} voxels above ({100*n_above/data.size:.2f}%)")

    # Step 2: distance transform
    pp = pixel_pitch if pixel_pitch is not None else None
    dt = vigra.filters.distanceTransform(thresholded, pixel_pitch=pp)
    print(f"{prefix}Distance transform: min={dt.min():.8f} max={dt.max():.8f} "
          f"mean={dt.mean():.8f}")

    # Step 3: Gaussian smoothing of DT (for seeds)
    if sigma:
        dt_smooth = ff.gaussianSmoothing(dt, sigma)
    else:
        dt_smooth = dt
    print(f"{prefix}Smoothed DT (sigma={sigma}): min={dt_smooth.min():.8f} "
          f"max={dt_smooth.max():.8f} mean={dt_smooth.mean():.8f}")

    # Step 4: local maxima -> seeds
    compute_maxima = (vigra.analysis.localMaxima if dt_smooth.ndim == 2
                      else vigra.analysis.localMaxima3D)
    seeds_raw = compute_maxima(dt_smooth, marker=np.nan,
                               allowAtBorder=True, allowPlateaus=True)
    seeds_bool = np.isnan(seeds_raw)
    seeds = vigra.analysis.labelMultiArrayWithBackground(seeds_bool.view("uint8"))
    n_seeds = int(seeds.max())
    print(f"{prefix}Seeds: {n_seeds} local maxima found")

    # Step 5: weight map
    dt_norm = 1.0 - (dt_smooth - dt_smooth.min()) / dt_smooth.max()
    if sigma:
        hmap = alpha * ff.gaussianSmoothing(data, sigma) + (1.0 - alpha) * dt_norm
    else:
        hmap = alpha * data + (1.0 - alpha) * dt_norm
    print(f"{prefix}Weight map: min={hmap.min():.8f} max={hmap.max():.8f} "
          f"mean={hmap.mean():.8f}")

    # Step 6: seeded watershed
    ws, max_id = vigra.analysis.watershedsNew(hmap, seeds=seeds)
    print(f"{prefix}Watershed: {max_id} segments")

    return {
        "input": data,
        "thresholded": thresholded,
        "dt": dt,
        "dt_smooth": dt_smooth,
        "seeds_bool": seeds_bool,
        "seeds": seeds,
        "dt_norm": dt_norm,
        "hmap": hmap,
        "ws": ws,
    }


def _compare_arrays(name, a, b):
    """Compare two arrays and report differences."""
    if a.shape != b.shape:
        print(f"    {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    if a.dtype != b.dtype:
        print(f"    {name}: dtype differs ({a.dtype} vs {b.dtype}), casting for comparison")
        b = b.astype(a.dtype)

    if np.array_equal(a, b):
        print(f"    {name}: IDENTICAL")
        return True

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    n_diff = int(np.count_nonzero(diff > 0))
    pct = 100.0 * n_diff / a.size
    print(f"    {name}: DIFFERS at {n_diff:,} voxels ({pct:.4f}%)")
    print(f"      max abs diff: {diff.max():.10e}")
    print(f"      mean abs diff (nonzero only): {diff[diff > 0].mean():.10e}")

    # Show a few example locations
    zs, ys, xs = np.nonzero(diff > 0)
    for i in range(min(3, len(zs))):
        z, y, x = int(zs[i]), int(ys[i]), int(xs[i])
        print(f"      e.g. [{z},{y},{x}]: A={a[z,y,x]}  B={b[z,y,x]}")

    return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blimp-probs", required=True,
                        help="Boundary probabilities as blimp would read them")
    parser.add_argument("--ilastik-probs", default=None,
                        help="Boundary probabilities as ilastik sees them (optional)")
    parser.add_argument("--ws-threshold", type=float, default=0.5)
    parser.add_argument("--ws-sigma", type=float, default=3.0)
    parser.add_argument("--ws-alpha", type=float, default=0.9)
    parser.add_argument("--ws-min-size", type=int, default=100)
    parser.add_argument("--pixel-pitch", type=float, nargs="*", default=None)
    parser.add_argument("--block-id", type=int, default=0,
                        help="Which 128^3 block to analyze (default: 0)")
    args = parser.parse_args()

    import nifty.tools as nt

    # --- Load blimp probabilities ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from multicut_from_ilp import _Float32LazyArray, _open_channel_lazy

    arr_b, fh_b = _open_channel_lazy(args.blimp_probs, None)
    lazy_b = _Float32LazyArray(arr_b)
    shape = tuple(lazy_b.shape)
    print(f"Volume shape: {shape}")
    print(f"Source dtype: {arr_b.dtype}")

    ndim = len(shape)
    BLOCK_SHAPE = (128,) * ndim
    HALO = [10] * ndim
    blocking = nt.blocking([0] * ndim, list(shape), list(BLOCK_SHAPE))
    print(f"Blocks: {blocking.numberOfBlocks} x {BLOCK_SHAPE}, halo={HALO}")
    print(f"Analyzing block {args.block_id}\n")

    block_b, block_info = _load_block(lazy_b, blocking, args.block_id, HALO)
    print("=" * 60)
    print("BLIMP path:")
    print("=" * 60)
    results_b = _step_by_step_watershed(
        block_b, args.ws_threshold, args.ws_sigma, args.ws_alpha,
        args.ws_min_size, args.pixel_pitch, label="blimp",
    )

    if args.ilastik_probs is None:
        print("\nNo --ilastik-probs given. Running same data again to confirm determinism...")
        print("=" * 60)
        print("SECOND RUN (same data):")
        print("=" * 60)
        results_b2 = _step_by_step_watershed(
            block_b.copy(), args.ws_threshold, args.ws_sigma, args.ws_alpha,
            args.ws_min_size, args.pixel_pitch, label="run2",
        )
        print("\n--- Determinism check ---")
        for key in results_b:
            _compare_arrays(key, results_b[key], results_b2[key])
    else:
        # Load ilastik probabilities
        arr_i, fh_i = _open_channel_lazy(args.ilastik_probs, None)
        lazy_i = _Float32LazyArray(arr_i)
        shape_i = tuple(lazy_i.shape)
        if shape_i != shape:
            print(f"ERROR: shape mismatch: blimp={shape} ilastik={shape_i}")
            sys.exit(1)

        block_i, _ = _load_block(lazy_i, blocking, args.block_id, HALO)
        print()
        print("=" * 60)
        print("ILASTIK path:")
        print("=" * 60)
        results_i = _step_by_step_watershed(
            block_i, args.ws_threshold, args.ws_sigma, args.ws_alpha,
            args.ws_min_size, args.pixel_pitch, label="ilastik",
        )

        print()
        print("=" * 60)
        print("COMPARISON (step by step):")
        print("=" * 60)
        first_diff = None
        for key in results_b:
            identical = _compare_arrays(key, results_b[key], results_i[key])
            if not identical and first_diff is None:
                first_diff = key

        if first_diff:
            print(f"\n  >>> First divergence at: {first_diff}")
            if first_diff == "input":
                print("      The probability data itself differs — check dtype, "
                      "export precision, and compression.")
            elif first_diff == "thresholded":
                print("      Values near the threshold boundary differ — "
                      "likely float precision in the input.")
            elif first_diff in ("dt", "dt_smooth"):
                print("      The distance transform or its smoothing differs — "
                      "check fastfilters vs vigra.filters versions.")
            elif first_diff in ("seeds_bool", "seeds"):
                print("      Seed detection diverged — localMaxima3D is sensitive "
                      "to plateaus in the smoothed distance transform.")
        else:
            print("\n  All intermediates are identical — watershed should match!")

        if fh_i is not None:
            fh_i.close()

    if fh_b is not None:
        fh_b.close()


if __name__ == "__main__":
    main()
