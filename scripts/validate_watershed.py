#!/usr/bin/env python
"""Compare watershed superpixels from ilp-mc-block against ilastik reference.

Reads two label volumes (h5 or zarr), checks whether they are pixel-identical,
and if not, produces a detailed diagnostic report showing *where* and *how*
they differ.

Usage
-----
    python scripts/validate_watershed.py ours.zarr reference.h5
    python scripts/validate_watershed.py ours.zarr reference.h5 --save-diff diff.zarr
    python scripts/validate_watershed.py ours.zarr reference.h5 --slices 50 100 150

Supports .h5/.hdf5 (reads the first dataset) and .zarr volumes.
A trailing singleton channel axis (zyxc with c == 1) is automatically removed.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_volume(path: str) -> np.ndarray:
    """Load a label volume from h5 or zarr, squeezing a singleton channel."""
    p = Path(path)
    if p.suffix in (".h5", ".hdf5"):
        with h5py.File(path, "r") as f:
            datasets: list[str] = []
            f.visititems(
                lambda name, obj: datasets.append(name)
                if isinstance(obj, h5py.Dataset)
                else None
            )
            if not datasets:
                raise ValueError(f"No datasets in {path}")
            data = f[datasets[0]][()]
    elif ".zarr" in str(p):
        import zarr

        store = zarr.open(path, mode="r")
        # If the store is a group, grab the first array inside it.
        if hasattr(store, "arrays"):
            arrays = list(store.arrays())
            if arrays:
                data = arrays[0][1][()]
            else:
                data = store[()]
        else:
            data = store[()]
    else:
        raise ValueError(f"Unsupported format: {path}")

    # Squeeze trailing singleton channel (zyxc with c==1).
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3-D volume (zyx) after squeezing, got shape {data.shape}"
        )
    return data


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _label_sizes(vol: np.ndarray) -> dict[int, int]:
    """Return {label: voxel_count} for every label in *vol*."""
    labels, counts = np.unique(vol, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def _contingency_matrix(a: np.ndarray, b: np.ndarray):
    """Build a sparse contingency matrix between flat label arrays *a* and *b*.

    Returns (row_ids, col_ids, counts) where row_ids index into labels of *a*
    and col_ids index into labels of *b*.
    """
    from collections import Counter

    pairs = Counter(zip(a.ravel().tolist(), b.ravel().tolist()))
    rows, cols, counts = [], [], []
    for (r, c), n in pairs.items():
        rows.append(r)
        cols.append(c)
        counts.append(n)
    return np.array(rows), np.array(cols), np.array(counts)


def _is_pure_relabeling(a: np.ndarray, b: np.ndarray) -> tuple[bool, dict | None]:
    """Check whether *b* is a bijective relabeling of *a*.

    Returns (True, mapping_a_to_b) if every label in *a* maps to exactly one
    label in *b* and vice-versa, otherwise (False, None).
    """
    rows, cols, counts = _contingency_matrix(a, b)

    # For each label in a, check it maps to exactly one label in b.
    from collections import defaultdict

    a_to_b: dict[int, set[int]] = defaultdict(set)
    b_to_a: dict[int, set[int]] = defaultdict(set)
    for r, c in zip(rows.tolist(), cols.tolist()):
        a_to_b[r].add(c)
        b_to_a[c].add(r)

    mapping = {}
    for label_a, targets in a_to_b.items():
        if len(targets) != 1:
            return False, None
        target = next(iter(targets))
        if len(b_to_a[target]) != 1:
            return False, None
        mapping[label_a] = target

    return True, mapping


def _per_slice_disagreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return an array of length Z with the fraction of differing voxels per slice."""
    n_slices = a.shape[0]
    fracs = np.empty(n_slices, dtype=np.float64)
    slice_size = a.shape[1] * a.shape[2]
    for z in range(n_slices):
        fracs[z] = np.count_nonzero(a[z] != b[z]) / slice_size
    return fracs


def _diff_bounding_box(diff_mask: np.ndarray):
    """Return the tight (z, y, x) bounding box of True voxels in *diff_mask*."""
    zs, ys, xs = np.nonzero(diff_mask)
    if len(zs) == 0:
        return None
    return (
        (int(zs.min()), int(zs.max())),
        (int(ys.min()), int(ys.max())),
        (int(xs.min()), int(xs.max())),
    )


def _worst_slices(per_slice: np.ndarray, n: int = 5) -> list[tuple[int, float]]:
    """Return the *n* z-slices with the highest disagreement fraction."""
    order = np.argsort(per_slice)[::-1]
    result = []
    for idx in order[:n]:
        if per_slice[idx] > 0:
            result.append((int(idx), float(per_slice[idx])))
    return result


def _split_merge_analysis(a: np.ndarray, b: np.ndarray):
    """Identify labels that were split or merged between *a* and *b*.

    A label in *a* is "split" if it overlaps with more than one label in *b*.
    A label in *b* is "merged" if it receives voxels from more than one label in *a*.

    Returns (splits, merges) where each is a dict mapping a label to the set
    of labels in the other volume it overlaps with.
    """
    rows, cols, _ = _contingency_matrix(a, b)
    from collections import defaultdict

    a_to_b: dict[int, set[int]] = defaultdict(set)
    b_to_a: dict[int, set[int]] = defaultdict(set)
    for r, c in zip(rows.tolist(), cols.tolist()):
        a_to_b[r].add(c)
        b_to_a[c].add(r)

    splits = {k: v for k, v in a_to_b.items() if len(v) > 1}
    merges = {k: v for k, v in b_to_a.items() if len(v) > 1}
    return splits, merges


# ---------------------------------------------------------------------------
# Diff saving / visualization
# ---------------------------------------------------------------------------

def _save_diff_zarr(path: str, diff_mask: np.ndarray):
    """Save the boolean diff mask as a uint8 zarr array."""
    import zarr

    z = zarr.open(path, mode="w", shape=diff_mask.shape, dtype="uint8", chunks=(128, 128, 128))
    z[:] = diff_mask.astype(np.uint8)
    print(f"  Diff mask saved to {path}")


def _save_slice_images(a: np.ndarray, b: np.ndarray, diff_mask: np.ndarray,
                       slices: list[int], out_dir: str):
    """Save side-by-side PNG images for the given z-slices."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping slice images.")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for z in slices:
        if z < 0 or z >= a.shape[0]:
            print(f"  Skipping out-of-range slice z={z}")
            continue
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(a[z], cmap="nipy_spectral", interpolation="nearest")
        axes[0].set_title(f"Ours  z={z}")
        axes[1].imshow(b[z], cmap="nipy_spectral", interpolation="nearest")
        axes[1].set_title(f"Reference  z={z}")
        axes[2].imshow(diff_mask[z], cmap="Reds", interpolation="nearest",
                       vmin=0, vmax=1)
        axes[2].set_title(f"Diff  z={z}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"diff_z{z:04d}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out / f'diff_z{z:04d}.png'}")


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def validate(ours_path: str, ref_path: str, *,
             save_diff: str | None = None,
             slices: list[int] | None = None,
             slice_dir: str | None = None,
             top_n_slices: int = 10):
    """Run full validation and print a report."""
    print("Loading volumes …")
    ours = _load_volume(ours_path)
    ref = _load_volume(ref_path)

    sep = "=" * 72
    print(f"\n{sep}")
    print("  WATERSHED VALIDATION REPORT")
    print(f"{sep}\n")

    # -- 1. Shape ----------------------------------------------------------
    print(f"  Ours shape:       {ours.shape}   dtype: {ours.dtype}")
    print(f"  Reference shape:  {ref.shape}   dtype: {ref.dtype}")
    if ours.shape != ref.shape:
        print("\n  ** SHAPE MISMATCH — cannot continue comparison **")
        sys.exit(1)
    print("  Shapes match: YES\n")

    # -- 2. Unique label counts --------------------------------------------
    ours_labels = set(np.unique(ours).tolist())
    ref_labels = set(np.unique(ref).tolist())
    n_ours = len(ours_labels)
    n_ref = len(ref_labels)
    print(f"  Unique labels (ours):       {n_ours}")
    print(f"  Unique labels (reference):  {n_ref}")
    if n_ours == n_ref:
        print("  Label counts match: YES")
    else:
        print(f"  Label counts match: NO  (difference: {n_ours - n_ref:+d})")

    # -- 3. Label set identity ---------------------------------------------
    only_ours = ours_labels - ref_labels
    only_ref = ref_labels - ours_labels
    if not only_ours and not only_ref:
        print("  Label sets identical: YES\n")
    else:
        print("  Label sets identical: NO")
        if only_ours:
            sample = sorted(only_ours)[:20]
            print(f"    Labels only in ours ({len(only_ours)} total): {sample}{'…' if len(only_ours) > 20 else ''}")
        if only_ref:
            sample = sorted(only_ref)[:20]
            print(f"    Labels only in ref  ({len(only_ref)} total): {sample}{'…' if len(only_ref) > 20 else ''}")
        print()

    # -- 4. Label value range ----------------------------------------------
    print(f"  Label range (ours):       [{min(ours_labels)}, {max(ours_labels)}]")
    print(f"  Label range (reference):  [{min(ref_labels)}, {max(ref_labels)}]")

    # Check consecutive 0-indexed
    expected_ours = set(range(n_ours))
    expected_ref = set(range(n_ref))
    print(f"  Ours 0-indexed consecutive:  {'YES' if ours_labels == expected_ours else 'NO'}")
    print(f"  Ref  0-indexed consecutive:  {'YES' if ref_labels == expected_ref else 'NO'}")
    # Check 1-indexed consecutive
    expected_ours_1 = set(range(1, n_ours + 1))
    expected_ref_1 = set(range(1, n_ref + 1))
    print(f"  Ours 1-indexed consecutive:  {'YES' if ours_labels == expected_ours_1 else 'NO'}")
    print(f"  Ref  1-indexed consecutive:  {'YES' if ref_labels == expected_ref_1 else 'NO'}")
    print()

    # -- 5. Pixel-identical check ------------------------------------------
    diff_mask = ours != ref
    n_diff = int(np.count_nonzero(diff_mask))
    n_total = int(ours.size)
    pct = 100.0 * n_diff / n_total

    if n_diff == 0:
        print("  *** PIXEL-IDENTICAL: YES ***")
        print(f"\n{sep}")
        print("  Volumes match perfectly — no further analysis needed.")
        print(f"{sep}\n")
        return

    print(f"  PIXEL-IDENTICAL: NO")
    print(f"  Differing voxels: {n_diff:,} / {n_total:,}  ({pct:.4f}%)\n")

    # -- 6. Relabeling check -----------------------------------------------
    print("  Checking if difference is just a relabeling …")
    is_relabel, mapping = _is_pure_relabeling(ours, ref)
    if is_relabel:
        print("  ** The volumes are a bijective relabeling of each other **")
        print("     (same topology, different label IDs)")
        sample = list(mapping.items())[:10]
        print(f"     Sample mapping (ours → ref): {sample}")
    else:
        print("  Not a pure relabeling — there are topological differences.\n")

    # -- 7. Split / merge analysis -----------------------------------------
    if not is_relabel:
        print("  Split / merge analysis:")
        splits, merges = _split_merge_analysis(ours, ref)
        print(f"    Labels in ours split across multiple ref labels: {len(splits)}")
        print(f"    Labels in ref receiving from multiple ours labels: {len(merges)}")
        if splits:
            sample = dict(list(splits.items())[:5])
            print(f"    Sample splits (ours_label → set of ref_labels): {sample}")
        if merges:
            sample = dict(list(merges.items())[:5])
            print(f"    Sample merges (ref_label ← set of ours_labels): {sample}")
        print()

    # -- 8. Per-slice disagreement -----------------------------------------
    per_slice = _per_slice_disagreement(ours, ref)
    worst = _worst_slices(per_slice, n=top_n_slices)
    print(f"  Per-slice disagreement (top {len(worst)} worst z-slices):")
    for z, frac in worst:
        print(f"    z={z:5d}  {100 * frac:8.4f}% differing")
    print()

    # -- 9. Bounding box of diff -------------------------------------------
    bbox = _diff_bounding_box(diff_mask)
    if bbox:
        (z0, z1), (y0, y1), (x0, x1) = bbox
        print(f"  Bounding box of all differences:")
        print(f"    z: [{z0}, {z1}]  (span {z1 - z0 + 1})")
        print(f"    y: [{y0}, {y1}]  (span {y1 - y0 + 1})")
        print(f"    x: [{x0}, {x1}]  (span {x1 - x0 + 1})")
        print()

    # -- 10. Superpixel size distribution ----------------------------------
    print("  Superpixel size statistics:")
    ours_sizes = _label_sizes(ours)
    ref_sizes = _label_sizes(ref)
    ours_vals = np.array(list(ours_sizes.values()), dtype=np.float64)
    ref_vals = np.array(list(ref_sizes.values()), dtype=np.float64)
    for tag, vals in [("ours", ours_vals), ("ref ", ref_vals)]:
        print(f"    {tag}:  min={int(vals.min()):>8,}  median={int(np.median(vals)):>8,}"
              f"  mean={vals.mean():>10.1f}  max={int(vals.max()):>8,}")
    print()

    # -- 11. Boundary agreement (are boundaries in the same places?) -------
    print("  Boundary analysis:")
    from scipy import ndimage

    def _boundary_mask(vol):
        """Voxels adjacent to a different label (6-connected)."""
        struct = ndimage.generate_binary_structure(3, 1)
        dilated = ndimage.grey_dilation(vol, footprint=struct)
        eroded = ndimage.grey_erosion(vol, footprint=struct)
        return dilated != eroded

    ours_bnd = _boundary_mask(ours)
    ref_bnd = _boundary_mask(ref)
    n_ours_bnd = int(np.count_nonzero(ours_bnd))
    n_ref_bnd = int(np.count_nonzero(ref_bnd))
    both_bnd = ours_bnd & ref_bnd
    n_both = int(np.count_nonzero(both_bnd))
    precision = n_both / n_ours_bnd if n_ours_bnd else 1.0
    recall = n_both / n_ref_bnd if n_ref_bnd else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"    Boundary voxels (ours):  {n_ours_bnd:,}")
    print(f"    Boundary voxels (ref):   {n_ref_bnd:,}")
    print(f"    Boundary overlap:        {n_both:,}")
    print(f"    Boundary precision:      {precision:.6f}")
    print(f"    Boundary recall:         {recall:.6f}")
    print(f"    Boundary F1:             {f1:.6f}")
    print()

    # -- 12. Save outputs --------------------------------------------------
    if save_diff:
        _save_diff_zarr(save_diff, diff_mask)

    if slices is not None and slice_dir:
        _save_slice_images(ours, ref, diff_mask, slices, slice_dir)
    elif slices is None and worst and slice_dir:
        auto_slices = [z for z, _ in worst[:5]]
        _save_slice_images(ours, ref, diff_mask, auto_slices, slice_dir)

    print(f"{sep}")
    if is_relabel:
        print("  RESULT: TOPOLOGICALLY IDENTICAL (pure relabeling)")
    else:
        print("  RESULT: DIFFERENCES FOUND — see report above")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate watershed superpixels against ilastik reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s output_watershed.zarr ilastik_ws.h5
              %(prog)s output.zarr ref.h5 --save-diff diff.zarr
              %(prog)s output.zarr ref.h5 --slice-dir ./diffs --slices 50 100 150
        """),
    )
    parser.add_argument("ours", help="Watershed from ilp-mc-block (.zarr or .h5)")
    parser.add_argument("reference", help="Watershed from ilastik (.zarr or .h5)")
    parser.add_argument("--save-diff", metavar="PATH",
                        help="Save boolean diff mask as a zarr array")
    parser.add_argument("--slices", type=int, nargs="*", metavar="Z",
                        help="Z-slices to render as PNGs (default: worst 5)")
    parser.add_argument("--slice-dir", metavar="DIR", default=None,
                        help="Directory for slice PNG output (enables image saving)")
    parser.add_argument("--top-n-slices", type=int, default=10,
                        help="Number of worst slices to show in the report (default: 10)")

    args = parser.parse_args()
    validate(
        args.ours,
        args.reference,
        save_diff=args.save_diff,
        slices=args.slices,
        slice_dir=args.slice_dir,
        top_n_slices=args.top_n_slices,
    )


if __name__ == "__main__":
    main()
