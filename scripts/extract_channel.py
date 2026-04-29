#!/usr/bin/env python3
"""Extract a single channel (index along axis 3) from a 4D HDF5 dataset."""

import argparse
import sys
import h5py
import numpy as np


def extract_channel(h5_path: str, dataset: str, index: int) -> None:
    stem, ext = h5_path.rsplit(".", 1) if "." in h5_path else (h5_path, "h5")
    out_path = f"{stem}-ch{index}.{ext}"

    with h5py.File(h5_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        if dataset not in f_in:
            sys.exit(f"Error: dataset '{dataset}' not found in {h5_path}.")

        ds = f_in[dataset]
        if ds.ndim != 4:
            sys.exit(f"Error: expected 4D dataset, got shape {ds.shape} ({ds.ndim}D).")

        n0, n1, n2, n_channels = ds.shape
        if not (0 <= index < n_channels):
            sys.exit(
                f"Error: index {index} out of range for axis 3 "
                f"(size {n_channels}, valid indices 0–{n_channels - 1})."
            )

        src_chunks = ds.chunks  # None if contiguous
        out_shape = (n0, n1, n2)
        out_chunks = src_chunks[:3] if src_chunks else None  # drop 4th axis

        print(f"Extracting channel {index}: {ds.shape} → {out_shape}")
        print(f"Chunks: {src_chunks} → {out_chunks}")

        out_ds = f_out.create_dataset(
            "channel", shape=out_shape, dtype=ds.dtype,
            chunks=out_chunks, compression="gzip"
        )

        if src_chunks is None:
            # Contiguous source — single read, no chunked iteration possible
            out_ds[:] = ds[:, :, :, index]
            print("Done (contiguous source, single read).")
        else:
            c0, c1, c2, _ = src_chunks
            total = ((n0 + c0 - 1) // c0) * ((n1 + c1 - 1) // c1) * ((n2 + c2 - 1) // c2)
            done = 0
            for i0 in range(0, n0, c0):
                for i1 in range(0, n1, c1):
                    for i2 in range(0, n2, c2):
                        s = np.s_[i0:i0+c0, i1:i1+c1, i2:i2+c2]
                        out_ds[s] = ds[s[0], s[1], s[2], index]
                        done += 1
                        print(f"  {done}/{total} chunks written", end="\r", flush=True)
            print()  # newline after progress

    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", help="Input HDF5 file")
    parser.add_argument("index", type=int, help="Channel index along axis 3")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name inside the HDF5 file (default: first dataset found)")
    args = parser.parse_args()

    if args.dataset is None:
        with h5py.File(args.h5_path, "r") as f:
            datasets = [k for k, v in f.items() if isinstance(v, h5py.Dataset)]
            if not datasets:
                sys.exit("Error: no top-level datasets found in the file.")
            if len(datasets) > 1:
                sys.exit(
                    f"Error: multiple datasets found {datasets}. "
                    "Specify one with --dataset."
                )
            args.dataset = datasets[0]
            print(f"Using dataset: '{args.dataset}'")

    extract_channel(args.h5_path, args.dataset, args.index)


if __name__ == "__main__":
    main()