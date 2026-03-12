"""
run_demo.py
===========
End-to-end demo: create toy HDF5 volumes, fit the classifier from the
included example ILP, then run in-memory blockwise multicut.

Usage:
    python run_demo.py [--out-dir /tmp/demo]
"""

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

ILP = REPO / "libs" / "example_mc_project.ilp"
CHANNELS = {
    "Raw Data": "raw.h5",
    "wsdt boundary channel": "boundary.h5",
}
VOL_SHAPE = (64, 64, 64)  # small enough to finish in seconds


def make_toy_volumes(out_dir: Path, rng: np.random.Generator) -> dict:
    """Write two small synthetic HDF5 volumes and return {channel: h5_path}."""
    paths = {}
    for ch_name, filename in CHANNELS.items():
        data = rng.random(VOL_SHAPE, dtype=np.float32)
        # boundary channel: make it look like membrane probabilities
        if "boundary" in ch_name:
            data = np.clip(rng.normal(0.3, 0.2, VOL_SHAPE).astype(np.float32), 0, 1)
        path = out_dir / filename
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)
        paths[ch_name] = str(path)
        print(f"  Wrote {path}  shape={data.shape} dtype={data.dtype}")
    return paths


def fit_rf(ilp: Path, out_dir: Path) -> Path:
    from fit_classifier import fit_rf_from_ilp

    rf = fit_rf_from_ilp(str(ilp), n_estimators=50, n_jobs=1)
    rf_path = out_dir / "rf.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"  RF saved to {rf_path}")
    return rf_path


def run_multicut(ilp: Path, rf_path: Path, vol_paths: dict, out_dir: Path) -> Path:
    from ilp_reader import read_feature_names
    from multicut_from_ilp import _run_in_memory

    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    channel_specs = [f"{ch}:{path}:/data" for ch, path in vol_paths.items()]
    out_h5 = out_dir / "segmentation.h5"

    seg = _run_in_memory(
        ilp_path=str(ilp),
        rf=rf,
        channel_specs=channel_specs,
        output_path=str(out_h5),
        output_key="/seg",
        beta=0.5,
        block_shape=[32, 32, 32],
        halo=[8, 8, 8],
        internal_solver="kernighan-lin",
        n_threads=4,
        use_2dws=False,
        ws_threshold=0.5,
        ws_sigma=2.0,
        verbose=True,
    )
    return out_h5, seg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/blockwise_mc_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("=" * 60)
    print("Step 1: Creating toy HDF5 volumes …")
    vol_paths = make_toy_volumes(out_dir, rng)

    print("\nStep 2: Fitting RandomForest from ILP …")
    rf_path = fit_rf(ILP, out_dir)

    print("\nStep 3: Running blockwise multicut …")
    out_h5, seg = run_multicut(ILP, rf_path, vol_paths, out_dir)

    print("\n" + "=" * 60)
    n_segs = len(np.unique(seg))
    print(f"Done!  Output: {out_h5}")
    print(f"  Volume shape    : {seg.shape}")
    print(f"  Unique segments : {n_segs}")
    print("=" * 60)


if __name__ == "__main__":
    main()
