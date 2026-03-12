"""
make_test_ilp.py
================
Generate a synthetic-but-structurally-faithful multicut .ilp fixture.

An ilastik "Boundary-Based Segmentation with Multicut" project file is
just an HDF5 file whose serialization format is defined in

  ilastik/applets/edgeTraining/edgeTrainingSerializer.py

This script reproduces that exact layout without importing ilastik or
ilastikrag, so it works in any Python environment that has h5py and numpy.

Run:
    python tests/make_test_ilp.py [--output tests/test_multicut.ilp]
"""

import argparse
import ast
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Topology: a tiny synthetic RAG (region adjacency graph) on 20 superpixels
# ---------------------------------------------------------------------------
# Superpixel IDs go from 1 to 20.
# Edges form a chain  1-2-3-…-20 plus a few cross-links so we can have
# both "merge" and "split" labels without cycling.

_SP_IDS = list(range(1, 21))  # 20 superpixels

_ALL_EDGES = (
    # chain backbone
    [(i, i + 1) for i in range(1, 20)]
    # cross-links
    + [(1, 5), (5, 10), (10, 15), (3, 7), (8, 12), (12, 16), (2, 9), (14, 18)]
)
# de-duplicate, ensure sp1 < sp2
_ALL_EDGES = sorted({(min(a, b), max(a, b)) for a, b in _ALL_EDGES})

# Edge labels – matches the label set used in the e2e GUI test (1=merge, 2=split)
# We annotate a subset of the edges (the ones with IDs that appear in
# both the edge list and the labeldict below).
_EDGE_LABELS = {
    (1, 2): 2,
    (3, 4): 2,
    (5, 6): 1,
    (7, 8): 1,
    (9, 10): 2,
    (11, 12): 1,
    (13, 14): 2,
    (15, 16): 1,
    (5, 10): 2,
    (10, 15): 1,
}

# Feature channel names – exactly as ilastik stores them in FeatureNames
FEATURE_NAMES = {
    "Raw Data": ["standard_sp_mean"],
    "Probabilities-0": [],  # channel present but no features selected
    "Probabilities-1": ["standard_edge_mean"],
}

# DataFrame column names.  ilastikrag concatenates channel + "_" + feature.
FEAT_COLS = ["Probabilities-1_standard_edge_mean", "Raw Data_standard_sp_mean"]
ALL_COLS = ["sp1", "sp2"] + FEAT_COLS

APPLET_GROUP = "Training and Multicut"


# ---------------------------------------------------------------------------
# HDF5 helpers – mirrors ilastikrag.util.dataframe_to_hdf5
# ---------------------------------------------------------------------------


def _write_dataframe_to_hdf5(group: h5py.Group, df: pd.DataFrame) -> None:
    """
    Write a pandas DataFrame in the ilastikrag HDF5 format that ilp_reader
    expects:

        group/row_index          – 1-D int64 array
        group/column_index       – bytes scalar  (str(list_of_col_names))
        group/columns/000 …      – one float32 dataset per column, zero-padded
    """
    group.create_dataset("row_index", data=df.index.to_numpy(dtype=np.int64))
    col_repr = str(list(df.columns)).encode("utf-8")
    group.create_dataset("column_index", data=col_repr)
    cols_grp = group.create_group("columns")
    for i, col in enumerate(df.columns):
        cols_grp.create_dataset(f"{i:03d}", data=df[col].to_numpy(dtype=np.float64))


def _write_feature_names(parent: h5py.Group, feature_names: dict) -> None:
    """
    Mirrors SerialDictSlot._saveValue for a dict {str: list[str]}.
    Each list is stored as an HDF5 dataset of bytes-encoded strings.
    """
    grp = parent.create_group("FeatureNames")
    for channel, features in feature_names.items():
        encoded = [f.encode("utf-8") for f in features]
        # h5py needs a special dtype for empty variable-length strings
        if encoded:
            grp.create_dataset(channel, data=encoded)
        else:
            dt = h5py.special_dtype(vlen=bytes)
            grp.create_dataset(channel, shape=(0,), dtype=dt)


def _write_edge_labels(parent: h5py.Group, label_dict: dict, lane: int = 0) -> None:
    """
    Mirrors SerialEdgeLabelsDictSlot._serialize.
    """
    outer = parent.require_group("EdgeLabelsDict")
    subname = f"EdgeLabels{lane:04d}"
    grp = outer.create_group(subname)
    sp_ids = np.array(list(label_dict.keys()), dtype=np.uint32)   # (N, 2)
    labels = np.array(list(label_dict.values()), dtype=np.uint8)  # (N,)
    grp.create_dataset("sp_ids", data=sp_ids)
    grp.create_dataset("labels", data=labels)


def _write_edge_features(parent: h5py.Group, df: pd.DataFrame, lane: int = 0) -> None:
    """
    Mirrors SerialCachedDataFrameSlot._serialize for the EdgeFeatures group.
    Subname is '{:04}'.format(lane_index) (default SerialSlot subname).
    """
    outer = parent.require_group("EdgeFeatures")
    subname = f"{lane:04d}"
    grp = outer.create_group(subname)
    _write_dataframe_to_hdf5(grp, df)


# ---------------------------------------------------------------------------
# Build synthetic feature data
# ---------------------------------------------------------------------------


def _make_edge_features_df(edges, feat_cols, rng) -> pd.DataFrame:
    """
    Create a realistic-looking edge-feature DataFrame for the given edge list.
    Features are random floats in [0, 1].
    """
    sp1 = np.array([e[0] for e in edges], dtype=np.float64)
    sp2 = np.array([e[1] for e in edges], dtype=np.float64)
    feats = rng.random((len(edges), len(feat_cols))).astype(np.float64)
    data = np.column_stack([sp1, sp2, feats])
    return pd.DataFrame(data, columns=["sp1", "sp2"] + feat_cols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_test_ilp(output_path: str, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    edges = _ALL_EDGES
    df = _make_edge_features_df(edges, FEAT_COLS, rng)

    with h5py.File(output, "w") as f:
        # Top-level metadata (ilastik writes these)
        f.attrs["workflow"] = b"EdgeTrainingWithMulticutWorkflow"
        applet = f.create_group(APPLET_GROUP)

        _write_feature_names(applet, FEATURE_NAMES)
        _write_edge_labels(applet, _EDGE_LABELS, lane=0)
        _write_edge_features(applet, df, lane=0)

    print(f"Written: {output}")
    print(f"  Superpixels : {len(_SP_IDS)}")
    print(f"  Total edges : {len(edges)}")
    print(f"  Labeled edges: {len(_EDGE_LABELS)}")
    print(f"  Feature cols : {FEAT_COLS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/test_multicut.ilp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_test_ilp(args.output, seed=args.seed)
