"""
ilp_reader.py
Read training data and metadata from an ilastik "Boundary-Based Segmentation
with Multicut" project file (.ilp) without importing ilastik itself.

An .ilp file is an HDF5 file.  All relevant data lives under the group
  "Training and Multicut/"
which is the projectFileGroupName used by EdgeTrainingWithMulticutWorkflow.

Public API
----------
read_feature_names(ilp_path)
    Returns the FeatureNames dict: {channel_name -> [feature_name, ...]}.

read_training_data(ilp_path, lane=0)
    Returns (X, y, feature_columns) where X is a float32 ndarray of shape
    (N_annotated, N_features), y is a uint8 array of merge/split labels
    (1=merge, 2=split), and feature_columns is the list of feature names
    (the column names from the EdgeFeatures DataFrame, excluding sp1/sp2).

read_edge_features(ilp_path, lane=0)
    Returns the full EdgeFeatures DataFrame (all edges, labeled or not) as a
    pandas DataFrame with columns [sp1, sp2, feature_1, feature_2, ...].

read_edge_labels(ilp_path, lane=0)
    Returns a dict {(sp1, sp2): label} for the manually annotated edges.
"""

import ast

import h5py
import numpy as np
import pandas as pd

APPLET_GROUP = "Training and Multicut"


# ---------------------------------------------------------------------------
# Low-level HDF5 helpers
# ---------------------------------------------------------------------------


def _decode(v):
    """Decode bytes to str, leave other types unchanged."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8")
    return v


def _dataframe_from_hdf5(h5_group):
    """
    Reconstruct a pandas DataFrame from an ilastikrag-style HDF5 group.

    Layout written by ilastikrag.util.dataframe_to_hdf5:
        group/row_index          – 1-D array of row indices
        group/column_index       – scalar or 1-D array whose string
                                   representation is eval()-able to a list
                                   of column names
        group/columns/000, 001, … – one dataset per column, in sorted order
    """
    row_index = h5_group["row_index"][()]
    raw_col_idx = h5_group["column_index"][()]
    if isinstance(raw_col_idx, np.ndarray):
        col_repr = raw_col_idx.tobytes().decode("utf-8")
    else:
        col_repr = _decode(raw_col_idx)
    columns = ast.literal_eval(col_repr)

    cols_group = h5_group["columns"]
    sorted_keys = sorted(cols_group.keys())
    data = np.column_stack([cols_group[k][()] for k in sorted_keys])

    return pd.DataFrame(data, index=row_index, columns=columns)


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------


def read_feature_names(ilp_path: str) -> dict:
    """
    Return the feature names selected during ilastik training.

    Returns
    -------
    dict  {channel_name (str): [feature_name (str), ...]}

    Example
    -------
    {
        "Membrane Probabilities 0": [
            "standard_edge_mean",
            "standard_edge_quantiles_10",
            "standard_edge_quantiles_90",
        ],
        "Raw Data 0": [
            "standard_sp_mean",
            "standard_sp_quantiles_10",
            "standard_sp_quantiles_90",
        ],
    }
    """
    result = {}
    with h5py.File(ilp_path, "r") as f:
        fn_group = f[APPLET_GROUP]["FeatureNames"]
        for channel_name, item in fn_group.items():
            channel_name = _decode(channel_name)
            if isinstance(item, h5py.Group):
                # Nested dict (unexpected for FeatureNames, but handle gracefully)
                result[channel_name] = [
                    _decode(item[k][()]) for k in sorted(item.keys())
                ]
            else:
                raw = item[()]
                if isinstance(raw, np.ndarray):
                    result[channel_name] = [_decode(v) for v in raw]
                else:
                    result[channel_name] = [_decode(raw)]
    return result


def read_edge_labels(ilp_path: str, lane: int = 0) -> dict:
    """
    Return the manually annotated edge labels for one lane.

    Returns
    -------
    dict  {(sp1 (int), sp2 (int)): label (int)}
        label == 1  →  merge
        label == 2  →  split / boundary

    The superpixel IDs correspond to the watershed superpixels that were
    active when the user annotated the data inside ilastik.
    """
    subname = f"EdgeLabels{lane:04d}"
    with h5py.File(ilp_path, "r") as f:
        group = f[APPLET_GROUP]["EdgeLabelsDict"][subname]
        sp_ids = group["sp_ids"][()]   # (N, 2) uint32
        labels = group["labels"][()]   # (N,)   uint8
    return {(int(a), int(b)): int(lbl) for (a, b), lbl in zip(sp_ids, labels)}


def read_edge_features(ilp_path: str, lane: int = 0) -> pd.DataFrame:
    """
    Return the cached edge-feature DataFrame for one lane.

    Returns
    -------
    pandas.DataFrame  with columns [sp1, sp2, <feature_1>, <feature_2>, …]
        One row per edge in the RAG.  sp1/sp2 are the superpixel ID pair.
        The remaining columns are ilastikrag feature values (float32).

    Notes
    -----
    This DataFrame is only present when the ilastik project has been saved
    after computing features (the "Live Update" has been run or training has
    been triggered).  If absent, a KeyError is raised.
    """
    subname = f"{lane:04d}"
    with h5py.File(ilp_path, "r") as f:
        ef_group = f[APPLET_GROUP]["EdgeFeatures"][subname]
        df = _dataframe_from_hdf5(ef_group)
    return df


def read_training_data(ilp_path: str, lane: int = 0):
    """
    Join EdgeFeatures with EdgeLabelsDict to produce a labeled training set.

    Returns
    -------
    X : np.ndarray  shape (N_annotated, N_features)  dtype float32
        Feature matrix for the annotated edges only.
    y : np.ndarray  shape (N_annotated,)              dtype uint8
        Labels: 1 = merge, 2 = split / boundary.
    feature_columns : list[str]
        Names of the features (columns of X), in order.

    Notes
    -----
    Only edges that appear in EdgeLabelsDict are returned.  The sp1/sp2
    columns are excluded from X.
    """
    features_df = read_edge_features(ilp_path, lane=lane)
    labels_dict = read_edge_labels(ilp_path, lane=lane)

    if not labels_dict:
        raise ValueError(
            f"No edge labels found in lane {lane} of {ilp_path}. "
            "Make sure the project has been annotated and saved."
        )

    # Build index: (sp1, sp2) → row position in features_df
    edge_index = {
        (int(row.sp1), int(row.sp2)): idx
        for idx, row in features_df[["sp1", "sp2"]].iterrows()
    }

    feature_cols = [c for c in features_df.columns if c not in ("sp1", "sp2")]
    X_rows, y_vals = [], []
    missing = 0
    for (sp1, sp2), lbl in labels_dict.items():
        key = (sp1, sp2) if (sp1, sp2) in edge_index else (sp2, sp1)
        if key not in edge_index:
            missing += 1
            continue
        row_idx = edge_index[key]
        X_rows.append(features_df.loc[row_idx, feature_cols].values)
        y_vals.append(lbl)

    if missing:
        import warnings
        warnings.warn(
            f"{missing} annotated edges were not found in the EdgeFeatures "
            "cache. They will be skipped. Re-save the project with features "
            "computed to avoid this."
        )

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.uint8)
    return X, y, feature_cols
