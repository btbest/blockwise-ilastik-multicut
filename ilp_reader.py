"""
ilp_reader.py
Read training data and metadata from an ilastik "Boundary-Based Segmentation
with Multicut" project file (.ilp) without importing ilastik itself.

An .ilp file is an HDF5 file.  All relevant data lives under the group
  "Training and Multicut/"
which is the projectFileGroupName used by EdgeTrainingWithMulticutWorkflow.

Public API
----------
discover_lanes(ilp_path)
    Return sorted list of lane indices that have edge labels saved.

read_feature_names(ilp_path)
    Returns the FeatureNames dict: {channel_name -> [feature_name, ...]}.

read_wsdt_params(ilp_path)
    Returns the DT Watershed applet parameters as a dict:
    {threshold, min_size, sigma, alpha}.  Falls back to ilastik defaults
    when the group is absent (old projects).

read_training_data(ilp_path, lane=None)
    If lane is None (default), reads all lanes and concatenates.
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
import re

import h5py
import numpy as np
import pandas as pd

APPLET_GROUP = "Training and Multicut"
WSDT_GROUP = "DT Watershed"


# ---------------------------------------------------------------------------
# Low-level HDF5 helpers
# ---------------------------------------------------------------------------


def _open_ilp_file(ilp_path: str, mode: str = "r"):
    """
    Open an ilastik .ilp file with friendly error message if file is locked.

    Parameters
    ----------
    ilp_path : str
        Path to the .ilp file
    mode : str
        File mode (default "r" for read-only)

    Returns
    -------
    h5py.File
        The opened HDF5 file object

    Raises
    ------
    OSError
        If the file is already open in ilastik (file lock error) or other I/O issues
    """
    try:
        return h5py.File(ilp_path, mode)
    except OSError as e:
        error_msg = str(e).lower()
        # Windows (error 33 = ERROR_LOCK_VIOLATION)
        is_windows_lock = hasattr(e, 'winerror') and e.winerror == 33
        # Unix/macOS errors: "unable to lock", "resource busy", "device or resource busy"
        is_unix_lock = (
            "unable to lock" in error_msg
            or "resource busy" in error_msg
            or e.errno in (16, 13)  # EBUSY=16, EACCES=13
        )
        if is_windows_lock or is_unix_lock:
            raise OSError(
                f"The project file is already open in ilastik: {ilp_path}\n"
                f"Please close the file in ilastik before running this command."
            ) from e
        raise


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
    # column_index is stored either as a Python list repr (['a', 'b', ...]) or
    # as a numpy array repr (array(['a', 'b', ...], dtype=object)) depending on
    # the ilastikrag version.  Try literal_eval first; fall back to regex.
    try:
        columns = ast.literal_eval(col_repr)
    except (ValueError, SyntaxError):
        columns = re.findall(r"'([^']*)'", col_repr)

    cols_group = h5_group["columns"]
    sorted_keys = sorted(cols_group.keys())
    data = np.column_stack([cols_group[k][()] for k in sorted_keys])

    return pd.DataFrame(data, index=row_index, columns=columns)


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------


def discover_lanes(ilp_path: str) -> list:
    """
    Return sorted list of lane indices that have edge labels saved in the .ilp.

    A multi-lane project (e.g. trained on three 256³ crops) stores labels in
    EdgeLabels0000, EdgeLabels0001, EdgeLabels0002 etc.  This function
    discovers which ones are present.

    Returns
    -------
    list[int]  e.g. [0, 1, 2]
    """
    with _open_ilp_file(ilp_path) as f:
        keys = list(f[APPLET_GROUP]["EdgeLabelsDict"].keys())
    # keys look like "EdgeLabels0000"; strip the prefix to get the index
    return sorted(int(k[len("EdgeLabels"):]) for k in keys)


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
    with _open_ilp_file(ilp_path) as f:
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


def read_wsdt_params(ilp_path: str) -> dict:
    """
    Read DT Watershed applet parameters from an ilastik project file.

    Returns
    -------
    dict with keys:
        threshold   (float)       – boundary probability threshold (default 0.5)
        min_size    (int)         – minimum superpixel size in pixels (default 100)
        sigma       (float)       – smoothing applied to both seed and weight maps (default 3.0)
        alpha       (float)       – blend factor between boundary data and distance
                                    transform (default 0.9)
        pixel_pitch (list | None) – anisotropy factors [z, y, x]; None = isotropic
        blockwise   (bool)        – True when ilastik ran its blockwise (128³, halo=10)
                                    watershed; False for very old projects that ran on
                                    the full crop at once

    If the ``DT Watershed`` group is absent (e.g. very old project files) or
    any individual key is missing, the ilastik defaults are returned for that
    entry.

    Notes
    -----
    The ``InvertPixelProbabilities`` flag is *not* serialised by ilastik and
    therefore cannot be read from the file.  Pass it explicitly via the
    ``--ws-invert`` CLI flag when needed.
    """
    defaults = {
        "threshold": 0.5,
        "min_size":  100,
        "sigma":     3.0,
        "alpha":     0.9,
        "pixel_pitch": None,  # None → isotropic (no anisotropy correction)
        "blockwise": True,    # False only for very old projects (pre-v0.2 serialiser)
    }
    try:
        with _open_ilp_file(ilp_path) as f:
            if WSDT_GROUP not in f:
                return dict(defaults)
            g = f[WSDT_GROUP]
            result = {}
            result["threshold"] = float(g["Threshold"][()]) if "Threshold" in g else defaults["threshold"]
            result["min_size"]  = int(g["MinSize"][()])     if "MinSize"   in g else defaults["min_size"]
            result["sigma"]     = float(g["Sigma"][()])     if "Sigma"     in g else defaults["sigma"]
            result["alpha"]     = float(g["Alpha"][()])     if "Alpha"     in g else defaults["alpha"]

            # PixelPitch is stored as a list; [] means isotropic → pass None to elf.
            if "PixelPitch" in g:
                raw = g["PixelPitch"][()]
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                result["pixel_pitch"] = raw if raw else None
            else:
                result["pixel_pitch"] = defaults["pixel_pitch"]

            # BlockwiseWatershed is absent in v0.1 projects; SerialDefaultSlot
            # sets it to False in that case, meaning ilastik ran the watershed
            # on the full crop (not blockwise).
            if "BlockwiseWatershed" in g:
                result["blockwise"] = bool(g["BlockwiseWatershed"][()])
            else:
                result["blockwise"] = False  # old project, no key → ilastik default was False

            return result
    except Exception:
        return dict(defaults)


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
    with _open_ilp_file(ilp_path) as f:
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
    with _open_ilp_file(ilp_path) as f:
        ef_group = f[APPLET_GROUP]["EdgeFeatures"][subname]
        df = _dataframe_from_hdf5(ef_group)
    return df


def _read_single_lane(ilp_path: str, lane: int):
    """
    Join EdgeFeatures with EdgeLabelsDict for one lane.

    Returns (X, y, feature_columns) or raises ValueError / KeyError if data
    is not available for the given lane.
    """
    import warnings

    labels_dict = read_edge_labels(ilp_path, lane=lane)
    if not labels_dict:
        raise ValueError(
            f"No edge labels found in lane {lane} of {ilp_path}."
        )

    try:
        features_df = read_edge_features(ilp_path, lane=lane)
    except KeyError:
        raise KeyError(
            f"EdgeFeatures not found for lane {lane} in {ilp_path}. "
            "Open the project in ilastik, trigger training (or live update), "
            "and re-save before extracting the classifier."
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
        warnings.warn(
            f"Lane {lane}: {missing} annotated edges were not found in the "
            "EdgeFeatures cache and will be skipped."
        )

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_vals, dtype=np.uint8),
        feature_cols,
    )


def read_training_data(ilp_path: str, lane=None):
    """
    Join EdgeFeatures with EdgeLabelsDict to produce a labeled training set.

    Parameters
    ----------
    ilp_path : str
    lane : int or None
        If None (default), all lanes with saved labels are read and their
        training data is concatenated.  Pass an integer to read a single lane.

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
    columns are excluded from X.  When reading multiple lanes, the feature
    column names are taken from the first successfully read lane (they are
    identical across lanes for a given project).
    """
    import warnings

    if lane is not None:
        return _read_single_lane(ilp_path, lane)

    lanes = discover_lanes(ilp_path)
    if not lanes:
        raise ValueError(f"No edge label groups found in {ilp_path}.")

    all_X, all_y, feature_cols = [], [], None
    for l in lanes:
        try:
            X, y, cols = _read_single_lane(ilp_path, l)
            all_X.append(X)
            all_y.append(y)
            if feature_cols is None:
                feature_cols = cols
        except (KeyError, ValueError) as exc:
            warnings.warn(f"Skipping lane {l}: {exc}")

    if not all_X:
        raise ValueError(
            f"No usable training data found in any lane of {ilp_path}. "
            "Make sure at least one lane has both labels and cached features."
        )

    return np.concatenate(all_X), np.concatenate(all_y), feature_cols
