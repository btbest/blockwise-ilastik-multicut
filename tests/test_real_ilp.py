"""
tests/test_real_ilp.py
======================
Tests that exercise ilp_reader.py, fit_classifier.py, and the helper
functions in multicut_from_ilp.py against the genuine ilastik project at
libs/example_mc_project.ilp.

Known facts about that project (verified by inspecting the HDF5 file):
  - 2 lanes (0 and 1)
  - lane 0: 197 edges cached, 10 manually labelled (8 merge, 2 split)
  - lane 1: 350 edges cached,  2 manually labelled (1 merge, 1 split)
  - 9 features per edge (3 from "Raw Data", 6 from "wsdt boundary channel")
  - Feature columns use numpy array repr in the HDF5 column_index dataset
    (not a plain Python list repr), which is the format written by
    recent versions of ilastikrag.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXAMPLE_ILP = REPO_ROOT / "libs" / "example_mc_project.ilp"

# Skip every test in this module when the file is absent (e.g. shallow CI
# checkouts that don't pull large binaries).
pytestmark = pytest.mark.skipif(
    not EXAMPLE_ILP.exists(),
    reason=f"Real ILP fixture not found at {EXAMPLE_ILP}",
)


# ---------------------------------------------------------------------------
# ilp_reader – discover_lanes
# ---------------------------------------------------------------------------


def test_discover_lanes_returns_both_lanes():
    from ilp_reader import discover_lanes

    lanes = discover_lanes(str(EXAMPLE_ILP))
    assert lanes == [0, 1]


def test_discover_lanes_sorted():
    from ilp_reader import discover_lanes

    lanes = discover_lanes(str(EXAMPLE_ILP))
    assert lanes == sorted(lanes)


# ---------------------------------------------------------------------------
# ilp_reader – read_feature_names
# ---------------------------------------------------------------------------


def test_read_feature_names_channels():
    from ilp_reader import read_feature_names

    fn = read_feature_names(str(EXAMPLE_ILP))
    assert set(fn.keys()) == {"Raw Data", "wsdt boundary channel"}


def test_read_feature_names_raw_data():
    from ilp_reader import read_feature_names

    fn = read_feature_names(str(EXAMPLE_ILP))
    assert fn["Raw Data"] == [
        "standard_sp_mean",
        "standard_sp_quantiles_10",
        "standard_sp_quantiles_90",
    ]


def test_read_feature_names_wsdt():
    from ilp_reader import read_feature_names

    fn = read_feature_names(str(EXAMPLE_ILP))
    assert fn["wsdt boundary channel"] == [
        "edgeregion_edge_regionradii_0",
        "edgeregion_edge_regionradii_1",
        "edgeregion_edge_regionradii_2",
        "standard_edge_mean",
        "standard_edge_quantiles_10",
        "standard_edge_quantiles_90",
    ]


def test_read_feature_names_total_count():
    from ilp_reader import read_feature_names

    fn = read_feature_names(str(EXAMPLE_ILP))
    total = sum(len(v) for v in fn.values())
    assert total == 9


# ---------------------------------------------------------------------------
# ilp_reader – read_edge_labels
# ---------------------------------------------------------------------------


def test_read_edge_labels_lane0_count():
    from ilp_reader import read_edge_labels

    labels = read_edge_labels(str(EXAMPLE_ILP), lane=0)
    assert len(labels) == 10


def test_read_edge_labels_lane0_classes():
    from ilp_reader import read_edge_labels

    labels = read_edge_labels(str(EXAMPLE_ILP), lane=0)
    values = set(labels.values())
    assert values == {1, 2}, "Expected both merge (1) and split (2) labels"


def test_read_edge_labels_lane0_merge_count():
    from ilp_reader import read_edge_labels

    labels = read_edge_labels(str(EXAMPLE_ILP), lane=0)
    assert sum(1 for v in labels.values() if v == 1) == 8


def test_read_edge_labels_lane0_split_count():
    from ilp_reader import read_edge_labels

    labels = read_edge_labels(str(EXAMPLE_ILP), lane=0)
    assert sum(1 for v in labels.values() if v == 2) == 2


def test_read_edge_labels_lane1_count():
    from ilp_reader import read_edge_labels

    labels = read_edge_labels(str(EXAMPLE_ILP), lane=1)
    assert len(labels) == 2


def test_read_edge_labels_lane1_both_classes():
    from ilp_reader import read_edge_labels

    labels = read_edge_labels(str(EXAMPLE_ILP), lane=1)
    values = set(labels.values())
    assert values == {1, 2}


def test_read_edge_labels_keys_are_tuples():
    from ilp_reader import read_edge_labels

    labels = read_edge_labels(str(EXAMPLE_ILP), lane=0)
    for key in labels:
        assert isinstance(key, tuple) and len(key) == 2


def test_read_edge_labels_sp_ids_positive():
    from ilp_reader import read_edge_labels

    labels = read_edge_labels(str(EXAMPLE_ILP), lane=0)
    for sp1, sp2 in labels:
        assert sp1 > 0 and sp2 > 0


# ---------------------------------------------------------------------------
# ilp_reader – read_edge_features
# ---------------------------------------------------------------------------


def test_read_edge_features_lane0_shape():
    from ilp_reader import read_edge_features

    df = read_edge_features(str(EXAMPLE_ILP), lane=0)
    assert df.shape == (197, 11)


def test_read_edge_features_lane1_shape():
    from ilp_reader import read_edge_features

    df = read_edge_features(str(EXAMPLE_ILP), lane=1)
    assert df.shape == (350, 11)


def test_read_edge_features_columns_start_with_sp():
    from ilp_reader import read_edge_features

    df = read_edge_features(str(EXAMPLE_ILP), lane=0)
    assert list(df.columns[:2]) == ["sp1", "sp2"]


def test_read_edge_features_n_feature_columns():
    from ilp_reader import read_edge_features

    df = read_edge_features(str(EXAMPLE_ILP), lane=0)
    feature_cols = [c for c in df.columns if c not in ("sp1", "sp2")]
    assert len(feature_cols) == 9


def test_read_edge_features_column_names_contain_channel():
    from ilp_reader import read_edge_features

    df = read_edge_features(str(EXAMPLE_ILP), lane=0)
    feature_cols = [c for c in df.columns if c not in ("sp1", "sp2")]
    # Each feature column should mention either the raw or boundary channel
    for col in feature_cols:
        assert "Raw Data" in col or "wsdt boundary channel" in col, col


def test_read_edge_features_sp1_less_than_sp2():
    """ilastikrag stores edges with sp1 < sp2 by convention."""
    from ilp_reader import read_edge_features

    df = read_edge_features(str(EXAMPLE_ILP), lane=0)
    assert (df["sp1"] < df["sp2"]).all()


def test_read_edge_features_no_self_loops():
    from ilp_reader import read_edge_features

    df = read_edge_features(str(EXAMPLE_ILP), lane=0)
    assert (df["sp1"] != df["sp2"]).all()


def test_read_edge_features_consistent_columns_across_lanes():
    from ilp_reader import read_edge_features

    df0 = read_edge_features(str(EXAMPLE_ILP), lane=0)
    df1 = read_edge_features(str(EXAMPLE_ILP), lane=1)
    assert list(df0.columns) == list(df1.columns)


# ---------------------------------------------------------------------------
# ilp_reader – read_training_data (all lanes)
# ---------------------------------------------------------------------------


def test_read_training_data_all_lanes_x_shape():
    from ilp_reader import read_training_data

    X, y, _ = read_training_data(str(EXAMPLE_ILP))
    assert X.shape == (12, 9)  # 10 from lane 0 + 2 from lane 1


def test_read_training_data_all_lanes_y_shape():
    from ilp_reader import read_training_data

    X, y, _ = read_training_data(str(EXAMPLE_ILP))
    assert y.shape == (12,)


def test_read_training_data_all_lanes_x_dtype():
    from ilp_reader import read_training_data

    X, y, _ = read_training_data(str(EXAMPLE_ILP))
    assert X.dtype == np.float32


def test_read_training_data_all_lanes_y_dtype():
    from ilp_reader import read_training_data

    X, y, _ = read_training_data(str(EXAMPLE_ILP))
    assert y.dtype == np.uint8


def test_read_training_data_all_lanes_classes():
    from ilp_reader import read_training_data

    _, y, _ = read_training_data(str(EXAMPLE_ILP))
    assert set(y.tolist()) == {1, 2}


def test_read_training_data_all_lanes_feature_cols_count():
    from ilp_reader import read_training_data

    _, _, cols = read_training_data(str(EXAMPLE_ILP))
    assert len(cols) == 9


def test_read_training_data_feature_cols_no_sp():
    from ilp_reader import read_training_data

    _, _, cols = read_training_data(str(EXAMPLE_ILP))
    assert "sp1" not in cols and "sp2" not in cols


# ---------------------------------------------------------------------------
# ilp_reader – read_training_data (single lane)
# ---------------------------------------------------------------------------


def test_read_training_data_lane0_x_shape():
    from ilp_reader import read_training_data

    X, y, _ = read_training_data(str(EXAMPLE_ILP), lane=0)
    assert X.shape == (10, 9)


def test_read_training_data_lane0_y_shape():
    from ilp_reader import read_training_data

    X, y, _ = read_training_data(str(EXAMPLE_ILP), lane=0)
    assert y.shape == (10,)


def test_read_training_data_lane1_x_shape():
    from ilp_reader import read_training_data

    X, y, _ = read_training_data(str(EXAMPLE_ILP), lane=1)
    assert X.shape == (2, 9)


def test_read_training_data_lane1_classes():
    from ilp_reader import read_training_data

    _, y, _ = read_training_data(str(EXAMPLE_ILP), lane=1)
    assert set(y.tolist()) == {1, 2}


def test_read_training_data_x_no_nan():
    from ilp_reader import read_training_data

    X, _, _ = read_training_data(str(EXAMPLE_ILP))
    assert not np.isnan(X).any()


def test_read_training_data_x_finite():
    from ilp_reader import read_training_data

    X, _, _ = read_training_data(str(EXAMPLE_ILP))
    assert np.isfinite(X).all()


# ---------------------------------------------------------------------------
# fit_classifier – fit_rf_from_ilp
# ---------------------------------------------------------------------------


def test_fit_rf_returns_classifier():
    from fit_classifier import fit_rf_from_ilp
    from sklearn.ensemble import RandomForestClassifier

    rf = fit_rf_from_ilp(str(EXAMPLE_ILP), n_estimators=10, verbose=False)
    assert isinstance(rf, RandomForestClassifier)


def test_fit_rf_classes():
    from fit_classifier import fit_rf_from_ilp

    rf = fit_rf_from_ilp(str(EXAMPLE_ILP), n_estimators=10, verbose=False)
    assert list(rf.classes_) == [1, 2]


def test_fit_rf_n_features():
    from fit_classifier import fit_rf_from_ilp

    rf = fit_rf_from_ilp(str(EXAMPLE_ILP), n_estimators=10, verbose=False)
    assert rf.n_features_in_ == 9


def test_fit_rf_predict_proba_shape():
    from fit_classifier import fit_rf_from_ilp
    from ilp_reader import read_training_data

    rf = fit_rf_from_ilp(str(EXAMPLE_ILP), n_estimators=10, verbose=False)
    X, _, _ = read_training_data(str(EXAMPLE_ILP))
    proba = rf.predict_proba(X)
    assert proba.shape == (12, 2)


def test_fit_rf_predict_proba_sums_to_one():
    from fit_classifier import fit_rf_from_ilp
    from ilp_reader import read_training_data

    rf = fit_rf_from_ilp(str(EXAMPLE_ILP), n_estimators=10, verbose=False)
    X, _, _ = read_training_data(str(EXAMPLE_ILP))
    proba = rf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_fit_rf_split_proba_in_unit_interval():
    """Column 1 of predict_proba (class 2 = split) must be in [0, 1]."""
    from fit_classifier import fit_rf_from_ilp
    from ilp_reader import read_training_data

    rf = fit_rf_from_ilp(str(EXAMPLE_ILP), n_estimators=10, verbose=False)
    X, _, _ = read_training_data(str(EXAMPLE_ILP))
    split_col = int(np.argmax(rf.classes_))
    p_split = rf.predict_proba(X)[:, split_col]
    assert (p_split >= 0).all() and (p_split <= 1).all()


def test_fit_rf_single_lane():
    from fit_classifier import fit_rf_from_ilp

    rf = fit_rf_from_ilp(str(EXAMPLE_ILP), lane=0, n_estimators=10, verbose=False)
    assert rf.n_features_in_ == 9
    assert list(rf.classes_) == [1, 2]


def test_fit_rf_reproducible():
    from fit_classifier import fit_rf_from_ilp
    from ilp_reader import read_training_data

    rf1 = fit_rf_from_ilp(
        str(EXAMPLE_ILP), n_estimators=10, random_state=0, verbose=False
    )
    rf2 = fit_rf_from_ilp(
        str(EXAMPLE_ILP), n_estimators=10, random_state=0, verbose=False
    )
    X, _, _ = read_training_data(str(EXAMPLE_ILP))
    np.testing.assert_array_equal(
        rf1.predict_proba(X), rf2.predict_proba(X)
    )


def test_fit_rf_pickle_roundtrip(tmp_path):
    from fit_classifier import fit_rf_from_ilp
    from ilp_reader import read_training_data

    rf = fit_rf_from_ilp(str(EXAMPLE_ILP), n_estimators=10, verbose=False)
    pkl = tmp_path / "rf.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(rf, fh)
    with open(pkl, "rb") as fh:
        rf2 = pickle.load(fh)

    X, _, _ = read_training_data(str(EXAMPLE_ILP))
    np.testing.assert_array_equal(rf.predict_proba(X), rf2.predict_proba(X))


# ---------------------------------------------------------------------------
# multicut_from_ilp – helper / utility tests
# ---------------------------------------------------------------------------


def test_parse_channel_spec_h5_with_key():
    from multicut_from_ilp import _parse_channel_spec

    name, path, key = _parse_channel_spec("My Channel:/data/vol.h5:/raw")
    assert name == "My Channel"
    assert path == "/data/vol.h5"
    assert key == "/raw"


def test_parse_channel_spec_zarr_no_key():
    from multicut_from_ilp import _parse_channel_spec

    name, path, key = _parse_channel_spec("Probabilities:/data/probs.zarr")
    assert name == "Probabilities"
    assert path == "/data/probs.zarr"
    assert key is None


def test_parse_channel_spec_invalid():
    from multicut_from_ilp import _parse_channel_spec

    with pytest.raises(ValueError):
        _parse_channel_spec("no_colon_at_all")


def test_multicut_reads_feature_names_from_real_ilp():
    """run_blockwise_multicut calls read_feature_names internally; verify
    it can parse the real ILP's FeatureNames without error."""
    from ilp_reader import read_feature_names

    fn = read_feature_names(str(EXAMPLE_ILP))
    assert isinstance(fn, dict)
    assert len(fn) > 0
