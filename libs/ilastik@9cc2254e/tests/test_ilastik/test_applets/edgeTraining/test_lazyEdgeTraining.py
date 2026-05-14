"""Tests for ROI-local (lazy) edge training.

Proves that:
1. In lazy mode, requesting edge-related outputs (features, predictions)
   only triggers requests for the superpixel/voxel data in the respective
   ROI — NOT the full volume.
2. Incrementally computed edge features match those computed from the full
   volume for the same edges.
3. The classifier can train on incrementally cached features and produce
   valid predictions.
"""
import numpy as np
import pytest
import vigra

from ilastikrag.util import generate_random_voronoi

from lazyflow.graph import Graph, Operator, InputSlot, OutputSlot

from ilastik.applets.edgeTraining.opEdgeTraining import (
    OpEdgeTraining,
    OpIncrementalEdgeFeatures,
)


@pytest.fixture
def graph():
    return Graph()


class OpSpyArrayPiper(Operator):
    """A transparent piper that records which ROIs were requested.

    Used to verify that lazy-mode operators only request the ROI they need,
    not the full volume.
    """

    Input = InputSlot()
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requested_rois = []

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        self.requested_rois.append((tuple(roi.start), tuple(roi.stop)))
        self.Input(roi.start, roi.stop).writeInto(result).wait()

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi.start, roi.stop)

    def reset_spy(self):
        self.requested_rois.clear()


def _make_test_data(spatial_shape=(60, 60, 60), n_superpixels=50, seed=42):
    """Create test superpixels and voxel data.

    Returns:
        superpixels: uint32 vigra array with channel axis (zyxc)
        voxel_data: float32 vigra array with channel axis (zyxc)
    """
    superpixels = generate_random_voronoi(spatial_shape, n_superpixels)
    superpixels = superpixels.insertChannelAxis()
    voxel_data = superpixels.astype(np.float32)
    return superpixels, voxel_data


class TestOpIncrementalEdgeFeatures:
    """Test the incremental edge feature computation operator."""

    @pytest.fixture
    def spatial_shape(self):
        return (60, 60, 60)

    @pytest.fixture
    def test_data(self, spatial_shape):
        return _make_test_data(spatial_shape)

    @pytest.fixture
    def op_incremental(self, graph, test_data):
        superpixels, voxel_data = test_data

        op = OpIncrementalEdgeFeatures(graph=graph)
        op.Superpixels.setValue(superpixels)
        op.VoxelData.setValue(voxel_data, extra_meta={"channel_names": ["Grayscale"]})
        op.WatershedSelectedInput.setValue(voxel_data)
        op.FeatureNames.setValue({"Grayscale": ["standard_edge_mean"]})
        op.TrainRandomForest.setValue(True)
        return op

    def test_empty_before_any_roi(self, op_incremental):
        """Before any ROI is computed, features should be empty."""
        df = op_incremental.EdgeFeaturesDataFrame.value
        assert len(df) == 0

    def test_compute_for_roi_returns_edges(self, op_incremental, spatial_shape):
        """Computing features for a sub-ROI should return edges from that region."""
        half = spatial_shape[0] // 2
        roi_start = [0, 0, 0]
        roi_stop = [half, spatial_shape[1], spatial_shape[2]]

        new_edges = op_incremental.computeForRoi(roi_start, roi_stop)
        assert len(new_edges) > 0, "Expected edges in sub-ROI"

        # The accumulated features should now contain these edges
        df = op_incremental.EdgeFeaturesDataFrame.value
        assert len(df) == len(new_edges)

    def test_incremental_accumulation(self, op_incremental, spatial_shape):
        """Computing features for multiple ROIs should accumulate edges."""
        half = spatial_shape[0] // 2

        # First half
        edges1 = op_incremental.computeForRoi(
            [0, 0, 0],
            [half, spatial_shape[1], spatial_shape[2]]
        )
        count_after_first = len(op_incremental.EdgeFeaturesDataFrame.value)

        # Second half
        edges2 = op_incremental.computeForRoi(
            [half, 0, 0],
            [spatial_shape[0], spatial_shape[1], spatial_shape[2]]
        )
        count_after_second = len(op_incremental.EdgeFeaturesDataFrame.value)

        # Total should be >= first half (some edges may be shared at boundary)
        assert count_after_second >= count_after_first
        assert count_after_second >= len(edges2)

    def test_duplicate_roi_does_not_duplicate_edges(self, op_incremental, spatial_shape):
        """Computing the same ROI twice should not add duplicate edges."""
        roi_start = [0, 0, 0]
        roi_stop = list(spatial_shape)

        op_incremental.computeForRoi(roi_start, roi_stop)
        count_first = len(op_incremental.EdgeFeaturesDataFrame.value)

        op_incremental.computeForRoi(roi_start, roi_stop)
        count_second = len(op_incremental.EdgeFeaturesDataFrame.value)

        assert count_second == count_first, "Duplicate ROI added duplicate edges"

    def test_disjoint_rois_accumulate_independently(self, op_incremental, spatial_shape):
        """Disjoint ROIs (simulating jump navigation) should add independent edges."""
        third = spatial_shape[0] // 3

        # Request bottom third
        edges_bottom = op_incremental.computeForRoi(
            [0, 0, 0],
            [third, spatial_shape[1], spatial_shape[2]]
        )

        # Request top third (skip middle — like jump navigation)
        edges_top = op_incremental.computeForRoi(
            [2 * third, 0, 0],
            [spatial_shape[0], spatial_shape[1], spatial_shape[2]]
        )

        df = op_incremental.EdgeFeaturesDataFrame.value
        assert len(df) >= len(edges_bottom)
        assert len(df) >= len(edges_top)

    def test_reset_clears_cache(self, op_incremental, spatial_shape):
        """resetCache should clear all accumulated features."""
        op_incremental.computeForRoi([0, 0, 0], list(spatial_shape))
        assert len(op_incremental.EdgeFeaturesDataFrame.value) > 0

        op_incremental.resetCache()
        assert len(op_incremental.EdgeFeaturesDataFrame.value) == 0


class TestLazyEdgeTrainingRequestLocality:
    """Test that in lazy mode, edge-related outputs only request
    the input data for the respective ROI, not the full volume."""

    @pytest.fixture
    def spatial_shape(self):
        return (60, 60, 60)

    @pytest.fixture
    def test_data(self, spatial_shape):
        return _make_test_data(spatial_shape)

    def test_incremental_features_only_request_roi(self, graph, test_data, spatial_shape):
        """OpIncrementalEdgeFeatures.computeForRoi should only request
        superpixels and voxel data for the given ROI, not the full volume."""
        superpixels, voxel_data = test_data

        sp_spy = OpSpyArrayPiper(graph=graph)
        sp_spy.Input.setValue(superpixels)

        vd_spy = OpSpyArrayPiper(graph=graph)
        vd_spy.Input.setValue(voxel_data, extra_meta={"channel_names": ["Grayscale"]})

        op = OpIncrementalEdgeFeatures(graph=graph)
        op.Superpixels.connect(sp_spy.Output)
        op.VoxelData.connect(vd_spy.Output)
        op.WatershedSelectedInput.connect(vd_spy.Output)
        op.FeatureNames.setValue({"Grayscale": ["standard_edge_mean"]})
        op.TrainRandomForest.setValue(True)

        # Request only the first half
        half = spatial_shape[0] // 2
        roi_start = [0, 0, 0]
        roi_stop = [half, spatial_shape[1], spatial_shape[2]]

        sp_spy.reset_spy()
        vd_spy.reset_spy()

        op.computeForRoi(roi_start, roi_stop)

        # Verify that NO request covered the full volume
        full_shape = superpixels.shape
        for (req_start, req_stop) in sp_spy.requested_rois:
            req_shape = tuple(e - s for s, e in zip(req_start, req_stop))
            # The request should be smaller than the full volume in at least one dim
            assert req_shape != full_shape, (
                f"Superpixel request covered full volume: {req_start} to {req_stop}"
            )

        for (req_start, req_stop) in vd_spy.requested_rois:
            req_shape = tuple(e - s for s, e in zip(req_start, req_stop))
            assert req_shape != voxel_data.shape, (
                f"VoxelData request covered full volume: {req_start} to {req_stop}"
            )

        # Verify requests were within the ROI (with channel dim)
        for (req_start, req_stop) in sp_spy.requested_rois:
            # Spatial dims should be within [roi_start, roi_stop]
            for d in range(len(roi_start)):
                assert req_start[d] >= roi_start[d], (
                    f"SP request start {req_start} is before ROI start {roi_start} in dim {d}"
                )
                assert req_stop[d] <= roi_stop[d], (
                    f"SP request stop {req_stop} is after ROI stop {roi_stop} in dim {d}"
                )


class TestLazyModeEdgeTraining:
    """Integration test: verify the full lazy-mode edge training pipeline."""

    @pytest.fixture
    def spatial_shape(self):
        return (60, 60, 60)

    @pytest.fixture
    def test_data(self, spatial_shape):
        return _make_test_data(spatial_shape)

    def test_lazy_mode_classifier_training(self, graph, test_data, spatial_shape):
        """In lazy mode, the classifier should train on incrementally
        cached features and produce valid predictions."""
        superpixels, voxel_data = test_data

        multilane_op = OpEdgeTraining(graph=graph)
        multilane_op.VoxelData.resize(1)
        multilane_op.LazyMode.setValue(True)

        op_view = multilane_op.getLane(0)

        op_view.VoxelData.setValue(voxel_data, extra_meta={"channel_names": ["Grayscale"]})
        op_view.Superpixels.setValue(superpixels)
        op_view.WatershedSelectedInput.setValue(voxel_data)
        op_view.TrainRandomForest.setValue(True)

        multilane_op.FeatureNames.setValue({"Grayscale": ["standard_edge_mean", "standard_edge_count"]})

        # Compute features for the full volume via incremental path
        incremental_op = multilane_op.opIncrementalEdgeFeatures.getLane(0)
        incremental_op.computeForRoi([0, 0, 0], list(spatial_shape))

        # Verify we have features
        df = incremental_op.EdgeFeaturesDataFrame.value
        assert len(df) > 0, "No edges computed"

        # Pick edges to label (from the incremental features DataFrame)
        edge_ids = list(zip(df["sp1"].astype(int).values, df["sp2"].astype(int).values))
        assert len(edge_ids) >= 4, f"Need at least 4 edges, got {len(edge_ids)}"

        labels = {
            tuple(edge_ids[0]): 1,  # OFF
            tuple(edge_ids[1]): 1,  # OFF
            tuple(edge_ids[2]): 2,  # ON
            tuple(edge_ids[3]): 2,  # ON
        }

        op_view.EdgeLabelsDict.setValue(labels)
        op_view.FreezeClassifier.setValue(False)

        # Verify predictions are produced
        assert op_view.EdgeProbabilities.ready()
        edge_probs = op_view.EdgeProbabilities.value
        assert edge_probs is not None
        assert len(edge_probs) == len(df), (
            f"Expected {len(df)} predictions, got {len(edge_probs)}"
        )

        # Verify EdgeProbabilitiesDict works
        assert op_view.EdgeProbabilitiesDict.ready()
        edge_prob_dict = op_view.EdgeProbabilitiesDict.value
        assert len(edge_prob_dict) > 0

    def test_lazy_vs_global_features_agree(self, graph, test_data, spatial_shape):
        """Edge features from the incremental path should match those from
        the global path for the same edges (when both cover the full volume)."""
        superpixels, voxel_data = test_data

        # --- Global path ---
        global_op = OpEdgeTraining(graph=graph)
        global_op.VoxelData.resize(1)
        global_op.LazyMode.setValue(False)

        g_view = global_op.getLane(0)
        g_view.VoxelData.setValue(voxel_data, extra_meta={"channel_names": ["Grayscale"]})
        g_view.Superpixels.setValue(superpixels)
        g_view.WatershedSelectedInput.setValue(voxel_data)
        g_view.TrainRandomForest.setValue(True)
        global_op.FeatureNames.setValue({"Grayscale": ["standard_edge_mean"]})

        global_features_df = g_view.opComputeEdgeFeatures.EdgeFeaturesDataFrame.value

        # --- Lazy/incremental path ---
        lazy_op = OpEdgeTraining(graph=graph)
        lazy_op.VoxelData.resize(1)
        lazy_op.LazyMode.setValue(True)

        l_view = lazy_op.getLane(0)
        l_view.VoxelData.setValue(voxel_data, extra_meta={"channel_names": ["Grayscale"]})
        l_view.Superpixels.setValue(superpixels)
        l_view.WatershedSelectedInput.setValue(voxel_data)
        l_view.TrainRandomForest.setValue(True)
        lazy_op.FeatureNames.setValue({"Grayscale": ["standard_edge_mean"]})

        # Compute features for the full volume incrementally
        incremental_op = lazy_op.opIncrementalEdgeFeatures.getLane(0)
        incremental_op.computeForRoi([0, 0, 0], list(spatial_shape))
        lazy_features_df = incremental_op.EdgeFeaturesDataFrame.value

        # Both should have the same edges (same superpixel input)
        global_edges = set(zip(global_features_df["sp1"].astype(int), global_features_df["sp2"].astype(int)))
        lazy_edges = set(zip(lazy_features_df["sp1"].astype(int), lazy_features_df["sp2"].astype(int)))
        assert global_edges == lazy_edges, (
            f"Edge sets differ: {len(global_edges)} global vs {len(lazy_edges)} lazy"
        )

        # Features should be close (same data, same computation)
        feature_cols = [c for c in global_features_df.columns if c not in ("sp1", "sp2")]
        for col in feature_cols:
            global_vals = global_features_df.set_index(["sp1", "sp2"])[col].sort_index()
            lazy_vals = lazy_features_df.set_index(["sp1", "sp2"])[col].sort_index()
            np.testing.assert_allclose(
                global_vals.values, lazy_vals.values,
                rtol=1e-5, atol=1e-5,
                err_msg=f"Feature '{col}' values differ between global and lazy paths"
            )
