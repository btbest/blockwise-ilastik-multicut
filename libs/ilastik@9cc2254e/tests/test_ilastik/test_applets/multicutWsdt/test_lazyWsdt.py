"""Tests for deterministic blockwise watershed in OpWsdt.

Proves that:
1. Requesting disjoint ROIs produces the same superpixel IDs as requesting
   the full output.
2. Superpixel IDs are globally unique across independently-computed blocks.
3. The deterministic offset scheme produces consistent results regardless
   of request order.
"""
import numpy as np
import pytest
import vigra

from lazyflow.graph import Graph
from lazyflow.operators.opArrayPiper import OpArrayPiper

from ilastik.applets.wsdt.opWsdt import OpWsdt, OpCachedWsdt


@pytest.fixture
def graph():
    return Graph()


def _make_pmap(shape, seed=42):
    """Create a synthetic probability map with structure for watershed."""
    rng = np.random.RandomState(seed)
    # Create blob-like structures that will produce distinct superpixels
    pmap = rng.random(shape).astype("float32")
    # Smooth slightly so watershed produces reasonable regions
    for _ in range(2):
        pmap = vigra.filters.gaussianSmoothing(
            vigra.taggedView(pmap, "zyx" if len(shape) == 3 else "yx"),
            sigma=2.0,
        )
    pmap = np.asarray(pmap, dtype="float32")
    # Normalize to [0, 1]
    pmap = (pmap - pmap.min()) / (pmap.max() - pmap.min())
    return pmap


class TestDeterministicBlockwiseWatershed:
    """Test that OpWsdt produces deterministic, globally-unique superpixel IDs
    when different ROIs are requested independently."""

    @pytest.fixture
    def spatial_shape(self):
        return (64, 64, 64)

    @pytest.fixture
    def pmap_data(self, spatial_shape):
        pmap = _make_pmap(spatial_shape)
        return vigra.taggedView(pmap[..., np.newaxis], axistags="zyxc")

    @pytest.fixture
    def op_wsdt(self, graph, pmap_data):
        piper = OpArrayPiper(graph=graph)
        piper.Input.setValue(pmap_data)

        op = OpWsdt(graph=graph)
        op.Input.connect(piper.Output)
        op.ChannelSelections.setValue([0])
        op.Threshold.setValue(0.3)
        op.MinSize.setValue(10)
        op.Sigma.setValue(2.0)
        op.Alpha.setValue(0.9)
        op.BlockwiseWatershed.setValue(True)
        return op

    def test_full_output_has_nonzero_superpixels(self, op_wsdt):
        """Sanity check: the full output should contain superpixels."""
        full = op_wsdt.Superpixels[:].wait()
        assert full.max() > 0, "Watershed produced no superpixels"
        assert full.dtype == np.uint32

    def test_disjoint_rois_match_full_output(self, op_wsdt, spatial_shape):
        """Requesting disjoint ROIs should produce identical results to
        requesting the full volume."""
        full = op_wsdt.Superpixels[:].wait()

        # Split the volume into quadrants along the first axis
        mid = spatial_shape[0] // 2

        roi1_start = np.array([0, 0, 0, 0])
        roi1_stop = np.array([mid, spatial_shape[1], spatial_shape[2], 1])
        part1 = op_wsdt.Superpixels(roi1_start, roi1_stop).wait()

        roi2_start = np.array([mid, 0, 0, 0])
        roi2_stop = np.array([spatial_shape[0], spatial_shape[1], spatial_shape[2], 1])
        part2 = op_wsdt.Superpixels(roi2_start, roi2_stop).wait()

        # Reassemble
        reassembled = np.concatenate([part1, part2], axis=0)
        np.testing.assert_array_equal(
            reassembled, full,
            "Disjoint ROIs produced different results than full request"
        )

    def test_single_block_roi_matches_full_output(self, op_wsdt, spatial_shape):
        """A single canonical-block-sized ROI should match the corresponding
        region from the full output."""
        full = op_wsdt.Superpixels[:].wait()

        block_shape, _ = op_wsdt._get_canonical_block_config()
        # Request just the first canonical block
        roi_start = np.array([0] * len(spatial_shape) + [0])
        roi_stop = np.array(list(block_shape) + [1])
        roi_stop = np.minimum(roi_stop, np.array(list(spatial_shape) + [1]))

        block_result = op_wsdt.Superpixels(roi_start, roi_stop).wait()
        expected = full[tuple(slice(s, e) for s, e in zip(roi_start, roi_stop))]

        np.testing.assert_array_equal(
            block_result, expected,
            "Single block ROI does not match full output"
        )

    def test_ids_globally_unique_across_blocks(self, op_wsdt, spatial_shape):
        """Superpixel IDs from different canonical blocks must not overlap."""
        full = op_wsdt.Superpixels[:].wait()[..., 0]  # drop channel

        block_shape, _ = op_wsdt._get_canonical_block_config()
        from lazyflow.roi import getIntersectingBlocks
        block_starts = getIntersectingBlocks(
            block_shape,
            ([0] * len(spatial_shape), list(spatial_shape))
        )

        all_id_sets = []
        for bs in block_starts:
            block_stop = np.minimum(np.array(bs) + np.array(block_shape), spatial_shape)
            slicing = tuple(slice(int(s), int(e)) for s, e in zip(bs, block_stop))
            block_data = full[slicing]
            ids = set(np.unique(block_data)) - {0}
            all_id_sets.append(ids)

        # Check pairwise that no IDs overlap between blocks
        for i in range(len(all_id_sets)):
            for j in range(i + 1, len(all_id_sets)):
                overlap = all_id_sets[i] & all_id_sets[j]
                assert len(overlap) == 0, (
                    f"Blocks {i} and {j} share superpixel IDs: {overlap}"
                )

    def test_request_order_independence(self, op_wsdt, spatial_shape):
        """Results should be identical regardless of request order."""
        block_shape, _ = op_wsdt._get_canonical_block_config()
        from lazyflow.roi import getIntersectingBlocks
        block_starts = getIntersectingBlocks(
            block_shape,
            ([0] * len(spatial_shape), list(spatial_shape))
        )

        # Request blocks in forward order
        forward_results = {}
        for bs in block_starts:
            bs = tuple(bs)
            block_stop = tuple(min(int(s + b), int(sh))
                               for s, b, sh in zip(bs, block_shape, spatial_shape))
            roi_start = np.array(list(bs) + [0])
            roi_stop = np.array(list(block_stop) + [1])
            forward_results[bs] = op_wsdt.Superpixels(roi_start, roi_stop).wait().copy()

        # Request blocks in reverse order
        for bs in reversed(block_starts):
            bs = tuple(bs)
            block_stop = tuple(min(int(s + b), int(sh))
                               for s, b, sh in zip(bs, block_shape, spatial_shape))
            roi_start = np.array(list(bs) + [0])
            roi_stop = np.array(list(block_stop) + [1])
            reverse_result = op_wsdt.Superpixels(roi_start, roi_stop).wait()

            np.testing.assert_array_equal(
                reverse_result, forward_results[bs],
                f"Block at {bs}: forward and reverse request produced different results"
            )

    def test_jump_navigation_consistency(self, op_wsdt, spatial_shape):
        """Simulate jump navigation: request two distant, disjoint blocks
        and verify both match the full output."""
        full = op_wsdt.Superpixels[:].wait()

        block_shape, _ = op_wsdt._get_canonical_block_config()
        from lazyflow.roi import getIntersectingBlocks
        block_starts = getIntersectingBlocks(
            block_shape,
            ([0] * len(spatial_shape), list(spatial_shape))
        )

        if len(block_starts) < 2:
            pytest.skip("Volume too small for jump navigation test")

        # Request first and last block (maximally disjoint)
        for bs in [block_starts[0], block_starts[-1]]:
            bs = np.array(bs)
            block_stop = np.minimum(bs + np.array(block_shape), spatial_shape)
            roi_start = np.append(bs, 0)
            roi_stop = np.append(block_stop, 1)

            result = op_wsdt.Superpixels(roi_start, roi_stop).wait()
            expected = full[tuple(slice(int(s), int(e)) for s, e in zip(roi_start, roi_stop))]

            np.testing.assert_array_equal(
                result, expected,
                f"Jump navigation: block at {bs} does not match full output"
            )


class TestCachedWsdtBlockAlignment:
    """Test that OpCachedWsdt sets the cache block shape to match
    the canonical watershed block shape."""

    @pytest.fixture
    def spatial_shape(self):
        return (64, 64, 64)

    @pytest.fixture
    def pmap_data(self, spatial_shape):
        pmap = _make_pmap(spatial_shape)
        return vigra.taggedView(pmap[..., np.newaxis], axistags="zyxc")

    @pytest.fixture
    def op_cached_wsdt(self, graph, pmap_data):
        piper = OpArrayPiper(graph=graph)
        piper.Input.setValue(pmap_data)

        op = OpCachedWsdt(graph=graph)
        op.Input.connect(piper.Output)
        op.FreezeCache.setValue(False)
        op.BlockwiseWatershed.setValue(True)
        return op

    def test_cache_block_shape_matches_watershed(self, op_cached_wsdt):
        """The cache should use the same block shape as the watershed."""
        ws_block_shape = op_cached_wsdt._opWsdt.Superpixels.meta.ideal_blockshape
        cache_block_shape = op_cached_wsdt._opCache.BlockShape.value
        assert ws_block_shape is not None
        assert tuple(cache_block_shape) == tuple(ws_block_shape), (
            f"Cache block shape {cache_block_shape} != watershed block shape {ws_block_shape}"
        )

    def test_cached_output_matches_uncached(self, op_cached_wsdt, graph, pmap_data):
        """Cached output should be identical to direct OpWsdt output."""
        # Get cached result
        cached = op_cached_wsdt.Superpixels[:].wait()

        # Get direct result
        piper = OpArrayPiper(graph=graph)
        piper.Input.setValue(pmap_data)

        direct_op = OpWsdt(graph=graph)
        direct_op.Input.connect(piper.Output)
        direct_op.ChannelSelections.setValue([0])
        direct_op.Threshold.setValue(0.5)
        direct_op.MinSize.setValue(100)
        direct_op.Sigma.setValue(3.0)
        direct_op.Alpha.setValue(0.9)
        direct_op.BlockwiseWatershed.setValue(True)

        direct = direct_op.Superpixels[:].wait()

        np.testing.assert_array_equal(
            cached, direct,
            "Cached output differs from direct OpWsdt output"
        )
