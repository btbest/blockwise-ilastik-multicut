from collections import OrderedDict
import numpy as np
from typing import Optional, Sequence

from elf.segmentation.watershed import distance_transform_watershed
from elf.parallel.common import get_blocking

import vigra

from lazyflow.utility import OrderedSignal
from lazyflow.request import Request
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.roi import roiToSlice
from lazyflow.operators import OpBlockedArrayCache, OpMetadataInjector
from lazyflow.operators.generic import OpPixelOperator
from lazyflow.utility.timer import Timer

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import logging

logger = logging.getLogger(__name__)


def parallel_watershed(
    data: np.ndarray,
    threshold: float,
    sigma_seeds: float,
    sigma_weights: float,
    minsize: int,
    alpha: float,
    pixel_pitch: Sequence[float],
    non_max_suppression: bool,
    block_shape: Optional[Sequence[int]] = None,
    halo: Optional[Sequence[int]] = None,
    max_workers: Optional[int] = None,
):
    """Parallel dt watershed with hard block boundaries.

    parallel wrapper around elf.segmentation.watershed.distance_transform_watershed

    Args:
      data: data to run watershed on, ndarray with either 2 or 3 dims
      threshold: data will be thresholded at this value, distance transform will be calculated from
        the remaining mask
      sigma_seeds: smoothing factor that is applied to the watershed seed map
      sigma_weights: smoothing factor that is applied to the watershed weight map
      minsize: minimal boundary size in pixels
      alpha: how to blend data and distance transform
      pixel_pitch: anisotropy factor: pixel distance along the three axes
      non_max_suppression: flag to enable apply non-maxmimum suppression to filter out seeds
      block_shape: size of blocks to process, defaults to 128 for 3D, 512 for 2D in each spacial dimension
      halo: portion of each block to discard after processing for smoother boundary regions
        if not specified: 10 voxels around the block in each direction
      max_workers: if not specified or None, will use number of workers in the global Requests threadpool

    """

    shape = data.shape
    ndim = len(shape)

    assert ndim in [2, 3], "Watershed segmentor will only work on 2D and 3D data"

    base_block = 512 if ndim == 2 else 128
    # check for None arguments and set to default values
    block_shape = (base_block,) * ndim if block_shape is None else block_shape
    # nifty requires the halo shape to be of type list
    halo = [10] * ndim if halo is None else halo

    if max_workers is None:
        max_workers = max(1, Request.global_thread_pool.num_workers)

    logger.info(f"blockwise watershed with {max_workers} threads.")

    blocking = get_blocking(data, block_shape, roi=None, n_threads=max_workers)
    n_blocks = blocking.numberOfBlocks

    labels = np.zeros_like(data, dtype=np.uint32)

    # watershed for a single block
    def ws_block(block_index):
        nonlocal labels

        # get the block with halo and the slicings corresponding to
        # the block with halo, the block without halo and the
        # block without halo in loocal coordinates
        block = blocking.getBlockWithHalo(blockIndex=block_index, halo=halo)
        inner_slicing = roiToSlice(block.innerBlock.begin, block.innerBlock.end)
        outer_slicing = roiToSlice(block.outerBlock.begin, block.outerBlock.end)
        inner_local_slicing = roiToSlice(block.innerBlockLocal.begin, block.innerBlockLocal.end)

        with Timer() as btimer:
            # write watershed result to the label array
            ws_outer, _ = distance_transform_watershed(
                data[outer_slicing],
                threshold,
                sigma_seeds,
                sigma_weights,
                minsize,
                alpha,
                pixel_pitch,
                non_max_suppression,
            )

        logger.debug(
            f"processing block {block_index} {block.outerBlock.begin}-{block.outerBlock.end} took {btimer.seconds()}"
        )

        # elf started returning uint64 in 0.46. Casting here is not dangerous.
        # Up to now vigra is still used to produce the watershed in elf internally.
        ws_outer = ws_outer.astype("uint32")
        ws_inner = vigra.analysis.labelMultiArray(ws_outer[inner_local_slicing])

        labels[inner_slicing] = ws_inner
        # return the max-id for this block, that will be used as offset
        return ws_inner.max()

    with Timer() as wstimer:
        # run the watershed blocks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            offsets = np.fromiter(executor.map(ws_block, range(n_blocks)), dtype=np.int64)

    logger.info(f"parallel ws took {wstimer.seconds()} s")
    offsets = np.cumsum(offsets)

    # add the offset to blocks to make ids unique
    def add_offset_block(block_index):
        block = blocking.getBlock(block_index)
        block = roiToSlice(block.begin, block.end)
        labels[block] += offsets[block_index - 1]

    # add offsets in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        concurrent.futures.wait([executor.submit(add_offset_block, block_index) for block_index in range(1, n_blocks)])

    return labels, offsets[-1]


class OpWsdt(Operator):
    """Blockwise watershed operator with deterministic superpixel IDs.

    Each canonical block in the volume gets a deterministic ID offset based on
    its block index, so superpixel IDs are globally unique regardless of the
    order or subset of blocks computed. This allows arbitrary ROIs to be
    requested independently and still produce consistent IDs.
    """

    # Can be multi-channel (but you'll have to choose which channels you want to use)
    Input = InputSlot()

    # List input channels to use as the boundary map
    # (They'll be summed.)
    ChannelSelections = InputSlot(value=[0])

    Threshold = InputSlot(value=0.5)
    MinSize = InputSlot(value=100)
    Sigma = InputSlot(value=3.0)
    Alpha = InputSlot(value=0.9)
    PixelPitch = InputSlot(value=[])
    ApplyNonmaxSuppression = InputSlot(value=False)
    InvertPixelProbabilities = InputSlot(value=False)

    EnableDebugOutputs = InputSlot(value=False)

    BlockwiseWatershed = InputSlot(value=True)

    Superpixels = OutputSlot()

    # Upper bound on labels per canonical watershed block.
    # Block N gets IDs in range [N * MAX_LABELS_PER_BLOCK + 1, (N+1) * MAX_LABELS_PER_BLOCK].
    # With 128^3 blocks (~2M voxels), 2^21 is a safe upper bound.
    MAX_LABELS_PER_BLOCK = 2**21

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_results = None
        self.watershed_completed = OrderedSignal()

        self._opSelectedInput = OpSumChannels(parent=self)
        self._opSelectedInput.ChannelSelections.connect(self.ChannelSelections)
        self._opSelectedInput.InvertPixelProbabilities.connect(self.InvertPixelProbabilities)
        self._opSelectedInput.Input.connect(self.Input)

    def _get_canonical_block_config(self):
        """Return (block_shape, halo) for canonical watershed blocking."""
        spatial_shape = self.Input.meta.shape[:-1]  # drop channel
        ndim = len(spatial_shape)
        base_block = 512 if ndim == 2 else 128
        block_shape = tuple(min(base_block, s) for s in spatial_shape)
        halo = [10] * ndim
        return block_shape, halo

    def setupOutputs(self):
        if not self._opSelectedInput.Output.ready():
            self.Superpixels.meta.NOTREADY = True
            return

        assert self.Input.meta.getAxisKeys()[-1] == "c", "This operator assumes that channel is the last axis."
        self.Superpixels.meta.assignFrom(self.Input.meta)
        self.Superpixels.meta.shape = self.Input.meta.shape[:-1] + (1,)
        self.Superpixels.meta.dtype = np.uint32
        self.Superpixels.meta.display_mode = "random-colortable"

        # Set ideal_blockshape to match canonical watershed blocking so that
        # downstream caches request aligned blocks.
        block_shape, _ = self._get_canonical_block_config()
        self.Superpixels.meta.ideal_blockshape = block_shape + (1,)

        self.debug_results = None
        if self.EnableDebugOutputs.value:
            self.debug_results = OrderedDict()

    def execute(self, slot, subindex, roi, result):
        assert slot is self.Superpixels, "Unknown or unconnected output slot: {}".format(slot)

        if self.debug_results:
            self.debug_results.clear()

        # distance_transform_watershed expects a default value of None for pixel_pitch.
        if self.PixelPitch.value == []:
            pixel_pitch_to_pass = None
        else:
            pixel_pitch_to_pass = self.PixelPitch.value

        spatial_shape = self.Input.meta.shape[:-1]
        ndim = len(spatial_shape)

        if self.BlockwiseWatershed.value:
            self._execute_blockwise_deterministic(roi, result, pixel_pitch_to_pass, spatial_shape, ndim)
        else:
            # "compatibility" mode with older projects, where watershed was not
            # computed block-wise. Requests the entire ROI and runs single watershed.
            pmap = self._opSelectedInput.Output(roi.start, roi.stop).wait()
            ws, max_id = distance_transform_watershed(
                pmap[..., 0],
                self.Threshold.value,
                self.Sigma.value,
                self.Sigma.value,
                self.MinSize.value,
                self.Alpha.value,
                pixel_pitch_to_pass,
                self.ApplyNonmaxSuppression.value,
            )
            result[..., 0] = ws.astype("uint32")

        self.watershed_completed()

    def _execute_blockwise_deterministic(self, roi, result, pixel_pitch, spatial_shape, ndim):
        """Execute watershed with deterministic per-block ID offsets.

        For each canonical block overlapping the requested ROI:
        1. Request the block+halo from upstream
        2. Run distance_transform_watershed
        3. Re-label the inner region with vigra.analysis.labelMultiArray
        4. Add a deterministic offset: block_index * MAX_LABELS_PER_BLOCK
        5. Write the ROI-relevant portion into result
        """
        from lazyflow.roi import getIntersectingBlocks

        block_shape, halo = self._get_canonical_block_config()
        spatial_roi_start = np.array(roi.start[:-1])
        spatial_roi_stop = np.array(roi.stop[:-1])

        # Find which canonical blocks overlap the requested ROI
        block_starts = getIntersectingBlocks(block_shape, (spatial_roi_start, spatial_roi_stop))

        for block_start in block_starts:
            block_start = np.array(block_start)
            block_stop = np.minimum(block_start + np.array(block_shape), spatial_shape)

            # Compute the linear block index for deterministic offset
            blocks_per_dim = np.array([(s + b - 1) // b for s, b in zip(spatial_shape, block_shape)])
            block_index_nd = block_start // np.array(block_shape)
            block_index = int(np.ravel_multi_index(block_index_nd, blocks_per_dim))

            # Compute the outer (with halo) region, clipped to volume bounds
            outer_start = np.maximum(block_start - np.array(halo), 0)
            outer_stop = np.minimum(block_stop + np.array(halo), spatial_shape)

            # Request pmap for outer region (with channel dim)
            outer_start_with_c = np.append(outer_start, roi.start[-1])
            outer_stop_with_c = np.append(outer_stop, roi.stop[-1])
            pmap = self._opSelectedInput.Output(outer_start_with_c, outer_stop_with_c).wait()

            # Run watershed on outer region
            ws_outer, _ = distance_transform_watershed(
                pmap[..., 0],
                self.Threshold.value,
                self.Sigma.value,
                self.Sigma.value,
                self.MinSize.value,
                self.Alpha.value,
                pixel_pitch,
                self.ApplyNonmaxSuppression.value,
            )

            # Re-label inner region for clean consecutive labels
            ws_outer = ws_outer.astype("uint32")
            inner_local_start = block_start - outer_start
            inner_local_stop = block_stop - outer_start
            inner_local_slicing = tuple(slice(s, e) for s, e in zip(inner_local_start, inner_local_stop))
            ws_inner = vigra.analysis.labelMultiArray(ws_outer[inner_local_slicing])

            # Add deterministic offset (block 0 starts at 1, block 1 at MAX+1, etc.)
            ws_inner[ws_inner > 0] += np.uint32(block_index * self.MAX_LABELS_PER_BLOCK)

            # Write into result: intersection of this block with the requested ROI
            result_start = np.maximum(block_start, spatial_roi_start)
            result_stop = np.minimum(block_stop, spatial_roi_stop)

            # Slicing into result array (relative to ROI start)
            result_slicing = tuple(slice(int(s - r), int(e - r))
                                   for s, e, r in zip(result_start, result_stop, spatial_roi_start))
            result_slicing += (slice(0, 1),)  # channel dim

            # Slicing into ws_inner (relative to block start)
            ws_slicing = tuple(slice(int(s - b), int(e - b))
                               for s, e, b in zip(result_start, result_stop, block_start))

            result[result_slicing] = ws_inner[ws_slicing][..., np.newaxis]

    def propagateDirty(self, slot, subindex, roi):
        if slot is not self.EnableDebugOutputs:
            self.Superpixels.setDirty()


class OpCachedWsdt(Operator):
    RawData = InputSlot(optional=True)  # Used by the GUI for display only
    FreezeCache = InputSlot(value=True)

    Input = InputSlot()  # Can be multi-channel (but you'll have to choose which channel you want to use)
    ChannelSelections = InputSlot(value=[0])

    Threshold = InputSlot(value=0.5)
    MinSize = InputSlot(value=100)
    Sigma = InputSlot(value=3.0)
    Alpha = InputSlot(value=0.9)
    PixelPitch = InputSlot(value=[])
    ApplyNonmaxSuppression = InputSlot(value=False)
    InvertPixelProbabilities = InputSlot(value=False)

    EnableDebugOutputs = InputSlot(value=False)

    BlockwiseWatershed = InputSlot(value=True)

    Superpixels = OutputSlot()

    SuperpixelCacheInput = InputSlot(optional=True)
    CleanBlocks = OutputSlot()

    SelectedInput = OutputSlot()

    # Thresholding is cheap and best done interactively,
    # so expose an uncached slot just for it
    ThresholdedInput = OutputSlot()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        my_slot_names = set([slot.name for slot in self.inputSlots + self.outputSlots])
        wsdt_slot_names = set([slot.name for slot in OpWsdt.inputSlots + OpWsdt.outputSlots])
        assert wsdt_slot_names.issubset(my_slot_names), (
            "OpCachedWsdt should have all of the slots that OpWsdt has (and maybe more)."
            "Did you add a slot to OpWsdt and forget to add it to OpCachedWsdt?"
        )

        self._opWsdt = OpWsdt(parent=self)
        self._opWsdt.Input.connect(self.Input)
        self._opWsdt.ChannelSelections.connect(self.ChannelSelections)
        self._opWsdt.Threshold.connect(self.Threshold)
        self._opWsdt.MinSize.connect(self.MinSize)
        self._opWsdt.Sigma.connect(self.Sigma)
        self._opWsdt.Alpha.connect(self.Alpha)
        self._opWsdt.PixelPitch.connect(self.PixelPitch)
        self._opWsdt.ApplyNonmaxSuppression.connect(self.ApplyNonmaxSuppression)
        self._opWsdt.InvertPixelProbabilities.connect(self.InvertPixelProbabilities)
        self._opWsdt.EnableDebugOutputs.connect(self.EnableDebugOutputs)
        self._opWsdt.BlockwiseWatershed.connect(self.BlockwiseWatershed)

        self._opCache = OpBlockedArrayCache(parent=self)
        self._opCache.fixAtCurrent.connect(self.FreezeCache)
        self._opCache.Input.connect(self._opWsdt.Superpixels)
        self.Superpixels.connect(self._opCache.Output)
        self.CleanBlocks.connect(self._opCache.CleanBlocks)

        self._opSelectedInput = OpSumChannels(parent=self)
        self._opSelectedInput.ChannelSelections.connect(self.ChannelSelections)
        self._opSelectedInput.Input.connect(self.Input)
        self._opSelectedInput.InvertPixelProbabilities.connect(self.InvertPixelProbabilities)

        # rename output channel name
        self._opMetaInjector = OpMetadataInjector(parent=self)
        self._opMetaInjector.Input.connect(self._opSelectedInput.Output)
        self._opMetaInjector.Metadata.setValue({"channel_names": ["wsdt boundary channel"]})

        self.SelectedInput.connect(self._opMetaInjector.Output)

        self._opThreshold = OpPixelOperator(parent=self)
        # Attribute to track changes to thresholding function -> reduce unnecessary dirtiness
        # Any change to this operator triggers setting of the thresholding function
        # The lambda cannot be compared in a meaningful way in setValue -> threshold gets always dirty
        self._opThreshold._last_threshold = None
        self._opThreshold.Input.connect(self._opSelectedInput.Output)
        self.ThresholdedInput.connect(self._opThreshold.Output)

    def setupOutputs(self):
        # Set the cache block shape to match the watershed canonical blocking.
        # This ensures cache requests are always aligned with watershed blocks.
        if self._opWsdt.Superpixels.ready():
            ideal = self._opWsdt.Superpixels.meta.ideal_blockshape
            if ideal is not None:
                self._opCache.BlockShape.setValue(ideal)

        threshold = self.Threshold.value
        if threshold != self._opThreshold._last_threshold:
            self._opThreshold._last_threshold = threshold
            self._opThreshold.Function.setValue(lambda a: (a >= self.Threshold.value).astype(np.uint8))

    @property
    def debug_results(self):
        return self._opWsdt.debug_results

    @property
    def watershed_completed(self):
        return self._opWsdt.watershed_completed

    def execute(self, slot, subindex, roi, result):
        assert False, "Shouldn't get here"

    def _setInSlot(self, slot, subindex, roi, value):
        # Write the data into the cache
        assert slot is self.SuperpixelCacheInput
        slicing = roiToSlice(roi.start, roi.stop)
        self._opCache.Input[slicing] = value

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.EnableDebugOutputs and self.EnableDebugOutputs.value:
            # Force a refresh so the debug outputs will be updated.
            self._opCache.Input.setDirty()


class OpSumChannels(Operator):
    Input = InputSlot()
    ChannelSelections = InputSlot(value=[0])
    InvertPixelProbabilities = InputSlot(value=False)
    Output = OutputSlot()

    def setupOutputs(self):
        if len(self.ChannelSelections.value) == 0:
            self.Output.meta.NOTREADY = True
            return

        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.shape = self.Input.meta.shape[:-1] + (1,)

    def execute(self, slot, subindex, roi, result):
        channel_indexes = np.array(self.ChannelSelections.value)
        channel_indexes.sort()

        input_roi = roi.copy()
        input_roi.start[-1] = channel_indexes[0]
        input_roi.stop[-1] = channel_indexes[-1] + 1

        if len(channel_indexes) == 1:
            # Fetch in-place
            self.Input(input_roi.start, input_roi.stop).writeInto(result).wait()

        else:
            fetched_data = self.Input(input_roi.start, input_roi.stop).wait()
            channel_indexes = channel_indexes - channel_indexes[0]
            selected_data = fetched_data[..., tuple(channel_indexes)]
            result[..., 0] = selected_data.sum(axis=-1)

        if self.InvertPixelProbabilities.value:
            result[..., 0] = 1 - result[..., 0]

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.ChannelSelections or slot == self.InvertPixelProbabilities:
            # Everything is dirty
            self.Output.setDirty()
        elif slot == self.Input:
            # If any of the channels we care about became dirty, our output is dirty.
            channel_indexes = np.array(self.ChannelSelections.value).sort()
            first_channel = channel_indexes[0]
            last_channel = channel_indexes[-1]
            if roi.start[-1] > last_channel or roi.stop[-1] <= first_channel:
                return
            self.Output.setDirty(roi.start, roi.stop)
        else:
            assert False, "Unhandled input slot: {}".format(slot.name)
