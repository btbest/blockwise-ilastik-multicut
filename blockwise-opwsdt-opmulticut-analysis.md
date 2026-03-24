# Can OpWsdt and OpMulticut Compute Arbitrary ROIs Without Loading the Full Dataset?

## Context

The goal is to understand whether ilastik's `OpWsdt` and `OpMulticut` operators can be modified to serve arbitrary ROI requests without materializing the entire dataset in memory. This matters for large volumes where loading everything is infeasible. The blimp pipeline (`multicut_from_ilp.py`) already demonstrates a working blockwise approach outside lazyflow; the question is whether this can be brought into the lazyflow operator graph.

## Short Answer

**Yes, with one architectural caveat.** The pipeline has three stages with different locality properties:

| Stage | Locality | Can be blockwise? |
|-------|----------|-------------------|
| 1. Watershed (superpixels) | Fully local | Yes - each block independent |
| 2. RAG + features + multicut solve | Global barrier | RAG/features: blockwise accumulation. Solve: must be global, but operates on graph (not pixels) |
| 3. Pixel projection | Fully local | Already works blockwise |

The multicut solve is inherently global (it's a graph optimization over all edges), but its memory footprint scales with the number of superpixels/edges, not voxels. This is orders of magnitude smaller and generally manageable. Everything else can be made truly blockwise.

---

## Stage 1: OpWsdt - Blockwise Watershed

### Current Problem

`OpWsdt.execute()` (opWsdt.py:186-234) requests exactly the ROI it receives from upstream and runs watershed on that chunk. When `OpBlockedArrayCache` requests a single cache block, OpWsdt gets just that block — but superpixel IDs aren't globally unique across independently-computed cache blocks, and `parallel_watershed`'s internal blocking doesn't align with the cache blocking.

### Solution: Deterministic Block-Local Watershed

This mirrors blimp's `_ilastik_parallel_watershed` (multicut_from_ilp.py:474-584).

**Key idea:** Define a canonical blocking of the full volume (e.g. 128^3 with 10-voxel halo). When `execute()` is called with any ROI:

1. Determine which canonical blocks overlap the ROI
2. For each block: request block+halo from upstream, run `distance_transform_watershed`, run `vigra.analysis.labelMultiArray` on the inner region
3. Add a **deterministic offset**: `labels += block_id * MAX_LABELS_PER_BLOCK` (e.g. `MAX_LABELS_PER_BLOCK = 2^21`). This guarantees globally unique IDs without any cross-block coordination
4. Assemble results for the requested ROI

**Alignment trick:** Set `self.Superpixels.meta.ideal_blockshape` in `setupOutputs()` to match the canonical watershed block shape, and/or set `OpCachedWsdt._opCache.BlockShape` explicitly. This ensures the cache requests are always aligned with watershed blocks, so `execute()` always receives exactly one canonical block — the simplest case.

**Overflow check:** With uint32 and `MAX_LABELS_PER_BLOCK = 2^21`, this supports up to 2048 blocks (~26K^3 voxels). For larger volumes, use uint64 IDs.

**The `BlockwiseWatershed=False` legacy path** cannot be made ROI-aware without fundamental changes; it would continue to load everything (or could be deprecated for large volumes).

---

## Stage 2: RAG + Features + Multicut Solve

### 2a. Blockwise RAG Construction

**Current:** `OpCreateRag.execute()` (opEdgeTraining.py) does `superpixels = self.Superpixels[:].wait()` — loads the entire superpixel volume.

**Blockwise alternative** (proven in blimp, multicut_from_ilp.py:1032-1101):
1. Iterate blocks with halo (to capture cross-block edges)
2. For each block, request `Superpixels[outer_bb]`, build local `ilastikrag.Rag`
3. Collect edge_ids, canonicalize (sort endpoints), deduplicate across blocks
4. Produce a lightweight `BlockwiseRag` object holding just `edge_ids`, `num_edges`, `max_sp` — no pixel data

This is a drop-in replacement because `OpMulticutAgglomerator` only accesses `rag.edge_ids`, `rag.num_edges`, and `rag.max_sp` (opMulticut.py:159,200-201).

### 2b. Blockwise Edge Feature Computation

Same pattern: iterate blocks with halo, compute `ilastikrag` features per block, deduplicate/merge across blocks. Already proven in blimp (multicut_from_ilp.py:1032-1061).

### 2c. Global Multicut Solve — Stays Global

`OpMulticutAgglomerator.agglomerate_with_multicut()` (opMulticut.py:183-208) builds a nifty graph from `rag.edge_ids` and solves. This is inherently global but operates on the **graph** (proportional to number of superpixels and edges), not on voxel data. For a 256^3-blocked volume with 128^3 watershed blocks, even a 10K^3 volume has ~600K blocks with perhaps ~50M superpixels and ~150M edges — large but feasible in memory (a few GB for the graph).

**Optional enhancement:** Use elf's `blockwise_mc_impl` (hierarchical domain decomposition) for approximate blockwise multicut on extremely large graphs. This is already used in blimp (multicut_from_ilp.py:1121-1127). It's an approximation, not exact, so it could be offered as an alternative solver.

### 2d. Pixel Projection — Already Blockwise

`OpProjectNodeLabeling.execute()` (opMulticut.py:92-95) already works per-ROI:
```python
mapping_index_array = self.NodeLabels.value  # small 1D array, cached
self.Superpixels(roi.start, roi.stop).writeInto(result).wait()
result[:] = mapping_index_array[result]
```
No changes needed.

---

## Summary: What Would Need to Change

| File | Change |
|------|--------|
| `opWsdt.py` | Rewrite `execute()` for deterministic block-local watershed with global ID offsets; set `ideal_blockshape`/cache block shape in `setupOutputs()`/`OpCachedWsdt` |
| `opEdgeTraining.py` | Add `BlockwiseRag` class; replace `OpCreateRag` and `OpComputeEdgeFeatures` with blockwise versions that iterate blocks with halo and deduplicate |
| `opMulticut.py` | Verify `OpMulticutAgglomerator` works with `BlockwiseRag` (it should — only uses `.edge_ids`, `.num_edges`, `.max_sp`). No changes to projection. |

### Critical files
- `opWsdt.py`: `/home/user/blockwise-ilastik-multicut/libs/ilastik@9cc2254e/ilastik/applets/wsdt/opWsdt.py`
- `opMulticut.py`: `/home/user/blockwise-ilastik-multicut/libs/ilastik@9cc2254e/ilastik/applets/multicut/opMulticut.py`
- `opEdgeTraining.py`: `/home/user/blockwise-ilastik-multicut/libs/ilastik@9cc2254e/ilastik/applets/edgeTraining/opEdgeTraining.py`
- `multicut_from_ilp.py` (reference): `/home/user/blockwise-ilastik-multicut/multicut_from_ilp.py`

### The fundamental insight

The only truly global operation is the multicut graph solve, but it operates on a graph that is orders of magnitude smaller than the voxel data. Everything pixel-level (watershed, RAG construction, feature computation, label projection) can be made blockwise, following patterns already proven in blimp. The combination of deterministic superpixel ID offsets (for watershed) and blockwise edge accumulation with deduplication (for RAG/features) makes this work.
