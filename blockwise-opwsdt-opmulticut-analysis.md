# Can OpWsdt and OpMulticut Compute Arbitrary ROIs Without Loading the Full Dataset?

## Context

The goal is to understand whether ilastik's `OpWsdt` and `OpMulticut` operators can be modified to serve arbitrary ROI requests without materializing the entire dataset in memory. This matters for large volumes where loading everything is infeasible. The blimp pipeline (`multicut_from_ilp.py`) already demonstrates a working blockwise approach outside lazyflow; the question is whether this can be brought into the lazyflow operator graph.

There are **two distinct use cases** with very different requirements:

| Use case | What's needed | Can compute full watershed? |
|----------|---------------|----------------------------|
| **Export** (batch processing) | Full segmentation | Yes — we expect to process everything |
| **Interactive training** (browsing) | Edge colors for visible ROIs only | No — dataset may be terabytes |

The previous version of this analysis only addressed export. This revision focuses on the harder problem: **interactive edge-classifier training on arbitrarily large datasets**.

## Short Answer

**Yes, edges and their classifier-predicted costs can be computed for arbitrary ROIs** without a full-volume watershed or global RAG. The key insight is that during interactive training, the GUI only needs:

1. Superpixels for the visible slices (blockwise watershed)
2. Edges between those superpixels (local adjacency — computed per-slice by volumina's `SegmentationEdgesLayer`)
3. Features for those edges (local computation from superpixels + voxel data in a neighborhood)
4. Classifier predictions for those features (apply trained RF to local feature vectors)

None of these require global knowledge. The multicut segmentation preview must be dropped in interactive mode (it's a global optimization), but the classifier output visualization — edge colors showing predicted split/merge probability — works fully locally.

---

## How the Interactive GUI Currently Works

Tracing the data flow from `opEdgeTraining.py` through `edgeTrainingGui.py`:

### What the GUI displays

The GUI shows three edge-based layers (edgeTrainingGui.py:441-501):

1. **"Edge Probabilities"** — `SegmentationEdgesLayer(createDataSource(op.Superpixels))` with `pen_table` populated from `op.EdgeProbabilitiesDict.value`. Colors edges green (keep) to red (cut) based on classifier output.

2. **"Edge Labels"** — `LabelableSegmentationEdgesLayer(createDataSource(op.Superpixels))` with user-drawn labels. Interactive: user clicks edges to label them as "keep" or "cut".

3. **"Superpixel Edges"** — `SegmentationEdgesLayer(createDataSource(op.Superpixels))` with default yellow pen. Shows all superpixel boundaries.

### How edge rendering works

`SegmentationEdgesLayer` (volumina) takes a `DataSource` backed by `op.Superpixels`. For each visible 2D slice, it:
1. Requests the superpixel data for that slice from the DataSource (ROI-local)
2. Finds edges (boundaries between adjacent superpixel IDs) in the 2D slice
3. Looks up each edge `(sp1, sp2)` in its `pen_table` dict to get the color
4. If the edge isn't in `pen_table`, uses the default pen

**Critical observation:** Edge discovery is already per-slice/per-ROI. The `pen_table` is just a lookup dict consulted lazily. If an edge has no entry, it gets the default color — it doesn't crash.

### The current bottleneck

The problem is how `pen_table` gets populated (edgeTrainingGui.py:371-374):

```python
edge_probs = op.EdgeProbabilitiesDict.value  # ← global: ALL edges in volume
new_pens = {}
for id_pair, probability in edge_probs.items():
    new_pens[id_pair] = self.probability_pen_table[int(probability * 100)]
superpixel_edge_layer.pen_table.overwrite(new_pens)
```

This requests `EdgeProbabilitiesDict`, which triggers the entire chain:
- `OpCreateRag` → loads full superpixel volume → builds global RAG
- `OpComputeEdgeFeatures` → loads full voxel data → computes features for ALL edges
- `OpTrainEdgeClassifier` → trains RF on labeled edges
- `OpPredictEdgeProbabilities` → predicts ALL edges
- `OpEdgeProbabilitiesDict` → converts to dict of ALL edges

**Every step except classifier training is global.** This is what makes the current pipeline infeasible for large volumes during interactive use.

---

## Proposed Architecture: ROI-Local Edge Prediction

### Design Principle

Replace the "compute everything globally, cache, display" pattern with "compute on-demand per visible ROI, cache incrementally, display."

### What changes

#### 1. Blockwise Watershed (OpWsdt) — same as before

Deterministic block-local watershed with global ID offsets (`block_id * MAX_LABELS_PER_BLOCK`). Each block is independent. This is already detailed in Stage 1 of the previous analysis and remains unchanged.

**This is the foundation** — without deterministic, globally-unique superpixel IDs, nothing else works.

#### 2. ROI-Local Edge Feature Computation — NEW

**New operator: `OpRoiEdgeFeatures`**

Given a superpixel ROI (e.g., the current visible slice + a small halo), this operator:

1. Requests superpixels for the ROI (from blockwise-cached OpWsdt — cheap, already cached)
2. Builds a **local RAG** from just those superpixels (adjacency detection is purely local)
3. Requests voxel data (boundary probability map) for the same ROI
4. Computes edge features using `ilastikrag.Rag(local_superpixels).compute_features(local_voxels, feature_names)`
5. Returns a DataFrame of `{sp1, sp2, feature1, feature2, ...}` for edges in this ROI

**Why this is correct:** All standard edge features (`standard_edge_mean`, `standard_sp_mean`, etc.) are computed from voxel values at the boundary between two superpixels and/or region statistics of the superpixels. These are local — they only need the voxels in the neighborhood of the edge. With a small halo (e.g. 1 watershed block), features for edges near the ROI boundary are also captured correctly.

**What about superpixels that extend beyond the ROI?** Region-level features (like `standard_sp_mean`) would be computed from a truncated superpixel region. In practice this is acceptable for interactive preview — the user is viewing this region anyway, and features are "close enough" for classifier prediction. For export (where correctness matters more), the full blockwise pipeline from the previous analysis applies.

#### 3. Classifier Training — Minimal Changes

`OpTrainEdgeClassifier` (opEdgeTraining.py:337-394) does:

```python
for lane_index, (labels_dict_slot, features_slot) in ...:
    labels_dict = labels_dict_slot.value  # {(sp1,sp2): label} — user's labels
    edge_features_df = features_slot.value  # Full feature DataFrame
    features_and_labels_df = pd.merge(edge_features_df, labels_df, on=["sp1", "sp2"])
```

It merges the full feature DataFrame with the (sparse) labels dict to get features for labeled edges. The classifier only trains on labeled edges — typically tens to hundreds.

**For ROI-local mode:** Instead of computing features for ALL edges and then filtering to labeled ones, compute features only for labeled edges. Since the user labeled these edges interactively, they were visible at some point, meaning we already computed their features in step 2. We just need to **cache edge features as they're computed** and supply the cached subset to the trainer.

**Implementation:** Maintain a persistent `edge_features_cache: dict[(sp1,sp2)] -> feature_vector` that accumulates features as ROIs are viewed. When the classifier trains, it looks up features for labeled edges in this cache (they're guaranteed to be there, since the user could only label visible edges).

#### 4. ROI-Local Edge Prediction — NEW

**New operator: `OpRoiEdgePrediction`**

Given edge features (from step 2) and a trained classifier (from step 3):

1. Takes the local edge features DataFrame
2. Applies `classifier.predict_probabilities(feature_matrix)[:, 1]`
3. Returns `dict[(sp1,sp2)] -> probability` for edges in this ROI

This is trivially local — it's just a random forest prediction on a feature matrix.

#### 5. GUI Integration — Incremental pen_table Updates

Instead of the current pattern (overwrite entire pen_table when EdgeProbabilitiesDict changes), update the pen_table incrementally:

```python
# When a new ROI is viewed (or classifier is retrained):
local_probs = compute_edge_probs_for_roi(visible_roi)
new_pens = {}
for (sp1, sp2), prob in local_probs.items():
    new_pens[(sp1, sp2)] = self.probability_pen_table[int(prob * 100)]
superpixel_edge_layer.pen_table.update(new_pens)  # merge, not overwrite
```

Edges outside the current view retain their previously-computed colors (or default if never viewed). This is fine — the user can't see those edges anyway.

#### 6. Multicut Preview — Removed in Interactive Mode

The multicut solve (`OpMulticutAgglomerator`) requires a global RAG and global edge probabilities. This cannot be made ROI-local because multicut is a global optimization — cutting one edge affects the optimal solution for distant edges.

**For interactive mode:** Remove/disable the multicut segmentation preview. The user sees:
- Superpixel edges (always available, blockwise)
- Edge labels (their own annotations)
- **Edge probability colors** (classifier predictions for visible edges — the key interactive feedback)

This is sufficient for training the classifier. The user doesn't need to see the multicut result until export.

**Could we solve multicut on a local subgraph?** Technically yes — extract the subgraph for visible superpixels and solve. elf's `blockwise_mc_impl` does something similar (hierarchical block decomposition). But:
- The result would be a local approximation, potentially misleading
- It adds complexity for dubious user value during training
- The classifier predictions (edge colors) already give the user the information they need

If a local multicut preview is desired later, it could be added as an optional feature using elf's subproblem solver.

---

## Detailed Pipeline Comparison

### Current (global) pipeline during interactive training:

```
User views slice
  → volumina requests Superpixels[slice]
    → OpCachedWsdt requests full volume (!)
      → OpWsdt runs watershed on full volume (!)
  → GUI requests EdgeProbabilitiesDict.value
    → OpCreateRag loads full superpixels (!)
      → builds global RAG (!)
    → OpComputeEdgeFeatures loads full voxels (!)
      → computes features for ALL edges (!)
    → OpTrainEdgeClassifier (OK — only uses labeled edges)
    → OpPredictEdgeProbabilities predicts ALL edges (!)
    → OpEdgeProbabilitiesDict converts ALL to dict (!)
  → pen_table overwritten with ALL edge colors
  → SegmentationEdgesLayer renders visible edges with colors
```

### Proposed (ROI-local) pipeline:

```
User views slice
  → volumina requests Superpixels[slice]
    → OpCachedWsdt requests one watershed block (local!)
      → OpWsdt runs watershed on one block (local!)
  → trigger: compute predictions for visible ROI
    → build local RAG from Superpixels[visible_roi] (local!)
    → compute features for local edges (local!)
    → cache features for labeled edges
    → apply classifier to local features (local!)
    → update pen_table with local edge colors (incremental!)
  → SegmentationEdgesLayer renders visible edges with colors
```

**Memory comparison for a 10K^3 uint8 volume:**
- Current: ~1TB superpixels (uint32) + ~1TB voxels loaded
- Proposed: ~8MB superpixels per 128^3 block + ~2MB voxels per block

---

## Implementation Steps

### Phase 1: Blockwise OpWsdt (prerequisite)

As described in the "Stage 1" section — deterministic block-local watershed with global ID offsets. This enables everything else.

Files: `opWsdt.py`

### Phase 2: ROI-Local Edge Features + Prediction

1. **New operator `OpRoiEdgeFeatures`**: Given superpixels and voxel data for an ROI, builds local RAG, computes features, returns DataFrame.

2. **New operator `OpRoiEdgePrediction`**: Given local features + classifier, returns `dict[(sp1,sp2) -> probability]`.

3. **Modified `OpTrainEdgeClassifier`**: Instead of requiring global `EdgeFeaturesDataFrame`, accepts an incrementally-growing cache of per-edge features. Trains on the subset matching labeled edges.

4. **Modified GUI (`edgeTrainingGui.py`)**: On ROI change, trigger local feature computation + prediction. Update pen_table incrementally. Disable multicut preview layer.

Files: `opEdgeTraining.py`, `edgeTrainingGui.py`

### Phase 3: Export Pipeline (uses full blockwise approach)

For export/batch processing, use the full blockwise pipeline from the previous analysis:
- Blockwise RAG construction (iterate all blocks, deduplicate edges)
- Blockwise feature computation (iterate all blocks, merge features)
- Global multicut solve (on the graph, not voxels)
- Blockwise pixel projection (already works)

This can coexist with the interactive pipeline — they share the blockwise OpWsdt but diverge after that.

Files: `opEdgeTraining.py`, `opMulticut.py`

---

## Key Files

| File | Role |
|------|------|
| `opWsdt.py` | Blockwise watershed with deterministic IDs (Phase 1) |
| `opEdgeTraining.py` | ROI-local edge features/prediction operators; incremental feature cache (Phase 2) |
| `edgeTrainingGui.py` | Incremental pen_table updates; disable multicut preview (Phase 2) |
| `opMulticut.py` | Unchanged for interactive; blockwise RAG/features for export (Phase 3) |
| `multicut_from_ilp.py` | Reference implementation for all blockwise patterns |

## Summary

The fundamental insight is that **edge feature computation and classifier prediction are local operations**. An edge's features depend only on the voxels near the boundary between two superpixels. A classifier prediction is just a function evaluation on a feature vector. Neither requires global context.

The only global operation is the multicut solve, but during interactive training the user doesn't need it — classifier-predicted edge colors (green=keep, red=cut) provide the feedback needed to train the classifier effectively. The multicut solve is only needed at export time, when the full blockwise pipeline processes the entire dataset anyway.

With blockwise watershed (deterministic IDs) providing the foundation, ROI-local edge features + prediction can be layered on top to give a fully lazy, on-demand interactive experience even for terabyte-scale datasets.
