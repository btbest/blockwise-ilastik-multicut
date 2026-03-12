# blockwise-ilastik-multicut

A CLI tool (and Jupyter notebook) to run edge classifiers trained in ilastik's
"Boundary-Based Segmentation with Multicut" workflow on large volumes using
elf's blockwise multicut solver.

`libs/` contains reference snapshots of
[ilastik](https://github.com/ilastik/ilastik) and
[elf](https://github.com/constantinpape/elf).

---

## Motivation

ilastik's interactive training UX makes it easy to annotate a handful of
superpixel edges as "merge" or "split" and get a well-tuned random forest
classifier in minutes. However, ilastik's built-in multicut solver is not
designed for large-than-memory volumes. elf provides an efficient **blockwise**
multicut solver (hierarchical graph decomposition) that scales to large volumes.

This project bridges the two:

1. Reads the edge training data stored inside an ilastik `.ilp` project file.
2. Re-fits a `sklearn.RandomForestClassifier` on those training pairs — using
   the **exact same ilastikrag feature vectors** that ilastik computed, which
   are cached inside the `.ilp` file alongside the training labels. No
   re-computation of features is needed for the training step, and no fidelity
   is lost compared to the original ilastik classifier.
3. At inference time, processes the volume **blockwise**: for each block the
   watershed and ilastikrag features are computed independently (bounded memory
   per block), the sklearn RF predicts edge probabilities, and elf's blockwise
   multicut solver assembles the final segmentation.

---

## ILP file structure (HDF5 reference)

An ilastik `.ilp` file is an HDF5 file. The relevant groups for this project
are all under `Training and Multicut/` (the `projectFileGroupName`):

```
<project>.ilp  (HDF5)
└── Training and Multicut/
    ├── FeatureNames/           # dict: {channel_name → [ilastikrag feature names]}
    ├── EdgeLabelsDict/
    │   ├── EdgeLabels0000/     # one group per training crop (lane)
    │   ├── EdgeLabels0001/
    │   └── EdgeLabels0002/
    │       ├── sp_ids          # uint32 array (N, 2): superpixel id pairs
    │       └── labels          # uint8 array (N,):   1=merge  2=split
    ├── EdgeFeatures/
    │   ├── 0000/               # one group per lane; pandas DataFrame as HDF5:
    │   ├── 0001/               #   sp1, sp2, <feature_1>, <feature_2>, ...
    │   └── 0002/
    ├── Rags/                   # cached ilastikrag RAG (superpixel adjacency)
    │   └── Rag_0000/
    └── Output/                 # trained vigra random forest (not used directly)
        ├── Forest0000/
        ├── Forest0001/
        ├── known_labels
        ├── feature_names
        └── pickled_type
```

Key insight: `EdgeFeatures` and `EdgeLabelsDict` together form a complete
labeled training set. The feature vectors are already in ilastikrag's feature
space, so no feature re-computation is needed for the re-fit step.

---

## Architecture

### Training step (offline, runs on the `.ilp` file once)

```
.ilp  (trained on N crops / lanes)
  ├── EdgeFeatures/0000 … EdgeFeatures/000N  →  feature matrices per crop
  └── EdgeLabelsDict/EdgeLabels0000 … 000N   →  merge/split labels per crop

        discover_lanes() → [0, 1, 2, …]
        concat across all lanes
              ↓
  sklearn.RandomForestClassifier.fit(X_all_lanes, y_all_lanes)
              ↓
  save sklearn RF as rf.pkl
```

The resulting `rf.pkl` uses elf's expected sklearn interface:
`rf.predict_proba(features)[:, split_col]` → boundary probability per edge.

### Inference step — lazy/blockwise (for the full large volume)

```
Large volume (zarr / HDF5, any size — never fully loaded)
              ↓
  blockwise_two_pass_watershed(boundary_lazy, output=ws_memmap_on_disk)
              ↓
  [for each block with halo — sequential, bounded RAM]
    ws_block     = ws_memmap[outer_bb]        ← load one block from disk
    channel_data = {name: lazy[outer_bb]}     ← load one block per channel
    ilastikrag.Rag(ws_block)
    rag.compute_features(channel_data, feature_names)
    rf.predict_proba(features)[:, split_col]
    accumulate → global edge cost dict (in RAM, ~1–5 GB)
              ↓
  nifty.graph.undirectedGraph + insertEdges(edge_uvs)
              ↓
  blockwise_multicut(graph, costs, ws_memmap)  ← ws read block-by-block ✓
              ↓
  [for each block] node_labels[ws_memmap[bb]] → write zarr output
```

Memory peak: one block of input data + global edge dict (~10–15 GB total for a
typical 20 GB volume).

Memory is bounded per block. Only the global graph (superpixel adjacency + one
float per edge) must be held in memory simultaneously, which is small compared
to the raw voxel data.

---

## What you need before running inference

Three inputs are required:

1. **A trained `.ilp` project file** — created in ilastik's "Boundary-Based
   Segmentation with Multicut" workflow. This contains the training annotations
   and cached feature vectors used by `fit_classifier.py`.

2. **Membrane probability predictions** (and any other channels used during
   training) — these are **not** computed by this tool. Run ilastik's
   **Pixel Classification** workflow (or any other boundary detector) on your
   full volume first, then export the probability maps as HDF5 or zarr.

3. **The raw data volume** you want to segment — only needed if raw intensity
   was used as a channel during training (check with `read_feature_names`
   below).

There is **no separate `--input` flag**. All input volumes — raw data,
membrane predictions, and any other channels — are provided together through
`--channels` (see below).

---

## Channel mapping

During training, ilastik assigns each input channel an internal name such as
`"Membrane Probabilities 0"` or `"Raw Data 0"`. The set of channels and their
names for your project are stored in the `.ilp`'s `FeatureNames` group. You
can inspect them before running inference:

```python
from ilp_reader import read_feature_names

feature_names = read_feature_names("my_project.ilp")
# → {"Membrane Probabilities 0": ["standard_edge_mean", ...], "Raw Data 0": [...]}
```

The `--channels` argument maps each of those internal names to the
corresponding file on disk:

```
--channels "Membrane Probabilities 0:/path/to/boundary.h5:/data"
           "Raw Data 0:/path/to/raw.h5:/data"
```

- The name on the **left** of `:` must exactly match a key returned by
  `read_feature_names`.
- The **file path** (and optional HDF5 dataset key) on the right points to the
  corresponding volume for the new data you want to segment.
- You must supply **one entry per channel** that appears in `FeatureNames`.
  Some projects use only membrane probabilities; others also include raw data.

In other words, `--channels` is both the "here is the new volume to segment"
argument *and* the channel-name mapping — there is no separate input flag.

---

## Files

| File | Purpose |
|------|---------|
| `ilp_reader.py` | Read EdgeFeatures, EdgeLabelsDict, FeatureNames from `.ilp`; multi-lane aware |
| `fit_classifier.py` | Re-fit a sklearn RF from all training crops; save as pickle |
| `multicut_from_ilp.py` | CLI: in-memory or lazy blockwise inference using the fitted RF and elf multicut |
| `multicut_from_ilp.ipynb` | Notebook: same pipeline, step-by-step with inspection utilities |

---

## Requirements

```
conda install -c ilastik-forge ilastikrag vigra
conda install -c conda-forge scikit-learn h5py zarr nifty
pip install elf
```

`vigra` is only needed at inference time (for ilastikrag's RAG construction).
It is **not** needed to re-fit the classifier or to read the training data from
the `.ilp` file.

---

## Usage

### 0. Produce membrane probability predictions (prerequisite)

Before running this tool, use **ilastik's Pixel Classification** workflow (or
any other boundary detector) to generate membrane probability maps for the
volume you want to segment. Export the result as HDF5 or zarr. If raw
intensity was also used as a channel during training, keep that file handy too.

### 1. Inspect the `.ilp` file

Check which channels and ilastikrag features were used during training. The
channel names shown here are exactly what you will pass to `--channels` in
steps 3a/3b.

```python
from ilp_reader import discover_lanes, read_feature_names

# See how many training crops are in the project
print(discover_lanes("my_project.ilp"))
# → [0, 1, 2]  (trained on three 256³ crops)

# See which channels you must supply at inference time
feature_names = read_feature_names("my_project.ilp")
# → {"Membrane Probabilities 0": ["standard_edge_mean", ...], "Raw Data 0": [...]}
# You must provide one --channels entry for every key in this dict.
```

### 2. Re-fit the sklearn classifier (all crops automatically)

```bash
python fit_classifier.py \
    --ilp my_project.ilp \
    --output rf.pkl \
    --n-estimators 200 \
    --n-jobs 8
# lane defaults to None → reads and concatenates all three crops
```

### 3a. Run blockwise multicut — in-memory (volumes that fit in RAM)

Supply one `--channels` entry per channel from `read_feature_names`, mapping
each ilastik channel name to the corresponding file for your new volume:

```bash
python multicut_from_ilp.py \
    --ilp my_project.ilp \
    --rf rf.pkl \
    --channels "Membrane Probabilities 0:boundary.h5:/data" \
               "Raw Data 0:raw.h5:/data" \
    --output segmentation.h5 --output-key /seg \
    --block-shape 256 256 256 --halo 32 32 32 \
    --beta 0.5 --n-threads 8
```

### 3b. Run blockwise multicut — lazy mode (large volumes, e.g. 20 GB)

```bash
python multicut_from_ilp.py \
    --ilp my_project.ilp \
    --rf rf.pkl \
    --channels "Membrane Probabilities 0:boundary.zarr" \
               "Raw Data 0:raw.zarr" \
    --lazy \
    --ws-tmp /scratch/ws_tmp.dat \
    --output-zarr segmentation.zarr \
    --block-shape 256 256 256 --halo 32 32 32 \
    --beta 0.5 --n-threads 8
```

In lazy mode, disk space of `volume_shape × 8 bytes` is needed for the
watershed tempfile (`--ws-tmp`). This file is deleted automatically on
successful completion.

---

## Memory usage

- **Re-fit step**: negligible — reads DataFrames from HDF5 (one per crop).
- **Lazy inference per block** (256³ + 32-voxel halo):
  - Input data: ~0.5–1 GB per block (float32, 2 channels)
  - Watershed: stored on disk as a numpy memmap (uint64, ≈ 8× raw voxel count
    in bytes); never fully in RAM
  - ilastikrag.Rag: only the block's superpixels are needed in RAM ✓
- **Global edge dict**: all edge costs accumulated in a Python dict.
  For a 20 GB uint8 volume with ~1000 voxels/superpixel there are O(10⁷)
  edges → ~500 MB.
- **blockwise_multicut**: reads the watershed memmap one block at a time via
  `segmentation[bb]` (returns numpy from memmap) ✓
- **Estimated peak RAM for a 20 GB volume**: ~10–15 GB.

---

## Limitations and future work

- **Feature space at inference must match training.** The stored `FeatureNames`
  tells us exactly which ilastikrag features to compute. If the boundary
  channel layout (number of channels, channel order) changes between the
  `.ilp` training data and inference data, the channel mapping must be
  updated accordingly.
- **Watershed consistency.** The inference watershed is recomputed fresh for
  new data. Using the same watershed parameters as ilastik (DT Watershed
  applet settings) is recommended for best results; those parameters can be
  read from `DT Watershed/` in the `.ilp` file.
- **Out-of-core global graph assembly.** Currently the global superpixel graph
  is assembled in memory. For very large volumes (>10⁹ voxels) a disk-backed
  sparse representation (e.g. zarr-backed nifty graph) would be needed.
- **Multi-lane `.ilp` files.** The current implementation targets single-lane
  projects. Multi-lane support can be added by iterating over lanes in
  `EdgeLabels{NNNN}` / `EdgeFeatures/{NNNN}`.
