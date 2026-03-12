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
    │   └── EdgeLabels0000/
    │       ├── sp_ids          # uint32 array (N, 2): superpixel id pairs
    │       └── labels          # uint8 array (N,):   1=merge  2=split
    ├── EdgeFeatures/
    │   └── 0000/               # pandas DataFrame stored as HDF5:
    │       ├── columns         #   sp1, sp2, <feature_1>, <feature_2>, ...
    │       └── ...
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
.ilp
  ├── EdgeFeatures/0000  →  X  (N_edges × N_features float32 matrix)
  └── EdgeLabelsDict/0000  →  y  (merge=1 / split=2, only for annotated edges)

        join on (sp1, sp2)
              ↓
  sklearn.RandomForestClassifier.fit(X_labeled, y_labeled)
              ↓
  save sklearn RF as rf.pkl
```

The resulting `rf.pkl` uses elf's expected sklearn interface:
`rf.predict_proba(features)[:, 1]` → boundary probability per edge.

### Inference step (blockwise, runs on new data)

```
Raw data + boundary probability maps  (zarr / HDF5 / TIFF, any size)
              ↓
  [for each block with halo]
    elf watershed  →  superpixels (block)
    ilastikrag.Rag(superpixels)
    rag.compute_features(channel_data, feature_names)   ← same names as training
    rf.predict_proba(features)[:, 1]                    ← sklearn RF
    elf.segmentation.multicut.compute_edge_costs(probs)
    accumulate edge costs into global graph
              ↓
  elf.segmentation.multicut.blockwise_multicut(graph, costs, watershed)
              ↓
  segmentation (zarr / HDF5 output)
```

Memory is bounded per block. Only the global graph (superpixel adjacency + one
float per edge) must be held in memory simultaneously, which is small compared
to the raw voxel data.

---

## Channel mapping

The `FeatureNames` dict in the `.ilp` uses ilastik's internal channel names,
e.g. `"Membrane Probabilities 0"` or `"Raw Data 0"`. At inference time the CLI
needs to know which file / dataset corresponds to each channel name. This is
provided via a `--channels` argument:

```
--channels "Membrane Probabilities 0:/path/to/boundary.h5:/data"
           "Raw Data 0:/path/to/raw.h5:/data"
```

The notebook version shows how to inspect the stored channel names first with
`read_feature_names(ilp_path)`.

---

## Files

| File | Purpose |
|------|---------|
| `ilp_reader.py` | Read EdgeFeatures, EdgeLabelsDict, FeatureNames from `.ilp` (h5py only, no ilastik import) |
| `fit_classifier.py` | Re-fit a sklearn RF from `.ilp` training data; save as pickle |
| `multicut_from_ilp.py` | CLI: blockwise inference using the fitted RF and elf multicut |
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

### 1. Inspect the `.ilp` file

```python
from ilp_reader import read_feature_names, read_training_data

# See what channels and features were used during training
feature_names = read_feature_names("my_project.ilp")
# → {"Membrane Probabilities 0": ["standard_edge_mean", ...], "Raw Data 0": [...]}

# Load the (features, labels) training set
X, y, columns = read_training_data("my_project.ilp")
print(f"{len(y)} annotated edges, {X.shape[1]} features")
```

### 2. Re-fit the sklearn classifier

```bash
python fit_classifier.py \
    --ilp my_project.ilp \
    --output rf.pkl \
    --n-estimators 200 \
    --n-jobs 8
```

### 3. Run blockwise multicut on a large volume

```bash
python multicut_from_ilp.py \
    --ilp my_project.ilp \
    --rf rf.pkl \
    --channels "Membrane Probabilities 0:boundary.h5:/data" \
               "Raw Data 0:raw.h5:/data" \
    --output segmentation.h5 \
    --output-key /seg \
    --block-shape 256 256 256 \
    --halo 32 32 32 \
    --beta 0.5 \
    --n-threads 8
```

---

## Memory usage

- **Re-fit step**: negligible — reads a DataFrame from HDF5.
- **Inference per block** (256³ with 32-voxel halo): a float32 raw block is
  ~256 MB; superpixel labels add ~64 MB; features and RAG are small. Peak
  usage per block is dominated by the input data, typically 0.5–1 GB.
- **Global graph**: one float per edge. For a 1000³ volume with typical
  superpixel densities (~1000 voxels/superpixel) there are O(10⁷) edges
  → ~40 MB in float32. Elf's blockwise multicut then decomposes the
  optimization problem without loading everything at once.

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
