# Agent setup

## Environment

Create and activate the environment (from repo root):

```bash
micromamba create -f environment.yml -n blimp -y
/path/to/micromamba/bin/python -m pip install -e .
```

Verify installation:

```bash
blimp -h
```

**Important:** Files in `libs/` are external dependencies and must never be modified.


## Tests

```bash
pytest tests/
```

## Demo (end-to-end on synthetic data)

```bash
python run_demo.py          # writes output to /tmp/blockwise_mc_demo/
```

---

# blimp — Developer Documentation

This section covers the technical architecture, design decisions, and implementation details for developers working on blimp.

## Motivation

ilastik's interactive training UX makes it easy to annotate superpixel edges as "merge" or "split" and get a well-tuned random forest classifier in minutes. However, ilastik's built-in multicut solver cannot handle large-than-memory volumes. elf provides an efficient **blockwise** multicut solver (hierarchical graph decomposition) that scales to large volumes.

blimp bridges these two:

1. **Reads** edge training data from ilastik `.ilp` project files
2. **Re-fits** a `sklearn.RandomForestClassifier` using cached ilastikrag feature vectors (no re-computation needed)
3. **Processes** volumes blockwise at inference time: each block computes watershed and features independently (bounded RAM), predicts edges, then elf's solver assembles the final segmentation

## Architecture

### Training step (runs once on the `.ilp` file)

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

### Inference step — blockwise (for full large volumes)

```
Large volume (zarr / HDF5, any size — never fully loaded)
              ↓
  blockwise watershed (boundary_lazy, output=ws_memmap_on_disk)
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

**Memory peak:** One block of input data + global edge dict (~10–15 GB total for a typical 20 GB volume).

## ILP file structure (HDF5 reference)

An ilastik `.ilp` file is an HDF5 file. Relevant groups under `Training and Multicut/`:

```
<project>.ilp  (HDF5)
└── Training and Multicut/
    ├── FeatureNames/           # dict: {channel_name → [feature names]}
    ├── EdgeLabelsDict/
    │   ├── EdgeLabels0000/     # one group per training crop (lane)
    │   ├── EdgeLabels0001/
    │   └── EdgeLabels0002/
    │       ├── sp_ids          # uint32 array (N, 2): superpixel id pairs
    │       └── labels          # uint8 array (N,):   1=merge  2=split
    ├── EdgeFeatures/
    │   ├── 0000/               # one group per lane (pandas DataFrame as HDF5)
    │   ├── 0001/
    │   └── 0002/
    ├── Rags/                   # cached RAG (superpixel adjacency)
    │   └── Rag_0000/
    └── Output/                 # trained vigra random forest (not used)
        ├── Forest0000/
        ├── Forest0001/
        ├── known_labels
        ├── feature_names
        └── pickled_type
```

**Key insight:** `EdgeFeatures` + `EdgeLabelsDict` = complete labeled training set with features already in ilastikrag space. No re-computation needed.

## Code structure

| File | Purpose |
|------|---------|
| `blimp.py` | Main entrypoint: full pipeline in one command |
| `blimp_watershed.py` | Power-user entrypoint: watershed only (no multicut) |
| `_cli_params.py` | Shared CLI parameter definitions (watershed, blockwise) |
| `_cli_helpers.py` | Shared validation and parameter resolution |
| `ilp_reader.py` | Read training data, features, names from `.ilp` |
| `fit_classifier.py` | Re-fit sklearn RF from training crops |
| `multicut_from_ilp.py` | Lower-level blockwise inference with pre-fitted RF |

## Advanced usage: run steps separately

### Step 1 — re-fit the sklearn classifier

```bash
python fit_classifier.py \
    --ilp my_project.ilp \
    --output rf.pkl \
    --n-estimators 100 \
    --n-jobs 8
```

### Step 2 — blockwise multicut (large volumes)

```bash
python multicut_from_ilp.py \
    --ilp my_project.ilp \
    --rf rf.pkl \
    --channels "wsdt boundary channel:boundary.zarr" \
               "Raw Data:raw.zarr" \
    --lazy \
    --ws-tmp /scratch/ws_tmp.dat \
    --output-zarr segmentation.zarr \
    --block-shape 256 256 256 --halo 32 32 32 \
    --beta 0.5 --n-threads 8
```

Map channel names (from `read_feature_names`) to files. Disk space needed: `volume_shape × 8 bytes` for watershed.

## Dependencies

Key packages (see `environment.yml` for full list):

```
conda install -c ilastik-forge ilastikrag vigra
conda install -c conda-forge scikit-learn h5py zarr nifty
pip install elf
```

`vigra` is only needed at inference time (RAG construction). Not needed for re-fitting or reading training data.

## Memory usage

- **Re-fit:** negligible (reads DataFrames from HDF5)
- **Per block (256³ + 32-voxel halo):**
  - Input data: ~0.5–1 GB (float32, 2 channels)
  - Watershed: zarr on disk (uint64, ≈ 8× voxel bytes); never fully in RAM
  - ilastikrag.Rag: block's superpixels only ✓
- **Global edge dict:** All edges in RAM. For 20 GB uint8 volume: ~500 MB
- **blockwise_multicut:** Reads watershed memmap block-by-block ✓
- **Estimated peak RAM (20 GB volume):** ~10–15 GB

## Limitations and future work

- **Out-of-core graph assembly:** Currently holds global superpixel graph in memory. Very large volumes (>10⁹ voxels) would need disk-backed sparse representation (e.g., zarr-backed nifty graph).
