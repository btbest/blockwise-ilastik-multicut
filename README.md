# blimp — BLockwise Ilastik Multicut Pipeline

Run edge classifiers trained in ilastik on large 3D volumes.

**blimp** is a command-line tool that takes a trained ilastik project, raw data, and boundary predictions—then outputs a final segmentation. It's fast, memory-efficient, and works with volumes too large to fit in RAM.

---

## What you need

Before running blimp, gather three things:

1. **A trained `.ilp` project file** — created in ilastik's "Boundary-Based Segmentation with Multicut" workflow
2. **Boundary probability predictions** — HDF5 or zarr file with the same shape as your raw data (zyx axis order)
3. **Raw data volume** — HDF5 or zarr (zyx axis order)

> **Note:** Boundary predictions are not computed by blimp. Run ilastik's Pixel Classification workflow (or another boundary detector) first, then export the probability map.

---

## Installation

```bash
# Create the conda environment (installs dependencies)
conda env create -n blimp -f environment.yml
conda activate blimp

# Install blimp
pip install -e .
```

The `blimp` command will now be available on your PATH.

---

## Quick start

```bash
blimp \
    --ilp my_project.ilp \
    --raw raw.zarr \
    --probabilities boundary.zarr \
    --output-dir results/
```

**Output files in `results/`:**

| File | Contents |
|------|----------|
| `raw_segmentation.zarr` | Final segmentation (uint64, zyx) |
| `rf.pkl` | Fitted classifier |
| `params.json` | Call parameters for reproducibility |
| `raw_watershed.zarr` | Watershed superpixels |

### Input formats

Both `--raw` and `--probabilities` accept:

- **zarr:** `/path/to/file.zarr`
- **HDF5:** `/path/to/file.h5` (must contain exactly one dataset)
- **Windows paths:** `C:\Users\...\file.h5`

Volumes must be in **zyx(c) axis order** with the **same shape**. Singleton channels are OK.

---

## Common options

```bash
blimp --ilp project.ilp --raw raw.zarr --probabilities boundary.zarr --output-dir results/ \
    --max-block-shape 256 256 256      # block size (default)
    --halo 32 32 32                     # block overlap (default)
    --beta 0.5                          # merge/split bias (0.5 = balanced)
    --threads 8                         # parallel threads (default)
    --n-estimators 100                  # RF trees (default)
```

**Understanding beta:**
- `beta < 0.5`: favors merging segments
- `beta = 0.5`: balanced (default)
- `beta > 0.5`: favors splitting segments

For **anisotropic data** (e.g., 2D microscopy), use `--ws-method 2d`.

### All options

```
Required:
  --ilp PATH                  Ilastik .ilp project file
  --raw PATH                  Raw data volume
  --probabilities PATH        Boundary probability volume
  --output-dir DIR            Output directory

Blockwise / multicut:
  --max-block-shape Z Y X     Block size (default: 256 256 256)
  --halo Z Y X                Block overlap (default: 32 32 32)
  --beta FLOAT                Merge/split bias (default: 0.5)
  --threads INT               Parallel threads (default: 8)
  --solver                    kernighan-lin | greedy-additive | greedy-fixation
                              (default: kernighan-lin)

Classifier:
  --n-estimators INT          Random forest trees (default: 100)

Watershed:
  --ws-method                 ilastik | two-pass | 2d
  --ws-threshold FLOAT        Seed threshold (default: from .ilp or 0.5)
  --ws-sigma FLOAT            Gaussian smoothing (default: from .ilp or 3.0)
  --ws-min-size INT           Min superpixel size (default: from .ilp or 100)
  --ws-invert                 Flip probability map (interior→boundary)

Reuse watershed:
  --ws-zarr PATH              Use pre-computed watershed (skips ws step)
  --keep-watershed | --no-keep-watershed  Keep the watershed zarr (default: keep)
```

---

## Need help?

- **Why is my segmentation noisy?** Try adjusting `--beta` (values <0.5 merge more).
- **Is my data too large?** blimp processes blockwise. Peak RAM for a 20 GB volume is ~10–15 GB.
- **Can I reuse the watershed?** Yes—save it with `--keep-watershed` (default), then pass `--ws-zarr` to a new run.

---

## For developers

See [AGENTS.md](AGENTS.md) for:
- Architecture and design
- HDF5 `.ilp` file format reference
- Running individual pipeline steps
- Memory analysis
- Feature computation details

---

## License

See LICENSE file.
