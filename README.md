# blimp — BLockwise Ilastik Multicut Pipeline

Run edge classifiers trained in ilastik on large 3D volumes.

**blimp** is a command-line tool that takes a trained ilastik project, raw data, and boundary predictions—then outputs a final segmentation. The blockwise implementation allows it to handle volumes too large to fit in RAM.

Terabyte-scale stress testing to be done :)

---

## What you need

Before running blimp, gather three things:

1. **A trained `.ilp` project file** — created in ilastik's "Boundary-Based Segmentation with Multicut" workflow
2. **Boundary probability predictions** — HDF5 or zarr file with the same shape as your raw data (zyx axis order)
3. **Raw data volume** — HDF5 or zarr (zyx axis order)

How to get these:

### Raw data

Your electron microscope :)

blimp will need to be able to access the data in a *blockwise (chunkwise)* manner, which is not possible with .TIFF or .PNG files.
You need to pre-convert your dataset to HDF5 or Zarr, e.g.:
- By uploading to Webknossos, creating an empty Annotation, and then creating a Zarr share link for the Annotation (the path to your raw data for blimp will be `https://webknossos-share-link/1` - note the `/1`)
- Using the Data Conversion workflow in ilastik (convert to "compressed hdf5" or "single-scale OME-Zarr")
- Using another tool like `ngff-zarr`, `eubi-bridge`, ...

### Boundary probabilities

1. Extract a small number (5-15) of subvolumes from your raw data; 256 x 256 x 256 voxels each.
2. Train a classifier for segmenting membranes (boundaries) in these subvolumes, or find a pre-trained model that does a good job.
3. Once you have found or trained a decent classifier that works on your subvolumes, run the same classifier on the full dataset.

#### 1. Subvolume extraction

There are many ways to extract subvolumes from large datasets.
The best approach will probably to manually identify important regions in the dataset that need to be classified correctly.
* You could use a tool like MoBIE or BigDataViewer to find good coordinates and then export those crops.
* You could upload the dataset to Webknossos, create an annotation, and add Bounding Boxes to the annotation. Then download each bounding box individually.
* You could have an LLM write a script for you.

#### 2. Boundary classifier training

Our (obviously biased) recommendation: Use ilastik and train on your subvolumes.
* Browse https://bioimage.io for a model with keywords like "electron microscopy", "boundary", "membrane", then try the models in the Neural Network workflow
* Pixel Classification workflow
* Autocontext workflow
* Trainable Domain Adaptation workflow

In each workflow, train a classifier that distinguishes "membrane" from "everything else". Configure the export to export a *single channel* - the membrane channel (use the subregion settings in the export settings dialog).

#### 3. Generate probabilities for the whole dataset

Use batch processing in the respective ilastik workflow that you trained on your subvolumes.
Your dataset MUST be in HDF5 (.h5) format, or OME-Zarr, to make it possible for ilastik to process the dataset without trying to load it all and overloading your computer's memory.

### Trained ilastik Multicut project

Use the "Boundary-Based Segmentation with Multicut" workflow:
* Input data: Load your raw data subvolumes, then switch to Probabilities tab and load the probabilities for each subvolume in the corresponding line
* DT Watershed: Do optimise the parameters here.
  * The superpixel boundaries MUST align with your membranes.
  * It's no problem if there are *additional* boundaries (oversegmentation).
    You will train a boundary classifier that can easily learn to get rid of them and merge superpixels.
  * But you cannot later insert boundaries afterwards where there are none, and you cannot redraw boundaries if they are slightly off.
* Multicut: Train the classifier on all of your subvolumes.
  Remember:
  * Left mouse button: Lose it (bad boundary)
  * Right mouse button: Remain (good boundary)
* There is no need to actually export segmentations. Just save the project file and take it into blimp.

---

## Installation

```bash
# Download
git clone https://github.com/btbest/blockwise-ilastik-multicut.git
cd blockwise-ilastik-multicut

# Create conda environment (installs dependencies)
conda env create -n blimp -f environment.yml
conda activate blimp

# Install blimp
pip install -e .
```

The `blimp` command will now be available.
Try `blimp -h` to verify (this should print the help text).

---

## Run

`cd` to the folder with your data and the `.ilp`.

```bash
blimp \
    --ilp my_project.ilp \
    --raw raw.zarr \
    --probabilities boundary.zarr \
    --output-dir multicut_results/
```

**Output files in `multicut_results/`:**

| File | Contents |
|------|----------|
| `raw_segmentation.zarr` | Final segmentation (uint64, zyx) |
| `rf.pkl` | Fitted sklearn classifier (only if `--classifier-source sklearn`) |
| `params.json` | Call parameters for reproducibility |
| `raw_watershed.zarr` | Watershed superpixels (for debug or reuse) |

### Input formats

Both `--raw` and `--probabilities` accept:

- **zarr:** `/path/to/file.zarr`
- **HDF5:** `/path/to/file.h5` (must contain exactly one dataset)

Windows paths like `C:\Users\...\file.h5` are fine.

Volumes must be in **zyx(c) axis order** with the **same shape**. (ilastik-typical trailing singleton axis is ignored)

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

Note that `max_block_shape` determines block size for the blockwise *multicut* only.
The watershed uses ilastik block size (128x128x128).

What does beta do?
- `beta < 0.5`: favors merging segments
- `beta = 0.5`: balanced (default)
- `beta > 0.5`: favors splitting segments

### All options

```
Required:
  --ilp PATH                  Ilastik .ilp project file
  --raw PATH                  Raw data volume
  --probabilities PATH        Boundary probability volume
  --output-dir DIR            Output directory

Blockwise multicut:
  --max-block-shape Z Y X     Block size (default: 256 256 256)
  --halo Z Y X                Block overlap (default: 32 32 32)
  --beta FLOAT                Merge/split bias (default: 0.5)
  --threads INT               Parallel threads (default: 8)
  --solver                    kernighan-lin | greedy-additive | greedy-fixation
                              (default: kernighan-lin)

Classifier:
  --classifier-source         ilp | sklearn (default: ilp)
                              'ilp': extract trained classifier from .ilp
                              'sklearn': re-fit from training data
  --n-estimators INT          RF trees (only with --classifier-source sklearn; default: 100)

Watershed:
  --ws-threshold FLOAT        Seed threshold (default: from .ilp or 0.5)
  --ws-sigma FLOAT            Gaussian smoothing (default: from .ilp or 3.0)
  --ws-min-size INT           Min superpixel size (default: from .ilp or 100)
  --ws-invert                 Flip probability map (if high probability = *interior*, not boundary)

Reuse watershed:
  --ws-zarr PATH              Use pre-computed watershed (skips ws step)
  --no-keep-watershed         Discard watershed zarr (default: keep)
```

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
