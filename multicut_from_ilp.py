"""
multicut_from_ilp.py
Run elf's blockwise multicut on a (potentially large) volume using an edge
classifier extracted from an ilastik .ilp project file.

Two modes
---------
Default (in-memory):
    Loads all data into numpy arrays.  Suitable for volumes ≲ available RAM.

Lazy / blockwise (--lazy flag):
    All input channels are opened lazily (zarr or h5py Dataset).
    Watershed is computed block-by-block and written to a numpy memmap on
    disk.  Features and edge costs are accumulated blockwise.  Only a few
    blocks + the global edge cost dict need to fit in RAM simultaneously.
    Suitable for volumes up to tens of GB.

Usage
-----
In-memory (small volumes):

    python multicut_from_ilp.py \\
        --ilp my_project.ilp \\
        --rf rf.pkl \\
        --channels "Membrane Probabilities 0:/path/to/boundary.h5" \\
                   "Raw Data 0:/path/to/raw.h5" \\
        --output segmentation.h5 --output-key /seg

Lazy blockwise (large volumes, e.g. 20 GB):

    python multicut_from_ilp.py \\
        --ilp my_project.ilp \\
        --rf rf.pkl \\
        --channels "Membrane Probabilities 0:/path/to/boundary.zarr" \\
                   "Raw Data 0:/path/to/raw.zarr" \\
        --lazy \\
        --ws-zarr /scratch/watershed.zarr \\
        --output-zarr segmentation.zarr \\
        --block-shape 256 256 256 \\
        --halo 32 32 32

The watershed zarr is kept by default.  Pass --no-keep-watershed to delete
it after the run.  On a subsequent run pass --ws-zarr with the same path
to skip recomputation entirely.

Web / remote zarr (requires fsspec and aiohttp):

    python multicut_from_ilp.py \\
        --ilp my_project.ilp \\
        --rf rf.pkl \\
        --channels "Membrane Probabilities 0:https://webknossos.org/.../boundary.zarr/s0" \\
                   "Raw Data 0:https://webknossos.org/.../raw.zarr/s0" \\
        --lazy \\
        --ws-zarr /scratch/watershed.zarr \\
        --output-zarr segmentation.zarr \\
        --block-shape 256 256 256 \\
        --halo 32 32 32

Channel names must match those stored in the .ilp FeatureNames group.
Run: python -c "from ilp_reader import read_feature_names; print(read_feature_names('my.ilp'))"
to inspect channel names.
"""

import argparse
import math
import os
import pickle
import sys
import warnings

import h5py
import numpy as np

from functools import reduce
from operator import mul

from ilp_reader import read_feature_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_even_block_count(vol_shape, block_shape):
    """Return a (possibly reduced) block_shape whose total block count is even.

    ``elf``'s checkerboard two-pass watershed internally asserts that blocks
    can be split into two equally-sized halves.  This is only possible when
    the total number of blocks is even.  The total is odd when *all* per-axis
    block counts are odd (odd × odd × … = odd).

    Fix: find the first axis with an odd block count and *decrease* its block
    size just enough so that axis gains one more block (making that count even).
    The adjusted size is always ≤ the requested size, so memory usage stays
    within the user's budget.
    """
    n_blocks = [math.ceil(s / b) for s, b in zip(vol_shape, block_shape)]
    total = math.prod(n_blocks)
    if total % 2 == 0:
        return block_shape  # already fine

    block_shape = list(block_shape)
    for i, (s, b, n) in enumerate(zip(vol_shape, block_shape, n_blocks)):
        if n % 2 == 1:
            # Largest b_new such that ceil(s / b_new) == n + 1
            new_n = n + 1
            new_b = math.ceil(s / new_n)
            block_shape[i] = new_b
            break  # one even axis is enough to make the product even
    return tuple(block_shape)


# ---------------------------------------------------------------------------
# Channel / data loading helpers
# ---------------------------------------------------------------------------


_URL_SCHEMES = ("http://", "https://", "s3://", "gs://", "ftp://")


def _parse_channel_spec(spec: str):
    """
    Parse a channel specification of one of these forms:

        "Channel Name:/path/to/file.h5"               – HDF5 (single dataset, auto-detected)
        "Channel Name:/path/to/file.zarr"              – local zarr
        "Channel Name:https://host/path/to/array.zarr" – remote zarr URL
        "Channel Name:C:\\Users\\...\\file.h5"         – Windows absolute path

    Returns (channel_name, file_path_or_url, None).

    Everything after the first colon is treated as the file path.  HDF5
    dataset keys are no longer specified via colon notation; if an HDF5 file
    contains more than one dataset a ValueError is raised when it is opened.
    """
    if ":" not in spec:
        raise ValueError(
            f"Channel spec must be 'ChannelName:/path/to/file', got: {spec!r}"
        )
    first_colon = spec.index(":")
    channel_name = spec[:first_colon]
    file_path = spec[first_colon + 1:]
    return channel_name, file_path, None


def _open_channel_lazy(path: str, key: str | None):
    """
    Return a lazy array-like object for the channel data.

    For HDF5 files: the file must contain exactly one dataset, which is
      returned as an open h5py.Dataset (supports slice indexing).  Pass
      key=None (default); a ValueError is raised if the file contains more
      than one dataset.
    For local zarr stores: returns the zarr Array or Group item.
    For remote URLs (http/https/s3/…): opens the zarr store via fsspec;
      requires the ``fsspec`` package (and ``aiohttp`` for http/https URLs).

    The caller is responsible for keeping file handles open (see _ChannelStore).
    """
    if path.endswith(".h5") or path.endswith(".hdf5"):
        fh = h5py.File(path, "r")
        if key is not None:
            return fh[key], fh
        # Auto-detect the single dataset in the file.
        datasets = []
        fh.visititems(
            lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        if len(datasets) == 0:
            fh.close()
            raise ValueError(f"No datasets found in HDF5 file: {path!r}")
        if len(datasets) > 1:
            fh.close()
            raise ValueError(
                f"HDF5 file {path!r} contains multiple datasets {datasets}. "
                "The file must contain exactly one dataset."
            )
        return fh[datasets[0]], fh  # (dataset, handle_to_close)
    try:
        import zarr

        if any(path.startswith(s) for s in _URL_SCHEMES):
            try:
                import fsspec
            except ImportError as exc:
                raise ImportError(
                    "fsspec is required to open remote zarr URLs. "
                    "Install it with: pip install fsspec aiohttp"
                ) from exc
            mapper = fsspec.get_mapper(path)
            try:
                store = zarr.open(mapper, mode="r")
            except Exception:
                # zarr ≥3 probes zarr.json first (v3 format); the remote array
                # may be zarr v2 which stores metadata in .zarray instead.
                # Retry with an explicit v2 format request.
                zarr_major = int(zarr.__version__.split(".")[0])
                if zarr_major >= 3:
                    store = zarr.open(mapper, mode="r", zarr_format=2)
                else:
                    store = zarr.open(mapper, mode="r", zarr_version=2)
        else:
            store = zarr.open(path, mode="r")

        arr = store[key] if key else store
        return arr, None  # zarr manages its own handles
    except (ImportError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Cannot open {path}: {exc}") from exc


def _load_channel(path: str, key: str | None) -> np.ndarray:
    """Load a full channel into a numpy array (in-memory mode)."""
    arr, fh = _open_channel_lazy(path, key)
    data = arr[()]
    if fh is not None:
        fh.close()
    return data


class _ChannelStore:
    """Context manager that holds open lazy handles for all channels."""

    def __init__(self, channel_specs: list):
        self._specs = channel_specs
        self._handles = []
        self.arrays = {}  # channel_name → lazy array

    def __enter__(self):
        for spec in self._specs:
            ch_name, fpath, fkey = _parse_channel_spec(spec)
            arr, fh = _open_channel_lazy(fpath, fkey)
            self.arrays[ch_name] = arr
            if fh is not None:
                self._handles.append(fh)
        return self

    def __exit__(self, *_):
        for fh in self._handles:
            try:
                fh.close()
            except Exception:
                pass


class _Float32LazyArray:
    """Thin wrapper that casts sliced blocks to float32.

    vigra.analysis.watershedsNew only supports uint8 and float32.  H5py
    datasets and zarr arrays stored as float64 (or any other type) must be
    cast before being handed to elf / vigra.
    """

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.dtype = np.dtype("float32")
        self.ndim = arr.ndim

    def __getitem__(self, key):
        return np.asarray(self._arr[key], dtype=np.float32)

def _safe_distance_transform_watershed(input_, threshold, sigma_seeds, mask=None, **kwargs):
    """Wraps elf's distance_transform_watershed, handling flat / empty blocks.

    When a block contains no pixels above *threshold* (or when the resulting
    distance transform is entirely zero), elf's internal normalisation step
    ``dt / dt.max()`` produces NaN which propagates into vigra and ultimately
    causes a dtype mis-match crash (uint64 += float64).  Return an all-zero
    (background) segmentation immediately in that case.
    """
    from elf.segmentation.watershed import distance_transform_watershed

    # Use the masked region if a mask is provided, otherwise the full block.
    effective = input_ if mask is None else input_[mask]
    if effective.size == 0 or not (effective > threshold).any():
        return np.zeros(input_.shape, dtype="uint64"), 0

    return distance_transform_watershed(
        input_, threshold=threshold, sigma_seeds=sigma_seeds, mask=mask, **kwargs
    )


def _bigintprod(nums) -> int:
    """Product of an iterable using pure-Python integers.

    numpy.prod(nifty_block_shape, dtype=uint64) silently returns float64 on
    Windows when nifty exposes block-shape elements as 32-bit C integers:
    once the accumulated product exceeds INT32_MAX (~2.1 B) numpy promotes the
    accumulator to float64, ignoring the requested dtype.  Using Python's
    arbitrary-precision integers avoids the issue entirely.
    """
    return reduce(mul, map(int, nums), 1)


def _blockwise_two_pass_watershed(
        input_, block_shape, halo, ws_function=None, n_threads=None,
        mask=None, output=None, **kwargs
):
    """Drop-in replacement for elf's blockwise_two_pass_watershed.

    Identical to the elf implementation except the offset computation uses
    _bigintprod instead of np.prod to avoid a silent int32-overflow-to-float64
    promotion that occurs on Windows when nifty blockShape elements are
    32-bit C integers (offset = block_id * product_of_shape can exceed
    INT32_MAX for large volumes with many blocks).
    """
    import multiprocessing
    from concurrent import futures
    import vigra
    import nifty.tools as nt
    from tqdm import tqdm
    from elf.segmentation.watershed import distance_transform_watershed
    from elf.util import divide_blocks_into_checkerboard

    if ws_function is None:
        ws_function = distance_transform_watershed

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if output is None:
        output = np.zeros(input_.shape, dtype="uint64")
    assert output.shape == input_.shape

    blocking = nt.blocking([0, 0, 0], list(input_.shape), list(block_shape))
    block_ids_pass_one, block_ids_pass_two = divide_blocks_into_checkerboard(blocking)

    def run_block_one(block_id):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        outer_bb = tuple(slice(s, e) for s, e in zip(block.outerBlock.begin, block.outerBlock.end))
        input_block = input_[outer_bb]
        mask_block = None if mask is None else mask[outer_bb]
        ws, _ = ws_function(input_block, mask=mask_block, **kwargs)

        inner_bb = tuple(slice(s, e) for s, e in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(slice(s, e) for s, e in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
        ws = vigra.analysis.labelMultiArrayWithBackground(ws[local_bb].astype("uint32")).astype("uint64")

        # Use bigintprod to avoid silent int32-overflow-to-float64 on Windows.
        offset = np.uint64(_bigintprod([block_id] + list(blocking.blockShape)))
        if mask_block is None:
            ws += offset
        else:
            ws[mask_block[local_bb]] += offset
        output[inner_bb] = ws

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(run_block_one, block_ids_pass_one), total=len(block_ids_pass_one),
            desc="Run pass one of two-pass watershed",
        ))

    def run_block_two(block_id):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        outer_bb = tuple(slice(s, e) for s, e in zip(block.outerBlock.begin, block.outerBlock.end))
        input_block = input_[outer_bb]
        mask_block = None if mask is None else mask[outer_bb]
        seeds_block = output[outer_bb]

        seeds_block, seed_max, seed_id_mapping = vigra.analysis.relabelConsecutive(
            seeds_block, start_label=1, keep_zeros=True
        )

        ws, ws_max_id = ws_function(input_block, mask=mask_block, seeds=seeds_block, **kwargs)

        inner_bb = tuple(slice(s, e) for s, e in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(slice(s, e) for s, e in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
        ws = ws[local_bb]

        offset = _bigintprod([block_id] + list(blocking.blockShape))
        id_mapping = {v: k for k, v in seed_id_mapping.items()}
        assert 0 in id_mapping
        id_mapping.update({seed_id: seed_id + offset for seed_id in range(seed_max + 1, ws_max_id + 1)})
        ws = nt.takeDict(id_mapping, ws)

        output[inner_bb] = ws

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(run_block_two, block_ids_pass_two), total=len(block_ids_pass_two),
            desc="Run pass two of two-pass watershed",
        ))

    _, max_id, _ = vigra.analysis.relabelConsecutive(output, out=output)
    return output, max_id


def _safe_distance_transform_watershed(input_, threshold, sigma_seeds, mask=None, **kwargs):
    """Wraps elf's distance_transform_watershed, handling flat / empty blocks.

    When a block contains no pixels above *threshold* (or when the resulting
    distance transform is entirely zero), elf's internal normalisation step
    ``dt / dt.max()`` produces NaN which propagates into vigra and ultimately
    causes a dtype mis-match crash (uint64 += float64).  Return an all-zero
    (background) segmentation immediately in that case.
    """
    from elf.segmentation.watershed import distance_transform_watershed

    # Use the masked region if a mask is provided, otherwise the full block.
    effective = input_ if mask is None else input_[mask]
    if effective.size == 0 or not (effective > threshold).any():
        return np.zeros(input_.shape, dtype="uint64"), 0

    return distance_transform_watershed(
        input_, threshold=threshold, sigma_seeds=sigma_seeds, mask=mask, **kwargs
    )


# ---------------------------------------------------------------------------
# Watershed zarr: open existing or compute fresh
# ---------------------------------------------------------------------------


def _open_or_compute_watershed_zarr(
    ws_zarr_path, boundary_lazy, vol_shape, block_shape, halo,
    use_2dws, ws_threshold, ws_sigma, n_threads,
):
    """Return an open zarr array containing the watershed and the node count.

    The zarr stores **0-indexed** superpixel labels (0 … n_superpixels-1).
    The zarr attribute ``"n_superpixels"`` holds the total superpixel count
    (= the number of nifty graph nodes).

    If a zarr already exists at *ws_zarr_path* with the correct shape and the
    ``"n_superpixels"`` attribute, it is opened read-only and returned
    immediately — the watershed computation is skipped entirely.  This lets
    callers reuse a watershed from a previous run for faster debugging.

    When computing fresh the watershed is first written into a temporary
    numpy memmap (required by elf / vigra), then copied block-by-block into
    a zarr with labels shifted to 0-indexed.  The memmap is deleted
    immediately after the copy.
    """
    import zarr
    import nifty.tools as nt
    from elf.segmentation.watershed import stacked_watershed
    from pathlib import Path as _Path

    # --- Try to reuse an existing watershed zarr ---
    if os.path.exists(ws_zarr_path):
        try:
            existing = zarr.open(ws_zarr_path, mode="r")
            if (
                tuple(existing.shape) == tuple(vol_shape)
                and "n_superpixels" in existing.attrs
            ):
                n_nodes = int(existing.attrs["n_superpixels"])
                print(
                    f"Reusing existing watershed zarr: {ws_zarr_path} "
                    f"({n_nodes} superpixels) — skipping computation"
                )
                return existing, n_nodes
            print(
                f"  Existing {ws_zarr_path!r} has wrong shape or missing "
                f"attribute, recomputing …"
            )
        except Exception as exc:
            print(
                f"  Could not open {ws_zarr_path!r} ({exc}), recomputing …"
            )

    # --- Compute fresh watershed into a temporary memmap ---
    _memmap_path = str(_Path(ws_zarr_path).parent / "_ws_compute_tmp.dat")
    print(f"Computing blockwise watershed → {ws_zarr_path} …")
    ws_memmap = np.memmap(_memmap_path, dtype="uint64", mode="w+", shape=vol_shape)

    try:
        if use_2dws:
            print("  Using stacked 2D watershed (lazy z-slices) …")
            _, max_id = stacked_watershed(
                boundary_lazy,
                threshold=ws_threshold, sigma_seeds=ws_sigma,
                n_threads=n_threads, output=ws_memmap,
            )
        else:
            ws_block_shape = _ensure_even_block_count(vol_shape, block_shape)
            if ws_block_shape != block_shape:
                print(
                    f"  block_shape reduced {block_shape} → {ws_block_shape} "
                    f"(total block count must be even for checkerboard two-pass)"
                )
            _, max_id = _blockwise_two_pass_watershed(
                boundary_lazy,
                block_shape=ws_block_shape,
                halo=halo,
                ws_function=_safe_distance_transform_watershed,
                threshold=ws_threshold,
                sigma_seeds=ws_sigma,
                n_threads=n_threads,
                output=ws_memmap,
            )

        n_nodes = int(max_id)  # vigra 1-indexed max = number of superpixels
        print(f"  {n_nodes} superpixels; writing to zarr …")

        # Copy memmap (1-indexed, 1…max_id) → zarr (0-indexed, 0…max_id-1).
        # All pixels are guaranteed ≥ 1 (no mask in the lazy pipeline), so
        # the uint64 subtraction never wraps around.
        ws_zarr_arr = zarr.open(
            ws_zarr_path, mode="w",
            shape=vol_shape, dtype="uint64",
            chunks=block_shape,
        )
        _copy_blocking = nt.blocking(
            [0] * len(vol_shape), list(vol_shape), list(block_shape)
        )
        for _bid in range(_copy_blocking.numberOfBlocks):
            _blk = _copy_blocking.getBlock(_bid)
            _bb = tuple(
                slice(s, e) for s, e in zip(_blk.begin, _blk.end)
            )
            ws_zarr_arr[_bb] = ws_memmap[_bb] - np.uint64(1)
        ws_zarr_arr.attrs["n_superpixels"] = n_nodes

        print(f"  Watershed zarr written to {ws_zarr_path}")

    finally:
        del ws_memmap
        try:
            os.remove(_memmap_path)
        except Exception as _e:
            warnings.warn(
                f"Could not remove watershed temp file {_memmap_path!r}: {_e}"
            )

    return ws_zarr_arr, n_nodes


# ---------------------------------------------------------------------------
# Boundary channel identification
# ---------------------------------------------------------------------------


def _find_boundary_channel(feature_names: dict) -> str:
    """
    Identify the boundary/probability channel from the .ilp FeatureNames dict.

    Looks for a channel whose name contains 'boundary', 'wsdt', 'probabilit',
    or 'membrane' (case-insensitive).  Falls back to the first channel that
    does not contain 'raw', then to the first channel overall.

    This is needed because the watershed must run on the boundary probability
    map, not on raw intensity — and the dict insertion order cannot be relied
    upon to place the boundary channel first.
    """
    for name in feature_names:
        lower = name.lower()
        if any(kw in lower for kw in ("boundary", "wsdt", "probabilit", "membrane")):
            return name
    for name in feature_names:
        if "raw" not in name.lower():
            return name
    return next(iter(feature_names))


# ---------------------------------------------------------------------------
# Feature computation (ilastikrag)
# ---------------------------------------------------------------------------


def compute_ilastikrag_features(
    superpixels: np.ndarray,
    channel_data: dict,
    feature_names: dict,
):
    """
    Compute ilastikrag edge features for a (block of) superpixels.

    Parameters
    ----------
    superpixels : uint32/uint64 ndarray  (must be a plain numpy array)
    channel_data : dict  {channel_name: ndarray}
    feature_names : dict  {channel_name: [feature_name, ...]}

    Returns
    -------
    features : float32 ndarray  (N_edges, N_features)
    edge_ids : uint64 ndarray   (N_edges, 2)
    """
    import ilastikrag
    import pandas as pd
    import vigra

    ndim = superpixels.ndim
    axes = "zyx"[-ndim:]
    sp_vigra = vigra.taggedView(superpixels.astype(np.uint32), axes)
    rag = ilastikrag.Rag(sp_vigra)

    feature_dfs = []
    for channel_name, feat_names in feature_names.items():
        if channel_name not in channel_data:
            raise KeyError(
                f"Channel {channel_name!r} is required by the classifier but "
                f"was not provided. Available: {list(channel_data)}"
            )
        data = vigra.taggedView(np.asarray(channel_data[channel_name], dtype=np.float32), axes)
        df = rag.compute_features(data, feat_names)
        feat_cols = [c for c in df.columns if c not in ("sp1", "sp2")]
        df = df[feat_cols].rename(
            columns={c: f"{channel_name} {c}" for c in feat_cols}
        )
        feature_dfs.append(df)

    edge_ids = rag.edge_ids.astype(np.uint64)  # (N_edges, 2)
    features = pd.concat(feature_dfs, axis=1).values.astype(np.float32)
    return features, edge_ids


# ---------------------------------------------------------------------------
# In-memory pipeline (original, for moderate volumes)
# ---------------------------------------------------------------------------


def _run_in_memory(
    ilp_path, rf, channel_specs, output_path, output_key,
    beta, block_shape, halo, internal_solver, n_threads,
    use_2dws, ws_threshold, ws_sigma,
):
    import nifty
    from elf.segmentation.features import compute_rag, project_node_labels_to_pixels
    from elf.segmentation.multicut import blockwise_multicut, compute_edge_costs
    from elf.segmentation.watershed import distance_transform_watershed, stacked_watershed

    feature_names = read_feature_names(ilp_path)

    channel_data = {}
    for spec in channel_specs:
        ch_name, fpath, fkey = _parse_channel_spec(spec)
        print(f"  Loading {ch_name!r} from {fpath} …")
        channel_data[ch_name] = _load_channel(fpath, fkey)

    vol_shape = next(iter(channel_data.values())).shape
    boundary_channel = _find_boundary_channel(feature_names)
    boundary_map = channel_data[boundary_channel].astype(np.float32)

    print("Computing watershed …")
    if use_2dws:
        watershed, _ = stacked_watershed(
            boundary_map, threshold=ws_threshold, sigma_seeds=ws_sigma, n_threads=n_threads,
        )
    else:
        watershed, _ = distance_transform_watershed(
            boundary_map, threshold=ws_threshold, sigma_seeds=ws_sigma,
        )
    watershed = watershed.astype(np.uint32)

    print("Computing features …")
    # Compute features while the watershed still has vigra-convention 1-indexed
    # labels; ilastikrag.Rag treats 0 as background and must see 1-indexed SPs.
    features, edge_ids = compute_ilastikrag_features(watershed, channel_data, feature_names)

    # Re-index watershed and edge_ids to 0-based.
    # vigra / elf watershed is 1-indexed (labels 1..N, 0 never used without a mask).
    # Keeping the phantom node 0 in the nifty graph causes an isolated node that
    # triggers an off-by-one in blockwise_mc_impl when it sizes the reduced graph
    # from edge endpoints only (conda-forge python-elf 0.7.4).
    if watershed.min() > 0:
        watershed = watershed - 1
        edge_ids = edge_ids - 1

    n_labels = int(watershed.max()) + 1
    print(f"  {n_labels} superpixels")

    print("Predicting edge probabilities …")
    split_col = int(np.argmax(rf.classes_))
    edge_probs = rf.predict_proba(features)[:, split_col].astype(np.float32)

    costs = compute_edge_costs(edge_probs, beta=beta)

    print("Building graph …")
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(edge_ids)

    print(f"Running blockwise multicut (block_shape={block_shape}) …")
    node_labels = blockwise_multicut(
        graph, costs, watershed,
        internal_solver=internal_solver,
        block_shape=block_shape, n_threads=n_threads, halo=halo,
    )

    print("Projecting labels …")
    elf_rag = compute_rag(watershed, n_labels=n_labels, n_threads=n_threads)
    segmentation = project_node_labels_to_pixels(elf_rag, node_labels, n_threads=n_threads)
    print(f"  {len(np.unique(segmentation))} final segments")

    print(f"Saving to {output_path}:{output_key} …")
    with h5py.File(output_path, "a") as f:
        if output_key in f:
            del f[output_key]
        f.create_dataset(output_key, data=segmentation, compression="gzip")

    return segmentation


# ---------------------------------------------------------------------------
# Lazy / blockwise pipeline (for large volumes)
# ---------------------------------------------------------------------------


def _run_lazy(
    ilp_path, rf, channel_specs, output_zarr_path, output_zarr_key,
    beta, block_shape, halo, internal_solver, n_threads,
    use_2dws, ws_threshold, ws_sigma, ws_zarr_path,
    keep_watershed=True,
):
    import nifty
    import nifty.tools as nt
    from elf.segmentation.multicut import blockwise_multicut, compute_edge_costs

    try:
        import zarr
    except ImportError:
        raise ImportError("zarr is required for lazy mode: conda install -c conda-forge zarr")

    feature_names = read_feature_names(ilp_path)

    # --- Open all channels lazily ---
    with _ChannelStore(channel_specs) as store:
        lazy_arrays = store.arrays
        boundary_channel = _find_boundary_channel(feature_names)
        if boundary_channel not in lazy_arrays:
            raise KeyError(
                f"Boundary channel {boundary_channel!r} not in provided channels. "
                f"Available: {list(lazy_arrays)}"
            )
        boundary_lazy = _Float32LazyArray(lazy_arrays[boundary_channel])
        vol_shape = tuple(boundary_lazy.shape)
        print(f"Volume shape: {vol_shape}")

        # --- Blockwise watershed: reuse existing zarr or compute fresh ---
        ws_zarr_arr, n_nodes = _open_or_compute_watershed_zarr(
            ws_zarr_path=ws_zarr_path,
            boundary_lazy=boundary_lazy,
            vol_shape=vol_shape,
            block_shape=block_shape,
            halo=halo,
            use_2dws=use_2dws,
            ws_threshold=ws_threshold,
            ws_sigma=ws_sigma,
            n_threads=n_threads,
        )
        # ws_zarr_arr contains 0-indexed labels (0…n_nodes-1).
        # n_nodes is the number of superpixels = nifty graph node count.

        blocking = nt.blocking([0, 0, 0], list(vol_shape), list(block_shape))
        n_blocks = blocking.numberOfBlocks
        print(f"Watershed complete: {n_nodes} superpixels across {n_blocks} blocks.")

        # --- Blockwise feature computation ---
        # Accumulate edge arrays as numpy per block rather than building a Python
        # dict of tuple keys.  The dict approach costs ~300 bytes per edge in
        # Python object overhead; for large volumes (tens of millions of edges)
        # this easily exhausts RAM.  Numpy arrays use ~20 bytes per edge.
        print("Computing ilastikrag features blockwise …")

        split_col = int(np.argmax(rf.classes_))
        all_edges_list = []   # list of (N_i, 2) uint64 arrays (1-indexed, canonical)
        all_costs_list = []   # list of (N_i,)  float32 arrays

        for block_id in range(n_blocks):
            if block_id % max(1, n_blocks // 10) == 0:
                pct = 100 * block_id // n_blocks
                print(f"  block {block_id}/{n_blocks} ({pct}%) …")

            block = blocking.getBlockWithHalo(block_id, list(halo))
            outer_bb = tuple(
                slice(s, e)
                for s, e in zip(block.outerBlock.begin, block.outerBlock.end)
            )

            # Read 0-indexed zarr block and convert to 1-indexed for ilastikrag
            # (vigra treats 0 as background; our zarr uses 0 as first superpixel).
            ws_block = np.array(ws_zarr_arr[outer_bb]) + np.uint64(1)
            channel_block = {
                name: np.array(lazy_arrays[name][outer_bb])
                for name in feature_names
            }

            # Pass 1-indexed ws_block to ilastikrag (vigra treats 0 as background)
            features, edge_ids = compute_ilastikrag_features(
                ws_block, channel_block, feature_names
            )
            probs = rf.predict_proba(features)[:, split_col]
            costs = compute_edge_costs(probs.astype(np.float32), beta=beta)

            # Canonicalize edge endpoints (sp1 ≤ sp2) and accumulate.
            all_edges_list.append(np.sort(edge_ids, axis=1))
            all_costs_list.append(costs.astype(np.float32))

        if not all_edges_list:
            raise RuntimeError("No superpixel edges found; all blocks appear to be empty.")

        n_obs = sum(len(e) for e in all_edges_list)
        print(f"  {n_obs} edge observations across {n_blocks} blocks; deduplicating …")

        # Concatenate and deduplicate with numpy.
        # Pack each (sp1, sp2) pair into a single uint64 key, argsort, then keep
        # the last occurrence of each key (later blocks overwrite earlier ones on
        # ties, matching the behaviour of the former dict approach).
        all_edges = np.concatenate(all_edges_list, axis=0)      # (M, 2) uint64
        all_costs_arr = np.concatenate(all_costs_list, axis=0)  # (M,)   float32
        del all_edges_list, all_costs_list

        key = all_edges[:, 0] * np.uint64(n_nodes + 1) + all_edges[:, 1]
        order = np.argsort(key, kind="stable")
        key           = key[order]
        all_edges     = all_edges[order]
        all_costs_arr = all_costs_arr[order]
        del order

        # True where the key changes = last occurrence of each unique edge.
        keep = np.empty(len(key), dtype=bool)
        keep[-1] = True
        keep[:-1] = key[:-1] != key[1:]
        del key

        # Shift 1-indexed edge endpoints back to 0-indexed (zarr convention).
        edge_uvs  = all_edges[keep].astype(np.uint64) - np.uint64(1)
        edge_costs = all_costs_arr[keep]
        del all_edges, all_costs_arr, keep

        print(f"  {len(edge_uvs)} unique edges after deduplication.")

        # --- Build global nifty graph ---
        print(f"Building global graph ({n_nodes} nodes, {len(edge_uvs)} edges) …")
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(edge_uvs)
        del edge_uvs

        # --- Blockwise multicut ---
        # ws_zarr_arr supports __getitem__ with slice tuples, which is all
        # blockwise_mc_impl needs (it calls segmentation[bb] per block).
        print(f"Running blockwise multicut (block_shape={block_shape}, solver={internal_solver}) …")
        node_labels = blockwise_multicut(
            graph, edge_costs, ws_zarr_arr,
            internal_solver=internal_solver,
            block_shape=block_shape,
            n_threads=n_threads,
            halo=halo,
        )
        del edge_costs
        n_segments = len(np.unique(node_labels))
        print(f"Multicut complete: {n_segments} segments from {n_nodes} superpixels.")

        # --- Blockwise pixel projection → zarr ---
        print(f"Projecting labels and writing segmentation to {output_zarr_path} …")
        seg_out = zarr.open(
            output_zarr_path, mode="w",
            shape=vol_shape, dtype="uint64",
            chunks=block_shape,
        )
        for block_id in range(n_blocks):
            block = blocking.getBlock(block_id)
            inner_bb = tuple(
                slice(s, e) for s, e in zip(block.begin, block.end)
            )
            # ws_zarr_arr is 0-indexed; index directly into node_labels.
            ws_block = np.array(ws_zarr_arr[inner_bb])
            seg_block = node_labels[ws_block]
            seg_out[inner_bb] = seg_block

    # --- Keep or remove the watershed zarr ---
    if not keep_watershed:
        try:
            import shutil
            shutil.rmtree(ws_zarr_path)
            print(f"Removed watershed zarr {ws_zarr_path}")
        except Exception as e:
            warnings.warn(f"Could not remove watershed zarr {ws_zarr_path!r}: {e}")
    else:
        print(f"Watershed zarr kept at {ws_zarr_path}")

    print("Done.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_blockwise_multicut(
    ilp_path: str,
    rf_path: str,
    channel_specs: list,
    output_path: str = None,
    output_key: str = "/seg",
    output_zarr_path: str = None,
    output_zarr_key: str = "seg",
    lazy: bool = False,
    beta: float = 0.5,
    block_shape: tuple = (256, 256, 256),
    halo: tuple = (32, 32, 32),
    internal_solver: str = "kernighan-lin",
    n_threads: int = 8,
    use_2dws: bool = False,
    ws_threshold: float = 0.5,
    ws_sigma: float = 2.0,
    ws_zarr_path: str = "watershed.zarr",
    keep_watershed: bool = True,
):
    """
    Full multicut pipeline using an ilastik-trained sklearn RF.

    Parameters
    ----------
    lazy : bool
        If True, use the blockwise lazy pipeline (for large volumes).
        Requires zarr output (output_zarr_path).
        If False (default), load all data into memory (simpler, faster for
        volumes that fit in RAM).
    """
    print(f"Loading classifier from {rf_path} …")
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    feature_names = read_feature_names(ilp_path)
    print("Feature names per channel (from .ilp):")
    for ch, feats in feature_names.items():
        print(f"  {ch!r}: {feats}")

    if lazy:
        if output_zarr_path is None:
            raise ValueError("--output-zarr is required in lazy mode.")
        _run_lazy(
            ilp_path=ilp_path, rf=rf, channel_specs=channel_specs,
            output_zarr_path=output_zarr_path,
            output_zarr_key=output_zarr_key,
            beta=beta, block_shape=block_shape, halo=halo,
            internal_solver=internal_solver, n_threads=n_threads,
            use_2dws=use_2dws, ws_threshold=ws_threshold, ws_sigma=ws_sigma,
            ws_zarr_path=ws_zarr_path, keep_watershed=keep_watershed,
        )
    else:
        if output_path is None:
            raise ValueError("--output is required in in-memory mode.")
        return _run_in_memory(
            ilp_path=ilp_path, rf=rf, channel_specs=channel_specs,
            output_path=output_path, output_key=output_key,
            beta=beta, block_shape=block_shape, halo=halo,
            internal_solver=internal_solver, n_threads=n_threads,
            use_2dws=use_2dws, ws_threshold=ws_threshold, ws_sigma=ws_sigma,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run elf's blockwise multicut using an ilastik-trained sklearn RF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ilp", required=True)
    parser.add_argument("--rf", required=True, help="Pickled sklearn RF (from fit_classifier.py)")
    parser.add_argument(
        "--channels", nargs="+", required=True, metavar="NAME:FILE",
        help=(
            'Channel specs, e.g. "Membrane Probabilities 0:/boundary.h5". '
            "HDF5 files must contain exactly one dataset (auto-detected). "
            "Channel names must match those in the .ilp FeatureNames group."
        ),
    )

    # Output (in-memory mode)
    parser.add_argument("--output", default=None, help="Output HDF5 file (in-memory mode)")
    parser.add_argument("--output-key", default="/seg")

    # Output (lazy mode)
    parser.add_argument("--output-zarr", default=None, help="Output zarr path (lazy mode)")
    parser.add_argument("--output-zarr-key", default="seg")

    # Mode
    parser.add_argument(
        "--lazy", action="store_true",
        help="Enable lazy blockwise mode for large (>RAM) volumes.",
    )
    parser.add_argument(
        "--ws-zarr", default="watershed.zarr",
        help=(
            "Path for the watershed zarr (lazy mode, default: watershed.zarr). "
            "If the zarr already exists with the correct shape it is reused and "
            "the watershed step is skipped entirely."
        ),
    )
    parser.add_argument(
        "--keep-watershed", action=argparse.BooleanOptionalAction, default=True,
        help="Keep the watershed zarr after the run (default: keep). "
             "Pass --no-keep-watershed to delete it.",
    )

    # Multicut / watershed parameters
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--max-block-shape", type=int, nargs="+", default=[256, 256, 256], metavar="N",
        help="Maximum block shape; actual shape may be slightly smaller to satisfy "
             "checkerboard requirements (default: 256 256 256)",
    )
    parser.add_argument(
        "--halo", type=int, nargs="+", default=[32, 32, 32], metavar="N",
    )
    parser.add_argument(
        "--solver", default="kernighan-lin",
        choices=["kernighan-lin", "greedy-additive", "greedy-fixation"],
    )
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--use-2dws", action="store_true",
                        help="Use stacked 2D watersheds (for anisotropic data)")
    parser.add_argument("--ws-threshold", type=float, default=0.5)
    parser.add_argument("--ws-sigma", type=float, default=2.0)

    args = parser.parse_args()

    run_blockwise_multicut(
        ilp_path=args.ilp,
        rf_path=args.rf,
        channel_specs=args.channels,
        output_path=args.output,
        output_key=args.output_key,
        output_zarr_path=args.output_zarr,
        output_zarr_key=args.output_zarr_key,
        lazy=args.lazy,
        beta=args.beta,
        block_shape=tuple(args.max_block_shape),
        halo=tuple(args.halo),
        internal_solver=args.solver,
        n_threads=args.n_threads,
        use_2dws=args.use_2dws,
        ws_threshold=args.ws_threshold,
        ws_sigma=args.ws_sigma,
        ws_zarr_path=args.ws_zarr,
        keep_watershed=args.keep_watershed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
