import json
import logging
import math
import os
import warnings
from typing import Optional, Tuple

import h5py
import numpy as np

from functools import reduce
from operator import mul

from ilp_reader import read_feature_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OME_ZARR_ATTRS = {
    "multiscales": [{
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "datasets": [{
            "path": "s0",
            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
        }],
    }]
}


def _create_ome_zarr(path, shape, dtype, chunks):
    """Create a minimal OME-Zarr 0.4 store at *path* and return the s0 array.

    Creates a zarr group at *path*, writes OME-Zarr 0.4 multiscales metadata
    to the group's attributes, and creates the data array at sub-path "s0".
    Returns the array so callers can write into it directly.
    """
    import zarr
    group = zarr.open_group(path, mode="w")
    group.attrs.update(_OME_ZARR_ATTRS)
    return group.require_dataset("s0", shape=shape, dtype=dtype, chunks=chunks, exact=True, dimension_separator="/")

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


_H5_EXTENSIONS = (".h5", ".hdf5", ".hdf", ".ilp")


def _split_h5_path(path: str):
    """Split a compound HDF5 path into (file_path, internal_key).

    ilastik stores paths like ``/data/file.h5/exported_data`` where the
    portion after the ``.h5`` is the internal dataset path.  This helper
    finds the first HDF5-like extension in *path* and splits there.

    Returns
    -------
    (file_path, internal_key) if *path* contains an HDF5 extension,
    or (None, None) otherwise.  *internal_key* is ``None`` when no
    internal portion is present (e.g. plain ``file.h5``).
    """
    lower = path.lower()
    for ext in _H5_EXTENSIONS:
        idx = lower.find(ext)
        if idx == -1:
            continue
        end = idx + len(ext)
        # Make sure the extension is followed by nothing, a separator, or EOF
        if end < len(path) and path[end] not in ("/", "\\"):
            continue
        file_part = path[:end]
        rest = path[end:]
        # Strip leading separator from the internal key
        internal = rest.lstrip("/\\") or None
        return file_part, internal
    return None, None


def _coerce_json_attr(value):
    """Return an HDF5/zarr JSON-like attribute as text or a Python object."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _coerce_json_attr(value.item())
        if value.dtype.kind in ("S", "U"):
            return "".join(_coerce_json_attr(v) for v in value)
    return value


def _axes_from_axistags_json(axistags):
    """Extract an axis string like ``"zyxc"`` from vigra axistags JSON."""
    axistags = _coerce_json_attr(axistags)

    if isinstance(axistags, str):
        try:
            import vigra

            tags = vigra.AxisTags.fromJSON(axistags)
            return "".join(axis.key for axis in tags).lower()
        except ImportError:
            # Fall back to direct JSON parsing below for lightweight tests or
            # installs that only need axis metadata handling.
            pass
        except Exception as exc:
            raise ValueError(f"Could not parse vigra axistags: {exc}") from exc

        try:
            parsed = json.loads(axistags)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse vigra axistags JSON: {exc}") from exc
    elif isinstance(axistags, dict):
        parsed = axistags
    else:
        raise ValueError(
            f"Unsupported axistags attribute type: {type(axistags).__name__}"
        )

    try:
        return "".join(axis["key"] for axis in parsed["axes"]).lower()
    except Exception as exc:
        raise ValueError("Could not extract axis keys from axistags") from exc


def _read_axistags_axes(arr):
    """Return axes from ``arr.attrs['axistags']`` if present, else ``None``."""
    attrs = getattr(arr, "attrs", None)
    if attrs is None or "axistags" not in attrs:
        return None
    return _axes_from_axistags_json(attrs["axistags"])


def _normalize_axes(axes: str, ndim: int, source: str) -> str:
    axes = "".join(str(axes).strip().lower().replace(",", "").split())
    if len(axes) != ndim:
        raise ValueError(
            f"{source}: axes {axes!r} has length {len(axes)}, "
            f"but the array has {ndim} dimensions."
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"{source}: axes {axes!r} contains duplicate entries.")
    unknown = set(axes) - set("zyxc")
    if unknown:
        raise ValueError(
            f"{source}: axes {axes!r} contains unsupported axes "
            f"{sorted(unknown)!r}; expected z, y, x, and optional c."
        )
    if set(axes) not in (set("zyx"), set("zyxc")):
        raise ValueError(
            f"{source}: axes {axes!r} must contain exactly z, y, x, "
            "and optionally c."
        )
    return axes


def _default_axes_for_ndim(ndim: int, source: str) -> str:
    if ndim == 3:
        return "zyx"
    if ndim == 4:
        return "zyxc"
    raise ValueError(
        f"{source}: expected a 3-D or 4-D input array. Got {ndim} dimensions."
    )


def _expand_spatial_key(key):
    """Normalize a numpy-style key to one entry per output z/y/x axis."""
    if key is Ellipsis:
        return (slice(None), slice(None), slice(None))
    if not isinstance(key, tuple):
        key = (key,)

    if Ellipsis in key:
        if key.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis")
        ellipsis_pos = key.index(Ellipsis)
        n_missing = 3 - (len(key) - 1)
        if n_missing < 0:
            raise IndexError("too many indices for 3-D zyx array")
        key = key[:ellipsis_pos] + (slice(None),) * n_missing + key[ellipsis_pos + 1:]

    if len(key) > 3:
        raise IndexError("too many indices for 3-D zyx array")
    return key + (slice(None),) * (3 - len(key))


def _is_int_index(index):
    return isinstance(index, (int, np.integer))


class _ZYXLazyArray:
    """Lazy view that presents input arrays as 3-D zyx data.

    The source array may be stored in any order of z/y/x plus optional c.  The
    channel axis is selected while reading blocks so downstream code can keep
    using ordinary zyx slice tuples.
    """

    def __init__(self, arr, axes=None, channel_index=None, source="array"):
        self._arr = arr
        self._source = source
        self._source_shape = tuple(int(s) for s in arr.shape)
        self._source_ndim = len(self._source_shape)

        metadata_axes = None if axes is not None else _read_axistags_axes(arr)
        if axes is not None:
            resolved_axes = axes
            self.axes_source = "--axes"
        elif metadata_axes is not None:
            resolved_axes = metadata_axes
            self.axes_source = "axistags"
        else:
            if channel_index is not None:
                raise ValueError(
                    f"{source}: --channel-index requires axis metadata. "
                    "Provide --axes or an input array with vigra axistags."
                )
            resolved_axes = _default_axes_for_ndim(self._source_ndim, source)
            self.axes_source = "implicit"

        self.source_axes = _normalize_axes(
            resolved_axes, self._source_ndim, source
        )

        self._spatial_axis_to_source = {
            axis: self.source_axes.index(axis) for axis in "zyx"
        }
        self.shape = tuple(
            self._source_shape[self._spatial_axis_to_source[axis]]
            for axis in "zyx"
        )
        self.ndim = 3
        self.dtype = arr.dtype

        self._channel_axis = (
            self.source_axes.index("c") if "c" in self.source_axes else None
        )
        if self._channel_axis is None:
            self._channel_index = None
        else:
            n_channels = self._source_shape[self._channel_axis]
            if channel_index is None:
                if n_channels != 1:
                    raise ValueError(
                        f"{source}: input has {n_channels} channels on axis "
                        f"{self.source_axes!r}; pass --channel-index to select one."
                    )
                self._channel_index = 0
            else:
                if channel_index < 0 or channel_index >= n_channels:
                    raise ValueError(
                        f"{source}: channel index {channel_index} is out of "
                        f"bounds for {n_channels} channels."
                    )
                self._channel_index = int(channel_index)

    def __getitem__(self, key):
        spatial_key = _expand_spatial_key(key)
        source_key = []
        source_result_axes = []

        for axis in self.source_axes:
            if axis == "c":
                source_key.append(self._channel_index)
                continue

            out_axis = "zyx".index(axis)
            index = spatial_key[out_axis]
            source_key.append(index)
            if not _is_int_index(index):
                source_result_axes.append(axis)

        data = self._arr[tuple(source_key)]

        output_result_axes = [
            axis for axis, index in zip("zyx", spatial_key)
            if not _is_int_index(index)
        ]
        if len(output_result_axes) <= 1:
            return data

        transpose_order = tuple(
            source_result_axes.index(axis) for axis in output_result_axes
        )
        if transpose_order == tuple(range(len(transpose_order))):
            return data
        return np.transpose(data, transpose_order)


def _as_zyx_lazy_array(arr, axes=None, channel_index=None, source="array"):
    """Return a lazy array-like view with 3-D zyx shape."""
    return _ZYXLazyArray(
        arr, axes=axes, channel_index=channel_index, source=source
    )


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
    h5_file, h5_internal = _split_h5_path(path)
    if h5_file is not None:
        fh = h5py.File(h5_file, "r")
        # Explicit internal dataset path (from ilastik-style compound path)
        effective_key = key or h5_internal
        if effective_key is not None:
            return fh[effective_key], fh
        # Auto-detect the single dataset in the file.
        datasets = []
        fh.visititems(
            lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        if len(datasets) == 0:
            fh.close()
            raise ValueError(f"No datasets found in HDF5 file: {h5_file!r}")
        if len(datasets) > 1:
            fh.close()
            raise ValueError(
                f"HDF5 file {h5_file!r} contains multiple datasets {datasets}. "
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
            store = zarr.open(mapper, mode="r")
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

    def __init__(self, channel_specs: list, axes=None, channel_index=None):
        self._specs = channel_specs
        self._axes = axes
        self._channel_index = channel_index
        self._handles = []
        self.arrays = {}  # channel_name → lazy array

    def __enter__(self):
        for spec in self._specs:
            ch_name, fpath, fkey = _parse_channel_spec(spec)
            arr, fh = _open_channel_lazy(fpath, fkey)
            arr = _as_zyx_lazy_array(
                arr, axes=self._axes, channel_index=self._channel_index,
                source=fpath,
            )
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

    Integer-typed arrays (e.g. uint8 with values in 0–255) are automatically
    rescaled to [0, 1] so that downstream thresholds (typically 0.5) remain
    meaningful.

    A trailing size-1 channel dimension (e.g. shape (Z, Y, X, 1)) is
    automatically stripped so the rest of the pipeline always sees a 3-D
    spatial array.
    """

    def __init__(self, arr, normalize=True):
        self._arr = arr
        # Strip a trailing size-1 channel axis if present.
        self._squeeze_channel = (arr.ndim == 4 and arr.shape[-1] == 1)
        self.shape = arr.shape[:3] if self._squeeze_channel else arr.shape
        self.dtype = np.dtype("float32")
        self.ndim = len(self.shape)

        # If the source dtype is an integer type (e.g. uint8 0–255), we need
        # to rescale to [0, 1] so that thresholds and blending weights work
        # correctly.  Determine the scale factor once at init time.
        src_dtype = np.dtype(arr.dtype)
        if normalize and np.issubdtype(src_dtype, np.integer):
            self._scale = float(np.iinfo(src_dtype).max)
        else:
            self._scale = None

    def __getitem__(self, key):
        data = self._arr[key].astype(np.float32)
        if self._scale is not None:
            # This needs to EXACTLY copy the normalization in
            # EdgeTrainingWithMulticutWorkflow.connectLane.normalize_inplace !
            # The float imprecision difference between `data *= 1/255` vs `data /= 255`
            # is enough to propagate into Gaussians and local maxima to cause watershed differences.
            data[:] = data[:] / self._scale
        if self._squeeze_channel and data.ndim == 4:
            data = data[..., 0]
        return data


class _InvertedLazyArray:
    """Lazy wrapper that returns ``1 - block`` on every slice read.

    Used when ``InvertPixelProbabilities`` is active **and** the input
    probability map has high-value = interior (i.e. it represents
    *background* probability rather than *boundary* probability).

    elf's ``distance_transform_watershed`` expects **high-value = boundary**:
    it thresholds with ``input_ > threshold`` to find boundary foreground,
    then computes a distance transform of the interior (background) pixels
    to find seeds at the centres of interior regions.  When the input is
    already high-value = boundary, no inversion is needed.  When the input
    is high-value = interior (the opposite convention), this wrapper flips
    the values so elf sees high-value = boundary.

    Only the watershed input is inverted; the raw boundary values used for
    edge-feature computation are left unchanged.
    """

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.ndim  = arr.ndim

    def __getitem__(self, key):
        return np.float32(1.0) - self._arr[key]


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

    # Do NOT call vigra.analysis.relabelConsecutive on the full memmap here.
    # For a large volume (e.g. 5 GB input → ~38 GB uint64 memmap) that call is
    # single-threaded and tries to pull the entire array into RAM, causing many
    # hours of stall.  The caller (_open_or_compute_watershed_zarr) performs the
    # relabeling blockwise in parallel via _relabel_and_write_zarr instead.
    return output, None


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
# Parallel relabel + zarr write (replaces serial relabelConsecutive + copy)
# ---------------------------------------------------------------------------


def _relabel_and_write_zarr(ws_memmap, ws_zarr_arr, vol_shape, block_shape, n_threads):
    """Map sparse uint64 labels in *ws_memmap* to consecutive 0-indexed labels
    and write them to *ws_zarr_arr*.  Returns the number of unique superpixels.

    Two parallel passes over the data:
      Pass A – each thread reads its blocks from the memmap and returns the
               unique label values found there.  Fully parallel, I/O-bound.
      Pass B – after the global sorted-unique array is built (one
               ``np.searchsorted`` lookup maps any sparse label to its
               0-indexed consecutive counterpart), each thread reads its
               blocks, applies the mapping and writes the zarr chunk.

    This replaces two serial operations that stall on large volumes:
      • ``vigra.analysis.relabelConsecutive(memmap, out=memmap)`` – reads and
        writes the entire memmap in one shot, potentially pulling tens of GB
        into RAM.
      • A sequential Python ``for`` loop that copies memmap → zarr one block
        at a time.
    """
    import nifty.tools as nt
    from concurrent import futures
    from tqdm import tqdm

    blocking = nt.blocking([0] * len(vol_shape), list(vol_shape), list(block_shape))
    n_blocks = blocking.numberOfBlocks

    # --- Pass A: collect unique labels per block ---
    def _collect(bid):
        blk = blocking.getBlock(bid)
        bb = tuple(slice(s, e) for s, e in zip(blk.begin, blk.end))
        return np.unique(ws_memmap[bb])

    print("  Collecting unique superpixel labels (parallel) …")
    with futures.ThreadPoolExecutor(n_threads) as tp:
        per_block = list(tqdm(
            tp.map(_collect, range(n_blocks)),
            total=n_blocks, desc="  Unique-label scan",
        ))

    all_labels = np.unique(np.concatenate(per_block))
    del per_block
    n_nodes = int(len(all_labels))
    print(f"  {n_nodes} unique superpixel labels found.")

    # --- Pass B: apply mapping and write zarr in parallel ---
    # np.searchsorted(all_labels, x) maps each sparse label to its 0-indexed
    # consecutive position; this is correct because all_labels is sorted and
    # every element of ws_memmap is guaranteed to appear in all_labels.
    def _write(bid):
        blk = blocking.getBlock(bid)
        bb = tuple(slice(s, e) for s, e in zip(blk.begin, blk.end))
        block = np.array(ws_memmap[bb])
        ws_zarr_arr[bb] = np.searchsorted(all_labels, block).astype(np.uint64)

    print("  Writing watershed zarr (parallel) …")
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_write, range(n_blocks)),
            total=n_blocks, desc="  Zarr write",
        ))

    return n_nodes


# ---------------------------------------------------------------------------
# Ilastik-style parallel watershed  (mirrors ilastik's parallel_watershed)
# ---------------------------------------------------------------------------


def _ilastik_parallel_watershed(
    boundary_lazy, vol_shape,
    ws_threshold, ws_sigma, ws_min_size, ws_alpha,
    ws_pixel_pitch, ws_apply_nonmax,
    n_threads, output,
):
    """Blockwise watershed that exactly replicates ilastik's parallel_watershed.

    ilastik's ``OpWsdt.execute`` calls ``parallel_watershed`` on the full
    (in-RAM) boundary crop with ``block_shape=None`` and ``halo=None``, which
    resolve to 128³ voxels and 10-voxel halo respectively for 3-D data.
    The key property is **hard block boundaries**: every block is processed
    independently with no cross-block seed propagation.

    Because we use the same :func:`nifty.tools.blocking` call and identical
    per-block steps — ``elf.distance_transform_watershed`` followed by
    ``vigra.analysis.labelMultiArray`` — the superpixel boundaries produced
    here are pixel-identical to ilastik's, provided the same parameters are
    used.  The only known exception is very old projects where
    ``BlockwiseWatershed=False`` (ilastik ran on the full crop at once;
    we cannot replicate that for volumes that don't fit in RAM).

    Parameters
    ----------
    boundary_lazy : array-like supporting ``__getitem__`` with slice tuples
    vol_shape     : tuple[int, ...]   – (Z, Y, X) spatial shape
    ws_threshold  : float
    ws_sigma      : float             – applied to both seed map and weight map
    ws_min_size   : int
    ws_alpha      : float
    ws_pixel_pitch: list[float] | None
    ws_apply_nonmax : bool
    n_threads     : int
    output        : zarr.Array        - to write results into

    Returns
    -------
    (output, max_id) : the filled array and the maximum label value
    """
    import nifty.tools as nt
    import vigra
    from concurrent import futures
    from tqdm import tqdm
    from elf.segmentation.watershed import distance_transform_watershed

    ndim = len(vol_shape)
    assert ndim in [2, 3], f"Watershed segmentor will only work on 2D and 3D data, got shape {vol_shape!r}"
    # These are ilastik's hard-coded defaults for 3-D data (block_shape=None,
    # halo=None paths in parallel_watershed / get_blocking).
    BLOCK_SHAPE = (128,) * ndim
    HALO        = [10]  * ndim

    blocking = nt.blocking([0] * ndim, list(vol_shape), list(BLOCK_SHAPE))
    n_blocks  = blocking.numberOfBlocks
    # The returned watershed has 1-indexed superpixel IDs! There is no 0-superpixel.
    per_block_max = np.ones(n_blocks, dtype=np.int64)

    def _run_block(block_id):
        block      = blocking.getBlockWithHalo(block_id, HALO)
        outer_bb   = tuple(slice(s, e) for s, e in zip(block.outerBlock.begin,      block.outerBlock.end))
        local_bb   = tuple(slice(s, e) for s, e in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
        inner_bb   = tuple(slice(s, e) for s, e in zip(block.innerBlock.begin,      block.innerBlock.end))

        data_block = boundary_lazy[outer_bb]

        # Guard against empty / flat blocks to avoid NaN in elf's dt / dt.max().
        if not (data_block > ws_threshold).any():
            inner_shape = tuple(e - s for s, e in zip(block.innerBlock.begin, block.innerBlock.end))
            # ilastik has 1 for completely empty blocks, with a unique ID for each empty block.
            output[inner_bb] = np.ones(inner_shape, dtype=np.uint64)
            return 1

        ws_outer, _ = distance_transform_watershed(
            data_block,
            ws_threshold,
            ws_sigma,         # sigma_seeds
            ws_sigma,         # sigma_weights  (ilastik passes Sigma for both)
            ws_min_size,
            ws_alpha,
            ws_pixel_pitch,
            ws_apply_nonmax,
        )
        ws_outer = ws_outer.astype("uint32")
        # Same as ilastik: relabel the inner sub-block with vigra.labelMultiArray
        # so that labels are consecutive and disconnected fragments are separated.
        ws_inner = vigra.analysis.labelMultiArray(ws_outer[local_bb])
        output[inner_bb] = ws_inner.astype(np.uint64)
        return int(ws_inner.max())

    print(f"  (ilastik-style: {BLOCK_SHAPE} blocks, halo={HALO[0]})")
    with futures.ThreadPoolExecutor(n_threads) as tp:
        per_block_max[:] = list(tqdm(
            tp.map(_run_block, range(n_blocks)),
            total=n_blocks, desc="  Watershed blocks",
        ))

    cumulative = np.cumsum(per_block_max)

    def _add_offset(block_id):
        block      = blocking.getBlock(block_id)
        inner_bb = tuple(slice(s, e) for s, e in zip(block.begin, block.end))
        output[inner_bb] = output[inner_bb] + np.uint64(cumulative[block_id - 1])

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_add_offset, range(1, n_blocks)),
            total=(n_blocks - 1), desc="  Applying offsets",
        ))

    max_id = int(cumulative[-1]) if n_blocks > 0 else 0
    return output, max_id


# ---------------------------------------------------------------------------
# Watershed zarr: open existing or compute fresh
# ---------------------------------------------------------------------------


def _open_or_compute_watershed_zarr(
    ws_zarr_path, boundary_lazy, vol_shape, block_shape, halo,
    ws_method, ws_threshold, ws_sigma, ws_min_size, ws_alpha,
    ws_pixel_pitch, ws_apply_nonmax,
    n_threads,
) -> Tuple["zarr.Array", Optional[int]]:
    """Return an open zarr array containing the watershed and the node count.

    The zarr stores **0-indexed** superpixel labels (0 … n_superpixels-1).
    The zarr attribute ``"n_superpixels"`` holds the total superpixel count
    (= the number of nifty graph nodes).

    If a zarr already exists at *ws_zarr_path* with the correct shape and the
    ``"n_superpixels"`` attribute, it is opened read-only and returned
    immediately — the watershed computation is skipped entirely.  This lets
    callers reuse a watershed from a previous run for faster debugging.

    For the ``"ilastik"`` method the watershed writes directly into the zarr —
    no temporary memmap is needed because blocks are independent and produce
    globally consecutive 0-indexed labels without a relabeling pass.

    For ``"two-pass"`` and ``"2d"`` the watershed is first written into a
    temporary numpy memmap (required by elf / vigra for in-place seed
    propagation or stacked slices), then relabeled and copied to zarr.

    ws_method : ``"ilastik"`` | ``"two-pass"`` | ``"2d"``
        ``"ilastik"``   – mirrors ``parallel_watershed`` in ilastik's opWsdt.py:
                          128³ blocks, 10-voxel halo, hard block boundaries,
                          vigra.labelMultiArray per inner block, cumulative
                          offsets.  Produces pixel-identical superpixels when
                          the same parameters and the same boundary map are used.
        ``"two-pass"``  – elf checkerboard two-pass watershed (old default).
        ``"2d"``        – stacked 2-D watershed (for strongly anisotropic data).
    """
    import zarr
    from elf.segmentation.watershed import stacked_watershed
    from pathlib import Path as _Path

    # --- Try to reuse an existing watershed zarr ---
    expected_dataset_name = "s0"
    if os.path.exists(ws_zarr_path):
        try:
            existing_group = zarr.open_group(ws_zarr_path, mode="r")
            existing = existing_group[expected_dataset_name]
        except Exception as exc:
            raise ValueError(
                f"  Could not open {ws_zarr_path!r} (Expected zarr group "
                f"with a dataset {expected_dataset_name!r}). Message: {exc}"
            ) from exc

        if tuple(existing.shape) != tuple(vol_shape):
            raise ValueError(
                f"  Existing {ws_zarr_path!r} has wrong shape "
                f"(expected {tuple(vol_shape)}, found {existing.shape})"
            )
        print(
            f"Reusing existing watershed zarr: {ws_zarr_path} — skipping computation"
        )
        n_superpixels = existing.attrs.get("n_superpixels")
        return existing, n_superpixels

    print(f"Computing watershed ({ws_method!r} method) → {ws_zarr_path} …")

    if ws_method == "ilastik":
        # Blocks are independent and produce globally consecutive 0-indexed labels,
        # so we can write directly into the zarr — no memmap or relabeling needed.
        ndim = len(vol_shape)
        ws_zarr_arr = _create_ome_zarr(
            ws_zarr_path, shape=vol_shape, dtype="uint64",
            chunks=(128,) * ndim,   # must match _ilastik_parallel_watershed's BLOCK_SHAPE
        )
        _, n_superpixels = _ilastik_parallel_watershed(
            boundary_lazy, vol_shape,
            ws_threshold, ws_sigma, ws_min_size, ws_alpha,
            ws_pixel_pitch, ws_apply_nonmax,
            n_threads, ws_zarr_arr,
        )
        ws_zarr_arr.attrs["n_superpixels"] = n_superpixels
        print(f"  Watershed zarr written to {ws_zarr_path} ({n_superpixels} superpixels)")
        return ws_zarr_arr, n_superpixels

    # --- two-pass and 2d methods: compute into a temporary memmap, then
    #     relabel (labels are sparse / non-consecutive) and write to zarr ---
    _memmap_path = str(_Path(ws_zarr_path).parent / "_ws_compute_tmp.dat")
    ws_memmap = np.memmap(_memmap_path, dtype="uint64", mode="w+", shape=vol_shape)

    try:
        if ws_method == "2d":
            print("  Using stacked 2D watershed (lazy z-slices) …")
            _, max_id = stacked_watershed(
                boundary_lazy,
                threshold=ws_threshold, sigma_seeds=ws_sigma,
                sigma_weights=ws_sigma, min_size=ws_min_size, alpha=ws_alpha,
                n_threads=n_threads, output=ws_memmap,
            )
        elif ws_method == "two-pass":
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
                sigma_weights=ws_sigma,
                min_size=ws_min_size,
                alpha=ws_alpha,
                n_threads=n_threads,
                output=ws_memmap,
            )
        else:
            raise ValueError(f"Unknown ws_method {ws_method!r}; choose 'ilastik', 'two-pass', or '2d'.")

        ws_zarr_arr = _create_ome_zarr(
            ws_zarr_path, shape=vol_shape, dtype="uint64",
            chunks=block_shape,
        )
        n_superpixels = _relabel_and_write_zarr(
            ws_memmap, ws_zarr_arr, vol_shape, block_shape, n_threads
        )
        ws_zarr_arr.attrs["n_superpixels"] = n_superpixels
        print(f"  Watershed zarr written to {ws_zarr_path} ({n_superpixels} superpixels)")

    finally:
        del ws_memmap
        try:
            os.remove(_memmap_path)
        except Exception as _e:
            warnings.warn(
                f"Could not remove watershed temp file {_memmap_path!r}: {_e}"
            )

    return ws_zarr_arr, n_superpixels


# ---------------------------------------------------------------------------
# Channel handling
# ---------------------------------------------------------------------------


def _find_raw_channel(feature_names: dict) -> str:
    """Return the name of the raw data channel (contains 'raw', case-insensitive)."""
    for name in feature_names:
        if "raw" in name.lower():
            return name
    raise ValueError(
        f"Cannot identify raw data channel in: {list(feature_names)}. "
        "Expected a channel name containing 'raw' (case-insensitive)."
    )


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


def _build_channel_spec(channel_name: str, path: str) -> str:
    return f"{channel_name}:{path}"


# ---------------------------------------------------------------------------
# Feature computation (ilastikrag)
# ---------------------------------------------------------------------------


def _compute_features_two_sp(superpixels, channel_data, feature_names):
    """
    Compute ilastikrag-compatible edge features for a block with exactly
    2 foreground superpixels (1 edge) without calling ilastikrag's C++ layer.

    ilastikrag crashes on 2-node RAGs (a degenerate but valid configuration
    that occurs, e.g., when a single membrane bisects the entire block).
    This fallback replicates the standard ilastikrag feature semantics with
    pure numpy / scipy:

      standard_sp_*          → |stat(sp_a) − stat(sp_b)|   (absolute difference)
      standard_edge_*        → stat over boundary voxels
      edgeregion_edge_*radii → PCA eigenvalues of boundary-voxel coordinates

    Any other feature name falls back to 0.0 with a warning.

    Returns
    -------
    features : float32 ndarray  (1, N_features)
    edge_ids : uint64 ndarray   (1, 2)
    """
    import pandas as pd
    from scipy.ndimage import binary_dilation

    fg = np.sort(np.unique(superpixels[superpixels > 0]))
    sp_a, sp_b = int(fg[0]), int(fg[1])

    mask_a = superpixels == sp_a
    mask_b = superpixels == sp_b

    # 6-connected structuring element (face-adjacency only, matching vigra's RAG).
    ndim = superpixels.ndim
    struct = np.zeros((3,) * ndim, dtype=bool)
    center = (1,) * ndim
    for ax in range(ndim):
        for delta in (-1, 1):
            idx = list(center)
            idx[ax] += delta
            struct[tuple(idx)] = True

    # Two-sided boundary (voxels of sp_a adjacent to sp_b, and vice versa).
    boundary_mask = (mask_a & binary_dilation(mask_b, structure=struct)) | \
                    (mask_b & binary_dilation(mask_a, structure=struct))
    boundary_coords = np.argwhere(boundary_mask).astype(np.float32)

    feature_dfs = []
    for channel_name, feat_names in feature_names.items():
        data = np.asarray(channel_data[channel_name], dtype=np.float32)
        vals_a = data[mask_a].ravel()
        vals_b = data[mask_b].ravel()
        vals_e = data[boundary_mask].ravel()

        row = {}
        for fn in feat_names:
            if fn == "standard_sp_mean":
                row[fn] = float(abs(vals_a.mean() - vals_b.mean()))
            elif fn.startswith("standard_sp_quantiles_"):
                q = int(fn.rsplit("_", 1)[-1])
                row[fn] = float(abs(np.percentile(vals_a, q) - np.percentile(vals_b, q)))
            elif fn == "standard_edge_mean":
                row[fn] = float(vals_e.mean()) if vals_e.size else 0.0
            elif fn.startswith("standard_edge_quantiles_"):
                q = int(fn.rsplit("_", 1)[-1])
                row[fn] = float(np.percentile(vals_e, q)) if vals_e.size else 0.0
            elif fn.startswith("edgeregion_edge_regionradii_"):
                idx = int(fn.rsplit("_", 1)[-1])
                if boundary_coords.shape[0] >= ndim:
                    centered = boundary_coords - boundary_coords.mean(axis=0)
                    cov = (centered.T @ centered) / boundary_coords.shape[0]
                    eigvals = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0.0))
                    eigvals = eigvals[::-1]  # largest first (eigvalsh returns ascending)
                    row[fn] = float(eigvals[idx]) if idx < eigvals.size else 0.0
                else:
                    row[fn] = 0.0
            else:
                warnings.warn(
                    f"Feature {fn!r} has no 2-SP fallback implementation; using 0.0.",
                    stacklevel=4,
                )
                row[fn] = 0.0

        feat_cols = list(row.keys())
        df = pd.DataFrame([row])[feat_cols].rename(
            columns={c: f"{channel_name} {c}" for c in feat_cols}
        )
        feature_dfs.append(df)

    features = pd.concat(feature_dfs, axis=1).values.astype(np.float32)
    edge_ids = np.array([[sp_a, sp_b]], dtype=np.uint64)
    return features, edge_ids


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

    Raises
    ------
    ValueError  if the block has fewer than 2 unique foreground superpixel
                labels (no edges → nothing to compute).
    """
    import pandas as pd

    if len(np.unique(superpixels)) < 2:
        # Only one superpixel - no edges. Should never happen with ilastik_parallel_watershed
        # because the halo always contains superpixels from other blocks.
        return None, None

    import ilastikrag
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
        data = vigra.taggedView(channel_data[channel_name], axes)
        df = rag.compute_features(data, feat_names)
        feat_cols = [c for c in df.columns if c not in ("sp1", "sp2")]
        df = df[feat_cols].rename(
            columns={c: f"{channel_name} {c}" for c in feat_cols}
        )
        feature_dfs.append(df)

    edge_ids = rag.edge_ids.astype(np.uint64)  # (N_edges, 2)
    features = pd.concat(feature_dfs, axis=1).values
    assert features.dtype == np.float32, f"Unexpected feature dtype: {features.dtype}"
    return features, edge_ids


def compute_edge_costs(edge_ids, edge_probabilities: np.ndarray, beta, threshold) -> np.ndarray:
    """
    Convert edge probabilities to energies for the multicut problem.

    Copied from ilastik.applets.multicut.opMulticut.compute_edge_weights.

    edge_ids:
        The list of edges in the graph. shape=(N, 2)
    edge_probabilities:
        1-D, float (1.0 means edge is CUT, disconnecting the two SPs)
    beta:
        scalar (float)
    threshold:
        scalar (float), moves the 0 of the edge weights (default threshold = 0.5)

    Special behavior:
        If any node has ID 0, all of it's edges will be given an
        artificially low energy, to prevent it from merging with its
        neighbors, regardless of what the edge_probabilities say.
    """

    # P(Edge=CUT), clipped to avoid log(0).
    p1 = np.clip(edge_probabilities, 0.001, 0.999)
    # Rescale [0; t] to [0; 0.5], and [t; 1] to [0.5; 1].
    p1 = np.where(p1 <= threshold, p1 / (2 * threshold), 0.5 + (p1 - threshold) / (2 * (1 - threshold)))
    # log((p0 / p1) + log((1-beta) / beta)), where p0 = 1 - p1 is P(Edge=NOT CUT).
    edge_weights = np.log(np.reciprocal(p1) - 1) + np.log(1 / beta - 1)

    # See note special behavior, above
    edges_touching_zero = edge_ids[:, 0] == 0
    if edges_touching_zero.any():
        #logger.warning("Volume contains label 0, which will be excluded from the segmentation.")
        MINIMUM_ENERGY = -1000.0
        edge_weights[edges_touching_zero] = MINIMUM_ENERGY

    return edge_weights


# ---------------------------------------------------------------------------
# Lazy / blockwise pipeline (for large volumes)
# ---------------------------------------------------------------------------


def _run_lazy(
    ilp_path, rf, channel_specs, output_zarr_path, output_zarr_key,
    block_shape, halo, internal_solver, n_threads,
    ws_method, ws_threshold, ws_sigma, ws_min_size, ws_alpha,
    ws_pixel_pitch, ws_apply_nonmax, ws_invert,
    ws_zarr_path,
    mc_beta, mc_threshold,
    keep_watershed=True,
    axes=None,
    channel_index=None,
):
    import nifty
    import zarr
    import nifty.tools as nt
    from elf.segmentation.multicut import blockwise_multicut

    feature_names = read_feature_names(ilp_path)

    # --- Open all channels lazily ---
    with _ChannelStore(
        channel_specs, axes=axes, channel_index=channel_index
    ) as store:
        lazy_arrays = store.arrays
        boundary_channel = _find_boundary_channel(feature_names)
        if boundary_channel not in lazy_arrays:
            raise KeyError(
                f"Boundary channel {boundary_channel!r} not in provided channels. "
                f"Available: {list(lazy_arrays)}"
            )
        boundary_lazy = _Float32LazyArray(lazy_arrays[boundary_channel])
        vol_shape = tuple(boundary_lazy.shape)
        if len(vol_shape) != 3:
            raise ValueError(
                "Boundary probability volume must resolve to 3D zyx data. "
                f"Got shape: {vol_shape!r}"
            )
        print(f"Boundary probabilities shape: {vol_shape}")

        for name in feature_names:
            if name not in lazy_arrays:
                raise KeyError(
                    f"Channel {name!r} is required by the classifier but was "
                    f"not provided. Available: {list(lazy_arrays)}"
                )
            channel_shape = tuple(_Float32LazyArray(lazy_arrays[name]).shape)
            if channel_shape != vol_shape:
                raise ValueError(
                    f"Channel {name!r} resolves to shape {channel_shape}, "
                    f"but boundary probabilities resolve to {vol_shape}."
                )

        # --- Diagnostic: sample a small central patch to verify probability ---
        # convention.  elf's distance_transform_watershed expects high values
        # at boundaries; if the file instead stores P(background) the
        # superpixels will be inverted (seeds at boundary peaks).
        _diag_shape = tuple(min(s, 64) for s in vol_shape)
        _diag_start = tuple((s - d) // 2 for s, d in zip(vol_shape, _diag_shape))
        _diag_sl = tuple(slice(a, a + d) for a, d in zip(_diag_start, _diag_shape))
        _diag_patch = np.asarray(boundary_lazy[_diag_sl], dtype=np.float32)
        print(f"  Boundary probability sample (central {_diag_shape} patch):")
        print(f"    min={_diag_patch.min():.4f}  max={_diag_patch.max():.4f}  "
              f"mean={_diag_patch.mean():.4f}  fraction>{ws_threshold}={(_diag_patch > ws_threshold).mean():.3f}")
        if _diag_patch.mean() > 0.5:
            print(f"  WARNING: mean probability > 0.5 — if most of the volume is "
                  f"interior, this may indicate the file stores P(background) "
                  f"rather than P(boundary).  Consider passing --ws-invert or "
                  f"re-exporting the boundary channel.")
        del _diag_patch

        # Apply probability inversion for the watershed only (not for features).
        ws_input = _InvertedLazyArray(boundary_lazy) if ws_invert else boundary_lazy

        # --- Blockwise watershed: reuse existing zarr or compute fresh ---
        ws_zarr_arr, n_superpixels = _open_or_compute_watershed_zarr(
            ws_zarr_path=ws_zarr_path,
            boundary_lazy=ws_input,
            vol_shape=vol_shape,
            block_shape=block_shape,
            halo=halo,
            ws_method=ws_method,
            ws_threshold=ws_threshold,
            ws_sigma=ws_sigma,
            ws_min_size=ws_min_size,
            ws_alpha=ws_alpha,
            ws_pixel_pitch=ws_pixel_pitch,
            ws_apply_nonmax=ws_apply_nonmax,
            n_threads=n_threads,
        )

        blocking = nt.blocking([0, 0, 0], list(vol_shape), list(block_shape))
        n_blocks = blocking.numberOfBlocks
        sp_str = f": {n_superpixels} superpixels across {n_blocks} blocks." if n_superpixels else "."
        print(f"Watershed complete{sp_str}")

        # --- Blockwise feature computation ---
        # Accumulate edge arrays as numpy per block rather than building a Python
        # dict of tuple keys.  The dict approach costs ~300 bytes per edge in
        # Python object overhead; for large volumes (tens of millions of edges)
        # this easily exhausts RAM.  Numpy arrays use ~20 bytes per edge.
        print("Computing ilastikrag features blockwise …")

        # ilastikrag's StandardEdgeAccumulator has a bug where it checks
        # `if histogram_range:` (always True for a 2-element list) instead of
        # `if histogram_range[0] == histogram_range[1]:`.  This makes the
        # "All edge pixels are identical" warning fire for every block that
        # uses quantile features, producing thousands of spurious log lines.
        # Silence it here since it is not actionable.
        logging.getLogger(
            "ilastikrag.accumulators.standard.standard_edge_accumulator"
        ).setLevel(logging.ERROR)

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

            ws_block = ws_zarr_arr[outer_bb]  # Superpixels are already 1-indexed.
            channel_block = {
                name: _Float32LazyArray(lazy_arrays[name], normalize=(name!="Raw Data"))[outer_bb]
                for name in feature_names
            }

            features, edge_ids = compute_ilastikrag_features(
                ws_block, channel_block, feature_names
            )
            if features is None or edge_ids is None:
                # Block has < 2 superpixel labels → no edges; skip.
                continue

            probs = rf.predict_proba(features)[:, split_col]
            costs = compute_edge_costs(edge_ids, probs, beta=mc_beta, threshold=mc_threshold)

            # Canonicalize edge endpoints (sp1 ≤ sp2) and accumulate.
            all_edges_list.append(np.sort(edge_ids, axis=1))
            all_costs_list.append(costs)

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

        max_edge_node = int(all_edges.max())
        if n_superpixels:  # Node IDs should be consecutive 1…n_superpixels
            assert max_edge_node == n_superpixels, (
                f"Watershed claims to have {n_superpixels} superpixels, but "
                f"max node ID in edges is {max_edge_node}."
            )
        else:
            n_superpixels = max_edge_node
        key = all_edges[:, 0] * np.uint64(n_superpixels) + all_edges[:, 1]
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

        edge_uvs  = all_edges[keep].astype(np.uint64)
        edge_costs = all_costs_arr[keep]
        assert len(edge_uvs) > 0, f"Deduplication eradicated all edges? All: {all_edges}"
        del all_edges, all_costs_arr, keep

        print(f"  {len(edge_uvs)} unique edges after deduplication.")

        # ws_zarr_arr contains 1-indexed labels (1…n_superpixels).
        # nifty graph needs one extra node for a non-existent "background" superpixels (ID=0),
        # because this is what blockwise_multicut expects
        n_nodes = n_superpixels + 1
        # --- Build global nifty graph ---
        print(f"Building global graph ({n_nodes} nodes, {len(edge_uvs)} edges) …")
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(edge_uvs)
        del edge_uvs

        # --- Blockwise multicut ---
        # ws_zarr_arr supports __getitem__ with slice tuples, which is all
        # blockwise_mc_impl needs (it calls segmentation[bb] per block).
        print(f"Running blockwise multicut (block_shape={block_shape}, solver={internal_solver}) …")
        # nifty's C++ getBlockWithHalo binding requires List[int], not tuple.
        halo_list = list(halo) if halo is not None else None
        node_labels = blockwise_multicut(
            graph, edge_costs, ws_zarr_arr,
            internal_solver=internal_solver,
            block_shape=block_shape,
            n_threads=n_threads,
            halo=halo_list,
        )
        del edge_costs
        n_segments = len(np.unique(node_labels))
        print(f"Multicut complete: {n_segments} segments from {n_nodes} superpixels.")

        # --- Blockwise pixel projection → zarr ---
        print(f"Projecting labels and writing segmentation to {output_zarr_path} …")
        seg_out = _create_ome_zarr(
            output_zarr_path, shape=vol_shape, dtype="uint64",
            chunks=block_shape,
        )
        for block_id in range(n_blocks):
            block = blocking.getBlock(block_id)
            inner_bb = tuple(
                slice(s, e) for s, e in zip(block.begin, block.end)
            )
            # node_labels indices of course start at 0, while superpixel IDs start at 1.
            # edges should be referencing superpixel IDs though, so the extra node we
            # inserted earlier should have become the (unconnected) 0-node automatically.
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
