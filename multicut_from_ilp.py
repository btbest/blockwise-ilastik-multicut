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
        --ws-tmp /scratch/ws_tmp.dat \\
        --output-zarr segmentation.zarr \\
        --block-shape 256 256 256 \\
        --halo 32 32 32

Web / remote zarr (requires fsspec and aiohttp):

    python multicut_from_ilp.py \\
        --ilp my_project.ilp \\
        --rf rf.pkl \\
        --channels "Membrane Probabilities 0:https://webknossos.org/.../boundary.zarr/s0" \\
                   "Raw Data 0:https://webknossos.org/.../raw.zarr/s0" \\
        --lazy \\
        --ws-tmp /scratch/ws_tmp.dat \\
        --output-zarr segmentation.zarr \\
        --block-shape 256 256 256 \\
        --halo 32 32 32

Channel names must match those stored in the .ilp FeatureNames group.
Run: python -c "from ilp_reader import read_feature_names; print(read_feature_names('my.ilp'))"
to inspect channel names.
"""

import argparse
import os
import pickle
import sys
import warnings

import h5py
import numpy as np

from ilp_reader import read_feature_names


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
    use_2dws, ws_threshold, ws_sigma, verbose,
):
    import nifty
    from elf.segmentation.features import compute_rag, project_node_labels_to_pixels
    from elf.segmentation.multicut import blockwise_multicut, compute_edge_costs
    from elf.segmentation.watershed import distance_transform_watershed, stacked_watershed

    feature_names = read_feature_names(ilp_path)

    channel_data = {}
    for spec in channel_specs:
        ch_name, fpath, fkey = _parse_channel_spec(spec)
        if verbose:
            print(f"  Loading {ch_name!r} from {fpath} …")
        channel_data[ch_name] = _load_channel(fpath, fkey)

    vol_shape = next(iter(channel_data.values())).shape
    boundary_channel = _find_boundary_channel(feature_names)
    boundary_map = channel_data[boundary_channel].astype(np.float32)

    if verbose:
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

    if verbose:
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
    if verbose:
        print(f"  {n_labels} superpixels")

    if verbose:
        print("Predicting edge probabilities …")
    split_col = int(np.argmax(rf.classes_))
    edge_probs = rf.predict_proba(features)[:, split_col].astype(np.float32)

    costs = compute_edge_costs(edge_probs, beta=beta)

    if verbose:
        print("Building graph …")
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(edge_ids)

    if verbose:
        print(f"Running blockwise multicut (block_shape={block_shape}) …")
    node_labels = blockwise_multicut(
        graph, costs, watershed,
        internal_solver=internal_solver,
        block_shape=block_shape, n_threads=n_threads, halo=halo,
    )

    if verbose:
        print("Projecting labels …")
    elf_rag = compute_rag(watershed, n_labels=n_labels, n_threads=n_threads)
    segmentation = project_node_labels_to_pixels(elf_rag, node_labels, n_threads=n_threads)
    if verbose:
        print(f"  {len(np.unique(segmentation))} final segments")

    if verbose:
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
    use_2dws, ws_threshold, ws_sigma, ws_tmp_path, verbose,
):
    import nifty
    import nifty.tools as nt
    from elf.segmentation.multicut import blockwise_multicut, compute_edge_costs
    from elf.segmentation.watershed import blockwise_two_pass_watershed, stacked_watershed

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
        boundary_lazy = lazy_arrays[boundary_channel]
        vol_shape = tuple(boundary_lazy.shape)
        if verbose:
            print(f"Volume shape: {vol_shape}")

        # --- Blockwise watershed → numpy memmap ---
        if verbose:
            print(f"Computing blockwise watershed → {ws_tmp_path} …")
        ws_memmap = np.memmap(ws_tmp_path, dtype="uint64", mode="w+", shape=vol_shape)

        if use_2dws:
            # stacked_watershed operates slice-by-slice; supports pre-allocated output
            # For large z, still needs full input array — use lazy slicing
            if verbose:
                print("  Using stacked 2D watershed (lazy z-slices) …")
            _, max_id = stacked_watershed(
                boundary_lazy,
                threshold=ws_threshold, sigma_seeds=ws_sigma,
                n_threads=n_threads, output=ws_memmap,
            )
        else:
            _, max_id = blockwise_two_pass_watershed(
                boundary_lazy,
                block_shape=block_shape,
                halo=halo,
                threshold=ws_threshold,
                sigma_seeds=ws_sigma,
                n_threads=n_threads,
                output=ws_memmap,
                verbose=verbose,
            )
        # max_id is the 1-indexed maximum label from the vigra-based watershed.
        # We will shift to 0-indexed after feature computation (see below).
        if verbose:
            print(f"  {max_id} superpixels (max id = {max_id})")

        # --- Blockwise feature computation ---
        if verbose:
            print("Computing features blockwise …")

        split_col = int(np.argmax(rf.classes_))
        global_costs = {}  # {(sp1, sp2) canonical}: float32 cost

        blocking = nt.blocking([0, 0, 0], list(vol_shape), list(block_shape))
        n_blocks = blocking.numberOfBlocks
        for block_id in range(n_blocks):
            if verbose and (block_id % max(1, n_blocks // 10) == 0):
                print(f"  block {block_id}/{n_blocks} …")

            block = blocking.getBlockWithHalo(block_id, list(halo))
            outer_bb = tuple(
                slice(s, e)
                for s, e in zip(block.outerBlock.begin, block.outerBlock.end)
            )

            ws_block = np.array(ws_memmap[outer_bb])  # copy into RAM
            channel_block = {
                name: np.array(lazy_arrays[name][outer_bb])
                for name in feature_names
            }

            if ws_block.max() == 0:
                # empty block (fully masked), skip
                continue

            # Pass 1-indexed ws_block to ilastikrag (vigra treats 0 as background)
            features, edge_ids = compute_ilastikrag_features(
                ws_block, channel_block, feature_names
            )
            probs = rf.predict_proba(features)[:, split_col]
            costs = compute_edge_costs(probs.astype(np.float32), beta=beta)

            for (sp1, sp2), cost in zip(edge_ids.tolist(), costs.tolist()):
                key = (int(min(sp1, sp2)), int(max(sp1, sp2)))
                global_costs[key] = float(cost)

        if verbose:
            print(f"  Total edges: {len(global_costs)}")

        # --- Re-index to 0-based ---
        # vigra/elf watershed is 1-indexed (labels 1..max_id).  Keeping the phantom
        # node 0 in the nifty graph causes an isolated node that triggers an
        # off-by-one in blockwise_mc_impl (conda-forge python-elf 0.7.4).
        # Subtract 1 from every edge endpoint and from the memmap in-place.
        edge_uvs = np.array(list(global_costs.keys()), dtype=np.uint64) - 1
        edge_costs = np.array(list(global_costs.values()), dtype=np.float32)
        del global_costs  # free memory

        # Shift the memmap in-place (uint64; safe when all values > 0)
        ws_memmap -= 1
        n_nodes = int(max_id)  # 0-indexed max = max_id-1, so n_nodes = max_id

        # --- Build global nifty graph ---
        if verbose:
            print("Building global graph …")
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(edge_uvs)

        # --- Blockwise multicut ---
        # ws_memmap is a numpy.memmap (numpy subclass); blockwise_mc_impl
        # accesses it via segmentation[bb] slicing which works fine.
        if verbose:
            print(f"Running blockwise multicut (block_shape={block_shape}) …")
        node_labels = blockwise_multicut(
            graph, edge_costs, ws_memmap,
            internal_solver=internal_solver,
            block_shape=block_shape,
            n_threads=n_threads,
            halo=halo,
        )
        if verbose:
            print(f"  {len(np.unique(node_labels))} unique node labels")

        # --- Blockwise pixel projection → zarr ---
        if verbose:
            print(f"Writing segmentation to {output_zarr_path} …")
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
            ws_block = np.array(ws_memmap[inner_bb])
            seg_block = node_labels[ws_block]
            seg_out[inner_bb] = seg_block

    # --- Clean up memmap tempfile ---
    try:
        del ws_memmap
        os.remove(ws_tmp_path)
        if verbose:
            print(f"Removed tempfile {ws_tmp_path}")
    except Exception as e:
        warnings.warn(f"Could not remove tempfile {ws_tmp_path}: {e}")

    if verbose:
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
    ws_tmp_path: str = "ws_tmp.dat",
    verbose: bool = True,
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
    if verbose:
        print(f"Loading classifier from {rf_path} …")
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    feature_names = read_feature_names(ilp_path)
    if verbose:
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
            ws_tmp_path=ws_tmp_path, verbose=verbose,
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
            verbose=verbose,
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
        "--ws-tmp", default="ws_tmp.dat",
        help="Path for the watershed memmap tempfile (lazy mode, default: ws_tmp.dat)",
    )

    # Multicut / watershed parameters
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--block-shape", type=int, nargs="+", default=[256, 256, 256], metavar="N",
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
        block_shape=tuple(args.block_shape),
        halo=tuple(args.halo),
        internal_solver=args.solver,
        n_threads=args.n_threads,
        use_2dws=args.use_2dws,
        ws_threshold=args.ws_threshold,
        ws_sigma=args.ws_sigma,
        ws_tmp_path=args.ws_tmp,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
