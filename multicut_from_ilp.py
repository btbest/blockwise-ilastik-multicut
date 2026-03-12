"""
multicut_from_ilp.py
Run elf's blockwise multicut on a large volume using an edge classifier
extracted from an ilastik .ilp project file.

The classifier is supplied as a pickled sklearn RandomForestClassifier
(produced by fit_classifier.py).  Feature computation at inference time uses
ilastikrag, which is the same library ilastik uses internally, so the feature
space is guaranteed to match what the classifier was trained on.

The volume is processed channel-by-channel and block-by-block to keep memory
usage bounded.  elf's blockwise multicut then solves the partitioning problem
by hierarchical graph decomposition without loading the full graph at once.

Usage
-----
    python multicut_from_ilp.py \
        --ilp          my_project.ilp \
        --rf           rf.pkl \
        --channels     "Membrane Probabilities 0:/path/to/boundary.h5:/data" \
                       "Raw Data 0:/path/to/raw.h5:/data" \
        --output       segmentation.h5 \
        --output-key   /seg \
        --block-shape  256 256 256 \
        --halo         32 32 32 \
        --beta         0.5 \
        --n-threads    8
"""

import argparse
import pickle
import sys

import h5py
import numpy as np

from ilp_reader import read_feature_names


# ---------------------------------------------------------------------------
# Channel / data loading helpers
# ---------------------------------------------------------------------------


def _load_channel(path: str, key: str | None) -> np.ndarray:
    """Load a 3-D (or 2-D) array from HDF5 or zarr.  key is required for HDF5."""
    if path.endswith(".h5") or path.endswith(".hdf5"):
        if key is None:
            raise ValueError(f"An HDF5 key (--channels …:key) is required for {path}")
        with h5py.File(path, "r") as f:
            return f[key][()]
    try:
        import zarr
        store = zarr.open(path, mode="r")
        return (store[key] if key else store)[()]
    except Exception as exc:
        raise ValueError(f"Cannot open {path}: {exc}") from exc


def _parse_channel_spec(spec: str) -> tuple[str, str, str | None]:
    """
    Parse a channel specification string of the form:
        "Channel Name:/path/to/file.h5:/dataset/key"
    or
        "Channel Name:/path/to/file.zarr"  (zarr root array)
    Returns (channel_name, file_path, hdf5_key_or_None).
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Channel spec must be 'ChannelName:/path/to/file[:key]', got: {spec!r}"
        )
    channel_name = parts[0]
    file_path = parts[1]
    hdf5_key = parts[2] if len(parts) >= 3 else None
    return channel_name, file_path, hdf5_key


# ---------------------------------------------------------------------------
# Feature computation (ilastikrag)
# ---------------------------------------------------------------------------


def compute_ilastikrag_features(
    superpixels: np.ndarray,
    channel_data: dict,
    feature_names: dict,
) -> np.ndarray:
    """
    Compute ilastikrag edge features for a superpixel image.

    Parameters
    ----------
    superpixels : uint32 ndarray
        Superpixel label image (same shape as the channel data arrays).
    channel_data : dict  {channel_name: ndarray}
        One array per channel.  Keys must match those in *feature_names*.
    feature_names : dict  {channel_name: [feature_name, ...]}
        Which ilastikrag features to compute per channel.
        Use ``read_feature_names(ilp_path)`` to obtain this.

    Returns
    -------
    np.ndarray, shape (N_edges, N_features)
        Feature matrix in the same column order that ilastik used during
        training: channels are concatenated in dict-iteration order, features
        within each channel in the order given by *feature_names[channel]*.
    edge_ids : np.ndarray, shape (N_edges, 2)
        (sp1, sp2) superpixel id pairs, one row per edge.
    """
    import ilastikrag

    rag = ilastikrag.Rag(superpixels)

    feature_dfs = []
    for channel_name, feat_names in feature_names.items():
        if channel_name not in channel_data:
            raise KeyError(
                f"Channel {channel_name!r} required by the classifier is not "
                f"provided via --channels.  Available: {list(channel_data)}"
            )
        data = np.asarray(channel_data[channel_name], dtype=np.float32)
        df = rag.compute_features(data, feat_names)
        # Drop the sp1/sp2 columns; prefix with channel name to match training
        feat_cols = [c for c in df.columns if c not in ("sp1", "sp2")]
        df = df[feat_cols].rename(
            columns={c: f"{channel_name} {c}" for c in feat_cols}
        )
        feature_dfs.append(df)

    edge_ids = rag.edge_ids  # (N_edges, 2) uint32
    import pandas as pd
    features_df = pd.concat(feature_dfs, axis=1)
    return features_df.values.astype(np.float32), edge_ids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_blockwise_multicut(
    ilp_path: str,
    rf_path: str,
    channel_specs: list[str],
    output_path: str,
    output_key: str = "/seg",
    beta: float = 0.5,
    block_shape: tuple[int, ...] = (256, 256, 256),
    halo: tuple[int, ...] = (32, 32, 32),
    internal_solver: str = "kernighan-lin",
    n_threads: int = 8,
    use_2dws: bool = False,
    ws_threshold: float = 0.5,
    ws_sigma: float = 2.0,
    verbose: bool = True,
):
    """
    Full blockwise multicut pipeline using an ilastik-trained sklearn RF.

    Parameters
    ----------
    ilp_path : str
        Path to the ilastik .ilp project file (used for feature names only).
    rf_path : str
        Path to the pickled sklearn RandomForestClassifier (from fit_classifier.py).
    channel_specs : list[str]
        Channel specifications, each of the form:
        "Channel Name:/path/to/file.h5:/key"
        The channel names must match those stored in the .ilp FeatureNames.
    output_path : str
        Path to the HDF5 file where the segmentation is saved.
    output_key : str
        HDF5 dataset key for the output segmentation.
    beta : float
        Boundary bias for cost computation (0.5 = unbiased).
    block_shape : tuple[int, ...]
        Block shape for the blockwise multicut solver.
    halo : tuple[int, ...]
        Halo added around each block when computing features to avoid
        boundary artefacts.  Does not affect the multicut block decomposition.
    internal_solver : str
        elf multicut solver: "kernighan-lin", "greedy-additive", etc.
    n_threads : int
        Number of threads for parallel processing.
    use_2dws : bool
        If True, use stacked 2D watersheds (for anisotropic data).
    ws_threshold : float
        Threshold applied to the boundary probability map before
        distance-transform watershed.
    ws_sigma : float
        Smoothing sigma for the distance-transform watershed.
    """
    import nifty
    import nifty.graph.rag as nrag

    from elf.segmentation.features import compute_rag, project_node_labels_to_pixels
    from elf.segmentation.multicut import blockwise_multicut, compute_edge_costs
    from elf.segmentation.watershed import distance_transform_watershed, stacked_watershed

    # --- Load classifier and feature names ---
    if verbose:
        print(f"Loading classifier from {rf_path} …")
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    feature_names = read_feature_names(ilp_path)
    if verbose:
        print("Feature names (from .ilp):")
        for ch, feats in feature_names.items():
            print(f"  {ch}: {feats}")

    # --- Load channel data ---
    channel_data = {}
    for spec in channel_specs:
        ch_name, fpath, fkey = _parse_channel_spec(spec)
        if verbose:
            print(f"Loading channel {ch_name!r} from {fpath}:{fkey} …")
        channel_data[ch_name] = _load_channel(fpath, fkey)

    # Determine the volume shape from first channel
    vol_shape = next(iter(channel_data.values())).shape
    if verbose:
        print(f"Volume shape: {vol_shape}")

    # --- Identify boundary channel (first channel in feature_names) ---
    # We use the first channel for watershed; typically this is the boundary map.
    boundary_channel = next(iter(feature_names))
    boundary_map = channel_data[boundary_channel].astype(np.float32)

    # --- Compute watershed ---
    if verbose:
        print("Computing watershed …")
    if use_2dws:
        watershed, _ = stacked_watershed(
            boundary_map,
            threshold=ws_threshold,
            sigma_seeds=ws_sigma,
            n_threads=n_threads,
        )
    else:
        watershed, _ = distance_transform_watershed(
            boundary_map,
            threshold=ws_threshold,
            sigma_seeds=ws_sigma,
            n_threads=n_threads,
        )
    watershed = watershed.astype(np.uint32)
    n_labels = int(watershed.max()) + 1
    if verbose:
        print(f"  {n_labels} superpixels")

    # --- Build RAG and compute features (full volume) ---
    # NOTE: For truly large volumes, replace this block with a chunked loop
    # that builds a global graph incrementally.  See README for details.
    if verbose:
        print("Building RAG and computing ilastikrag features …")
    features, edge_ids = compute_ilastikrag_features(
        watershed, channel_data, feature_names
    )
    if verbose:
        print(f"  {len(edge_ids)} edges, {features.shape[1]} features")

    # --- Predict edge probabilities ---
    if verbose:
        print("Predicting edge probabilities …")
    probs = rf.predict_proba(features)
    # rf.classes_ is sorted, so the column for the higher class (split=2) is index 1
    # when only two classes {1,2} are present.
    split_class_idx = np.argmax(rf.classes_)
    edge_probs = probs[:, split_class_idx].astype(np.float32)

    # --- Compute edge costs ---
    if verbose:
        print("Computing edge costs …")
    edge_sizes = None
    try:
        # ilastikrag edge_sizes (number of boundary pixels per edge)
        import ilastikrag
        rag_obj = ilastikrag.Rag(watershed)
        edge_sizes = rag_obj.edge_sizes.astype(np.float32)
    except Exception:
        pass  # edge sizes are optional; fall back to unweighted costs

    costs = compute_edge_costs(
        edge_probs,
        edge_sizes=edge_sizes,
        beta=beta,
        weighting_scheme="all" if edge_sizes is not None else None,
    )

    # --- Build nifty graph from edge list ---
    if verbose:
        print("Building nifty graph …")
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(edge_ids)

    # --- Blockwise multicut ---
    if verbose:
        print(
            f"Running blockwise multicut "
            f"(block_shape={block_shape}, solver={internal_solver!r}) …"
        )
    node_labels = blockwise_multicut(
        graph,
        costs,
        watershed,
        internal_solver=internal_solver,
        block_shape=block_shape,
        n_threads=n_threads,
        halo=halo,
    )

    # --- Project node labels back to pixels ---
    if verbose:
        print("Projecting node labels to pixels …")
    elf_rag = compute_rag(watershed, n_labels=n_labels, n_threads=n_threads)
    segmentation = project_node_labels_to_pixels(elf_rag, node_labels, n_threads=n_threads)
    if verbose:
        n_seg = len(np.unique(segmentation))
        print(f"  {n_seg} final segments")

    # --- Save output ---
    if verbose:
        print(f"Saving segmentation to {output_path}:{output_key} …")
    with h5py.File(output_path, "a") as f:
        if output_key in f:
            del f[output_key]
        f.create_dataset(output_key, data=segmentation, compression="gzip")

    if verbose:
        print("Done.")
    return segmentation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run elf's blockwise multicut using an ilastik-trained sklearn RF."
        )
    )
    parser.add_argument("--ilp", required=True, help="Path to ilastik .ilp project file")
    parser.add_argument("--rf", required=True, help="Path to pickled sklearn RF (from fit_classifier.py)")
    parser.add_argument(
        "--channels",
        nargs="+",
        required=True,
        metavar="NAME:FILE[:KEY]",
        help=(
            'Channel specifications, e.g. '
            '"Membrane Probabilities 0:/path/to/boundary.h5:/data" '
            '"Raw Data 0:/path/to/raw.h5:/data". '
            'Channel names must match those stored in the .ilp FeatureNames group.'
        ),
    )
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument("--output-key", default="/seg", help="HDF5 dataset key for output (default: /seg)")
    parser.add_argument("--beta", type=float, default=0.5, help="Boundary bias (default: 0.5)")
    parser.add_argument(
        "--block-shape",
        type=int,
        nargs="+",
        default=[256, 256, 256],
        metavar="N",
        help="Block shape for multicut solver, e.g. 256 256 256 (default)",
    )
    parser.add_argument(
        "--halo",
        type=int,
        nargs="+",
        default=[32, 32, 32],
        metavar="N",
        help="Halo for feature computation, e.g. 32 32 32 (default)",
    )
    parser.add_argument(
        "--solver",
        default="kernighan-lin",
        choices=["kernighan-lin", "greedy-additive", "greedy-fixation"],
        help="Internal multicut solver (default: kernighan-lin)",
    )
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--use-2dws", action="store_true", help="Use stacked 2D watersheds")
    parser.add_argument("--ws-threshold", type=float, default=0.5,
                        help="Watershed threshold (default: 0.5)")
    parser.add_argument("--ws-sigma", type=float, default=2.0,
                        help="Watershed seed smoothing sigma (default: 2.0)")
    args = parser.parse_args()

    run_blockwise_multicut(
        ilp_path=args.ilp,
        rf_path=args.rf,
        channel_specs=args.channels,
        output_path=args.output,
        output_key=args.output_key,
        beta=args.beta,
        block_shape=tuple(args.block_shape),
        halo=tuple(args.halo),
        internal_solver=args.solver,
        n_threads=args.n_threads,
        use_2dws=args.use_2dws,
        ws_threshold=args.ws_threshold,
        ws_sigma=args.ws_sigma,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
