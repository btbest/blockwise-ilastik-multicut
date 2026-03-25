"""Shared CLI validation and parameter resolution for blimp commands.

Functions here resolve watershed parameters from a combination of CLI flags
and (optionally) an ilastik .ilp project file.
"""

from pathlib import Path

_H5_EXTENSIONS = (".h5", ".hdf5", ".hdf", ".ilp")


def data_stem(path: str) -> str:
    """Return a human-friendly stem for naming output files.

    For compound HDF5 paths like ``/data/block1_raw.h5/exported_data``
    returns ``"block1_raw"`` (the file stem, not the internal dataset).
    For plain paths like ``raw.zarr`` returns ``"raw"``.

    Uses ``PureWindowsPath`` so that Windows backslash separators are
    handled correctly even when running on Linux.
    """
    from pathlib import PureWindowsPath

    lower = path.lower()
    for ext in _H5_EXTENSIONS:
        idx = lower.find(ext)
        if idx == -1:
            continue
        end = idx + len(ext)
        if end < len(path) and path[end] not in ("/", "\\"):
            continue
        return PureWindowsPath(path[:end]).stem
    return PureWindowsPath(path).stem

import warnings

# Default values used when neither CLI flags nor .ilp provide a value.
_WS_DEFAULTS = {
    "threshold": 0.5,
    "sigma": 3.0,
    "min_size": 100,
    "alpha": 0.9,
}


def resolve_watershed_params(args, *, ilp_path=None):
    """Resolve final watershed parameter values.

    Priority: CLI flag > .ilp value > built-in default.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (must contain ws_threshold, ws_sigma,
        ws_min_size, ws_alpha, ws_invert, ws_method).
    ilp_path : str | None
        Path to .ilp file.  When given, DT Watershed parameters are read
        from it as fallback values.

    Returns
    -------
    dict with keys: ws_threshold, ws_sigma, ws_min_size, ws_alpha,
        ws_invert, ws_pixel_pitch, ws_apply_nonmax, ws_method.
    """
    if ilp_path is not None:
        from ilp_reader import read_wsdt_params
        ilp_ws = read_wsdt_params(ilp_path)
    else:
        ilp_ws = None

    def _pick(cli_val, ilp_key, default_key):
        if cli_val is not None:
            return cli_val
        if ilp_ws is not None:
            return ilp_ws[ilp_key]
        return _WS_DEFAULTS[default_key]

    ws_threshold = _pick(args.ws_threshold, "threshold", "threshold")
    ws_sigma     = _pick(args.ws_sigma,     "sigma",     "sigma")
    ws_min_size  = _pick(args.ws_min_size,  "min_size",  "min_size")
    ws_alpha     = _pick(args.ws_alpha,     "alpha",     "alpha")

    ws_invert       = args.ws_invert
    ws_pixel_pitch  = ilp_ws["pixel_pitch"] if ilp_ws is not None else None
    ws_apply_nonmax = False  # ApplyNonmaxSuppression; not serialised in .ilp

    # --- Watershed method selection ---
    if args.ws_method is not None:
        ws_method = args.ws_method
    elif ilp_ws is not None and ilp_ws["blockwise"]:
        ws_method = "ilastik"
    elif ilp_ws is not None and not ilp_ws["blockwise"]:
        warnings.warn(
            "The .ilp was saved with BlockwiseWatershed=False (an old project). "
            "ilastik ran the watershed on the full training crop at once, which we "
            "cannot replicate blockwise.  Falling back to 'two-pass'.  Pass "
            "--ws-method explicitly to suppress this warning.",
            stacklevel=2,
        )
        ws_method = "two-pass"
    else:
        # No .ilp provided and no --ws-method flag: sensible default.
        ws_method = "ilastik"

    return {
        "ws_threshold":    ws_threshold,
        "ws_sigma":        ws_sigma,
        "ws_min_size":     ws_min_size,
        "ws_alpha":        ws_alpha,
        "ws_invert":       ws_invert,
        "ws_pixel_pitch":  ws_pixel_pitch,
        "ws_apply_nonmax": ws_apply_nonmax,
        "ws_method":       ws_method,
    }


def resolve_lane_pairs(args, *, ilp_path):
    """Resolve the list of (raw, probabilities) pairs to process.

    When ``--raw`` and ``--probabilities`` are given on the command line, a
    single pair is returned.  Otherwise, all Raw Data / Probabilities lane
    pairs are read from the .ilp project's Input Data group.

    Also resolves ``--output-dir``: when not supplied, defaults to a
    ``blimp-output/`` directory next to the .ilp file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (may have ``raw``, ``probabilities``,
        ``output_dir`` set to ``None``).
    ilp_path : str
        Path to the .ilp project file.

    Returns
    -------
    pairs : list[dict]
        Each dict has keys ``"raw"`` and ``"probabilities"`` with
        absolute path strings.
    output_dir : pathlib.Path
        Resolved output directory.
    """
    import sys
    from pathlib import Path

    # --- Output directory ---
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(ilp_path).resolve().parent / "blimp-output"

    # --- Lane pairs ---
    has_raw = args.raw is not None
    has_prob = args.probabilities is not None

    if has_raw != has_prob:
        print(
            "error: --raw and --probabilities must both be given or both omitted.",
            file=sys.stderr,
        )
        sys.exit(2)

    if has_raw:
        return [{"raw": args.raw, "probabilities": args.probabilities}], output_dir

    # Read from the .ilp Input Data group
    from ilp_reader import read_input_data_paths

    pairs = read_input_data_paths(ilp_path)
    if not pairs:
        print(
            "error: no Raw Data + Probabilities lane pairs found in the .ilp "
            "Input Data group.  Provide --raw and --probabilities explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)

    return pairs, output_dir


def validate_watershed_params(args):
    """Validate that watershed-only mode has enough parameters.

    When --ilp is not given, the user must supply all four core watershed
    parameters explicitly (--ws-threshold, --ws-sigma, --ws-min-size,
    --ws-alpha), since there is no .ilp to fall back on.

    Raises SystemExit (via argparse-style error) on validation failure.
    """
    if getattr(args, "ilp", None):
        return  # .ilp provides defaults; nothing to validate

    # Without --ilp, --raw and --probabilities are required too
    missing_data = []
    if args.raw is None:
        missing_data.append("--raw")
    if args.probabilities is None:
        missing_data.append("--probabilities")
    if missing_data:
        import sys
        print(
            f"error: without --ilp, the following parameters must be "
            f"given explicitly: {', '.join(missing_data)}",
            file=sys.stderr,
        )
        sys.exit(2)

    cli_params = {
        "--ws-threshold": args.ws_threshold,
        "--ws-sigma":     args.ws_sigma,
        "--ws-min-size":  args.ws_min_size,
        "--ws-alpha":     args.ws_alpha,
    }
    missing = [name for name, val in cli_params.items() if val is None]
    if missing:
        import sys
        print(
            f"error: without --ilp, the following watershed parameters must be "
            f"given explicitly: {', '.join(missing)}\n"
            f"(These would normally be read from the .ilp project file.)",
            file=sys.stderr,
        )
        sys.exit(2)
