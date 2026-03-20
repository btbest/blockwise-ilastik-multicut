"""Shared CLI validation and parameter resolution for blimp commands.

Functions here resolve watershed parameters from a combination of CLI flags
and (optionally) an ilastik .ilp project file.
"""

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


def validate_watershed_params(args):
    """Validate that watershed-only mode has enough parameters.

    When --ilp is not given, the user must supply all four core watershed
    parameters explicitly (--ws-threshold, --ws-sigma, --ws-min-size,
    --ws-alpha), since there is no .ilp to fall back on.

    Raises SystemExit (via argparse-style error) on validation failure.
    """
    if getattr(args, "ilp", None):
        return  # .ilp provides defaults; nothing to validate

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
