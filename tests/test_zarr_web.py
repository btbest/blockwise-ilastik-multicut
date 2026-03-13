"""
tests/test_zarr_web.py
======================
Tests for loading raw data from remote zarr URLs (web / S3 / WebKnossos).

The example array used here is a small public zarr v2 dataset hosted on S3:
  https://s3.amazonaws.com/btbest-public/img.zarr/s0
  (array metadata: …/s0/.zarray)

Tests that actually fetch data over the network are skipped automatically
when no network connection is detected, so they are safe to run in offline
CI environments.
"""

import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXAMPLE_URL = "https://s3.amazonaws.com/btbest-public/img.zarr/s0"


def _network_available() -> bool:
    """Return True if a basic TCP connection to a well-known host succeeds."""
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except OSError:
        return False


network = pytest.mark.skipif(
    not _network_available(), reason="No network access – skipping remote zarr tests"
)


# ---------------------------------------------------------------------------
# _parse_channel_spec – URL handling (no network required)
# ---------------------------------------------------------------------------


def test_parse_channel_spec_https_url():
    """URL specs must not be split on the colon inside '://'."""
    from multicut_from_ilp import _parse_channel_spec

    name, path, key = _parse_channel_spec(f"Raw Data:{EXAMPLE_URL}")
    assert name == "Raw Data"
    assert path == EXAMPLE_URL
    assert key is None


def test_parse_channel_spec_https_url_preserves_full_url():
    """Any https URL with port or query params must be kept verbatim."""
    from multicut_from_ilp import _parse_channel_spec

    url = "https://example.com:8080/store.zarr/s0"
    name, path, key = _parse_channel_spec(f"Probs:{url}")
    assert path == url
    assert key is None


def test_parse_channel_spec_s3_url():
    from multicut_from_ilp import _parse_channel_spec

    url = "s3://my-bucket/data.zarr"
    name, path, key = _parse_channel_spec(f"Boundary:{url}")
    assert name == "Boundary"
    assert path == url
    assert key is None


def test_parse_channel_spec_local_zarr_still_works():
    """Existing local zarr specs must be unaffected by the URL change."""
    from multicut_from_ilp import _parse_channel_spec

    name, path, key = _parse_channel_spec("Probabilities:/data/probs.zarr")
    assert name == "Probabilities"
    assert path == "/data/probs.zarr"
    assert key is None


def test_parse_channel_spec_local_h5_still_works():
    """HDF5 specs without keys work; everything after the first colon is the path."""
    from multicut_from_ilp import _parse_channel_spec

    name, path, key = _parse_channel_spec("My Channel:/data/vol.h5")
    assert name == "My Channel"
    assert path == "/data/vol.h5"
    assert key is None


# ---------------------------------------------------------------------------
# _open_channel_lazy / _load_channel – real remote array (network required)
# ---------------------------------------------------------------------------


@network
def test_open_channel_lazy_url_returns_array():
    """Opening a remote zarr URL must return a lazy array-like with no file handle."""
    pytest.importorskip("fsspec")
    from multicut_from_ilp import _open_channel_lazy

    arr, fh = _open_channel_lazy(EXAMPLE_URL, key=None)
    assert fh is None, "zarr manages its own handles; no external handle expected"
    assert hasattr(arr, "shape"), "returned object should expose a .shape attribute"
    assert arr.ndim == 5, "example array is tczyx (5-D, singleton t and z)"


@network
def test_open_channel_lazy_url_dtype():
    pytest.importorskip("fsspec")
    from multicut_from_ilp import _open_channel_lazy

    arr, _ = _open_channel_lazy(EXAMPLE_URL, key=None)
    # Verify dtype is numeric and not object
    import numpy as np

    assert np.issubdtype(arr.dtype, np.number), f"unexpected dtype: {arr.dtype}"


@network
def test_open_channel_lazy_url_slice_readable():
    """A small slice of the remote array must be loadable without error."""
    pytest.importorskip("fsspec")
    from multicut_from_ilp import _open_channel_lazy
    import numpy as np

    arr, _ = _open_channel_lazy(EXAMPLE_URL, key=None)
    # Read a tiny corner – tczyx, singleton t and z, 3 channels
    chunk = np.array(arr[0:1, 0:2, 0:1, 0:2, 0:2])
    assert chunk.shape == (1, 2, 1, 2, 2)


@network
def test_load_channel_url_returns_full_array():
    """_load_channel must be able to fully materialise a remote zarr array."""
    pytest.importorskip("fsspec")
    from multicut_from_ilp import _load_channel
    import numpy as np

    data = _load_channel(EXAMPLE_URL, key=None)
    assert isinstance(data, np.ndarray)
    assert data.ndim == 5, "example array is tczyx (5-D)"
    assert data.size > 0
