import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


class ArrayWithAttrs:
    def __init__(self, data, attrs=None):
        self._data = np.asarray(data)
        self.attrs = attrs or {}
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.dtype = self._data.dtype

    def __getitem__(self, key):
        return self._data[key]


def test_input_axes_override_transposes_and_selects_channel():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.arange(2 * 5 * 4 * 3).reshape(2, 5, 4, 3)
    lazy = _as_zyx_lazy_array(
        ArrayWithAttrs(data),
        input_axes="cxyz",
        channel_index=1,
        source="probs.zarr",
    )


    assert lazy.shape == (3, 4, 5)
    expected = np.transpose(data[1], (2, 1, 0))
    np.testing.assert_array_equal(lazy[:, :, :], expected)
    np.testing.assert_array_equal(lazy[1:3, 1:4, 2:5], expected[1:3, 1:4, 2:5])


def test_axistags_metadata_is_used_when_axes_override_is_absent():
    vigra = pytest.importorskip("vigra")
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.arange(5 * 3 * 4 * 1).reshape(5, 3, 4, 1)
    attrs = {"axistags": vigra.defaultAxistags("xzyc").toJSON()}

    lazy = _as_zyx_lazy_array(ArrayWithAttrs(data, attrs=attrs), source="raw.h5")

    assert lazy.shape == (3, 4, 5)
    expected = np.transpose(data[..., 0], (1, 2, 0))
    np.testing.assert_array_equal(lazy[:, :, :], expected)


def test_input_axes_override_takes_precedence_over_axistags():
    vigra = pytest.importorskip("vigra")
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.arange(5 * 3 * 4).reshape(5, 3, 4)
    attrs = {"axistags": vigra.defaultAxistags("zyx").toJSON()}

    lazy = _as_zyx_lazy_array(
        ArrayWithAttrs(data, attrs=attrs),
        input_axes="xzy",
        source="raw.h5",
    )

    assert lazy.shape == (3, 4, 5)
    expected = np.transpose(data, (1, 2, 0))
    np.testing.assert_array_equal(lazy[:, :, :], expected)


def test_multi_channel_input_requires_channel_index():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.zeros((2, 3, 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="pass --channel-index"):
        _as_zyx_lazy_array(
            ArrayWithAttrs(data), input_axes="zyxc", source="probs.h5"
        )


def test_channel_index_requires_explicit_or_metadata_axes():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.zeros((2, 3, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires axis metadata"):
        _as_zyx_lazy_array(
            ArrayWithAttrs(data),
            channel_index=0,
            source="raw.zarr",
        )


def test_channel_index_bounds_are_checked():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.zeros((2, 3, 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="out of bounds"):
        _as_zyx_lazy_array(
            ArrayWithAttrs(data),
            input_axes="zyxc",
            channel_index=2,
            source="probs.h5",
        )


def test_channel_index_alias_with_underscore_is_accepted():
    import argparse
    from _cli_params import add_watershed_args

    parser = argparse.ArgumentParser()
    add_watershed_args(parser)
    args = parser.parse_args(["--channel_index", "1"])

    assert args.channel_index == 1
