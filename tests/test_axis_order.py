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


def test_input_axes_override_transposes_and_selects_probability_channel():
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


def test_multi_channel_probability_input_requires_probability_channel_index():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.zeros((2, 3, 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="pass --probability-channel-index"):
        _as_zyx_lazy_array(
            ArrayWithAttrs(data), input_axes="zyxc", source="probs.h5"
        )


def test_probability_channel_index_requires_explicit_or_metadata_axes():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.zeros((2, 3, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires axis metadata"):
        _as_zyx_lazy_array(
            ArrayWithAttrs(data),
            channel_index=0,
            source="raw.zarr",
        )


def test_probability_channel_index_requires_explicit_or_metadata_axes():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.zeros((2, 3, 4, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires axis metadata"):
        _as_zyx_lazy_array(
            ArrayWithAttrs(data),
            channel_index=0,
            source="raw.zarr",
        )


def test_probability_channel_index_bounds_are_checked():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.zeros((2, 3, 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="out of bounds"):
        _as_zyx_lazy_array(
            ArrayWithAttrs(data),
            input_axes="zyxc",
            channel_index=2,
            source="probs.h5",
        )


def test_probability_channel_index_is_ignored_without_channel_axis():
    from multicut_from_ilp import _as_zyx_lazy_array

    data = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    with pytest.warns(UserWarning, match="ignoring --probability-channel-index"):
        lazy = _as_zyx_lazy_array(
            ArrayWithAttrs(data),
            input_axes="zyx",
            channel_index=0,
            source="probs.h5",
        )
    assert lazy.source_n_channels == 1
    np.testing.assert_array_equal(lazy[:, :, :], data)


def test_channel_store_applies_probability_channel_index_only_to_boundary(tmp_path):
    h5py = pytest.importorskip("h5py")
    from multicut_from_ilp import _ChannelStore

    raw_data = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4, 1)
    prob_data = np.stack(
        [
            np.full((2, 3, 4), 11, dtype=np.uint8),
            np.full((2, 3, 4), 13, dtype=np.uint8),
        ],
        axis=-1,
    )
    raw_path = tmp_path / "raw.h5"
    prob_path = tmp_path / "probabilities.h5"
    with h5py.File(raw_path, "w") as f:
        f.create_dataset("data", data=raw_data)
    with h5py.File(prob_path, "w") as f:
        f.create_dataset("data", data=prob_data)

    specs = [
        f"Boundary:{prob_path}",
        f"Raw Data:{raw_path}",
    ]
    with _ChannelStore(
        specs,
        input_axes="zyxc",
        boundary_channel="Boundary",
        probability_channel_index=1,
    ) as store:
        np.testing.assert_array_equal(
            store.arrays["Boundary"][:, :, :],
            prob_data[..., 1],
        )
        np.testing.assert_array_equal(
            store.arrays["Raw Data"][:, :, :],
            raw_data[..., 0],
        )


def test_channel_store_rejects_multi_channel_raw_even_with_probability_index(tmp_path):
    h5py = pytest.importorskip("h5py")
    from multicut_from_ilp import _ChannelStore

    raw_path = tmp_path / "raw.h5"
    prob_path = tmp_path / "probabilities.h5"
    with h5py.File(raw_path, "w") as f:
        f.create_dataset("data", data=np.zeros((2, 3, 4, 2), dtype=np.uint8))
    with h5py.File(prob_path, "w") as f:
        f.create_dataset("data", data=np.zeros((2, 3, 4, 2), dtype=np.uint8))

    specs = [
        f"Boundary:{prob_path}",
        f"Raw Data:{raw_path}",
    ]
    with pytest.raises(
        ValueError,
        match="Raw/feature data must have no channel axis or exactly one channel",
    ):
        with _ChannelStore(
            specs,
            input_axes="zyxc",
            boundary_channel="Boundary",
            probability_channel_index=1,
        ):
            pass
