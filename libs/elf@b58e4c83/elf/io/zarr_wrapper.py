import zarr
from numcodecs import Blosc, GZip, Zstd

SUPPORTED_CODECS = ("blosc", "gzip", "zstd")
"""Supported compression codecs
"""

_CODEC_MAP = {
    "blosc": Blosc(),
    "gzip": GZip(),
    "zstd": Zstd(),
}


def zarr_open(*args, **kwargs):
    """@private
    """
    return zarr.open(*args, **kwargs)
