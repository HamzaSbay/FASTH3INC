"""Remote GGUF v3 header parser with incremental range fetching.

Parses GGUF v3 key-value metadata and tensor descriptors without downloading
the full file, dynamically expanding range reads until the full header is parsed.
"""

from __future__ import annotations

import math
import struct
from typing import Any, NamedTuple, Optional

try:
    import gguf
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    _HAS_GGUF_PKG = True
except ImportError:
    _HAS_GGUF_PKG = False

try:
    from . import hf_range
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools import hf_range

# GGML block sizing: type_id -> (elements_per_block, bytes_per_block)
DEFAULT_GGML_BLOCK_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 40),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (1, 1),     # I8
    17: (1, 2),     # I16
    18: (1, 4),     # I32
    19: (1, 8),     # I64
    20: (1, 8),     # F64
    24: (1, 1),     # I8
    30: (1, 2),     # BF16
}

DEFAULT_GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "I8",
    17: "I16",
    18: "I32",
    19: "I64",
    20: "F64",
    30: "BF16",
}

# GGUF Value types (0..12) -> struct unpack format
VALUE_TYPE_FORMATS: dict[int, str] = {
    0: "B",  # UINT8
    1: "b",  # INT8
    2: "H",  # UINT16
    3: "h",  # INT16
    4: "I",  # UINT32
    5: "i",  # INT32
    6: "f",  # FLOAT32
    7: "?",  # BOOL
    10: "Q", # UINT64
    11: "q", # INT64
    12: "d", # FLOAT64
}


class TensorInfo(NamedTuple):
    """Metadata for a single tensor stored inside a GGUF file."""
    name: str
    dims: list[int]
    type: str
    offset: int
    nbytes: int
    orig_shape: Optional[list[int]] = None


class GgufHeader(NamedTuple):
    """Parsed GGUF header including key-value metadata and tensor table."""
    kv: dict[str, Any]
    tensors: list[TensorInfo]
    data_offset: int


def _get_block_spec(type_id: int) -> tuple[int, int]:
    if _HAS_GGUF_PKG:
        try:
            qtype = GGMLQuantizationType(type_id)
            if qtype in GGML_QUANT_SIZES:
                return GGML_QUANT_SIZES[qtype]
        except (ValueError, KeyError):
            pass
    return DEFAULT_GGML_BLOCK_SIZES.get(type_id, (1, 1))


def _get_type_name(type_id: int) -> str:
    if _HAS_GGUF_PKG:
        try:
            return GGMLQuantizationType(type_id).name
        except ValueError:
            pass
    return DEFAULT_GGML_TYPE_NAMES.get(type_id, f"TYPE_{type_id}")


class _GrowingRangeReader:
    """Buffer that dynamically fetches slices of a remote URL as needed."""

    def __init__(self, url: str, initial_chunk: int = 64 * 1024, chunk_size: int = 64 * 1024):
        self.url = url
        self.chunk_size = chunk_size
        self.buf = bytearray()
        self.cursor = 0
        if initial_chunk > 0:
            initial_data = hf_range.read(self.url, 0, initial_chunk)
            self.buf.extend(initial_data)

    def ensure(self, n_bytes: int) -> None:
        needed = (self.cursor + n_bytes) - len(self.buf)
        if needed > 0:
            fetch_size = max(self.chunk_size, needed)
            start_off = len(self.buf)
            chunk = hf_range.read(self.url, start_off, start_off + fetch_size)
            if not chunk:
                raise EOFError(
                    f"Unexpected EOF while reading {self.url} at byte {start_off} (requested {n_bytes} bytes)"
                )
            self.buf.extend(chunk)

    def rd(self, fmt: str) -> Any:
        size = struct.calcsize("<" + fmt)
        self.ensure(size)
        val = struct.unpack_from("<" + fmt, self.buf, self.cursor)[0]
        self.cursor += size
        return val

    def rs(self) -> str:
        n = self.rd("Q")
        self.ensure(n)
        s = self.buf[self.cursor : self.cursor + n].decode("utf-8", "replace")
        self.cursor += n
        return s

    def rv(self, val_type: int) -> Any:
        if val_type == 8:  # STRING
            return self.rs()
        if val_type == 9:  # ARRAY
            elem_type = self.rd("I")
            count = self.rd("Q")
            return [self.rv(elem_type) for _ in range(count)]
        fmt = VALUE_TYPE_FORMATS.get(val_type)
        if fmt is None:
            raise ValueError(f"Unknown GGUF value type {val_type} at cursor {self.cursor}")
        return self.rd(fmt)


def header(url: str, initial_chunk: int = 64 * 1024, chunk_size: int = 64 * 1024) -> GgufHeader:
    """Incrementally read and parse a GGUF v3 header from a URL or local file.

    Args:
        url: URL (http/https/file) or local file path to the GGUF model.
        initial_chunk: Initial bytes to fetch in first range request.
        chunk_size: Incremental fetch size when header extends past current buffer.

    Returns:
        GgufHeader containing parsed kv pairs, tensor list, and aligned data_offset.
    """
    reader = _GrowingRangeReader(url, initial_chunk=initial_chunk, chunk_size=chunk_size)

    # Magic identifier
    reader.ensure(4)
    magic = reader.buf[reader.cursor : reader.cursor + 4]
    if bytes(magic) != b"GGUF":
        raise ValueError(f"Invalid GGUF magic bytes {magic!r} in {url}")
    reader.cursor += 4

    # Version and counts
    version = reader.rd("I")
    if version not in (2, 3):
        raise ValueError(f"Unsupported GGUF version {version} (expected 2 or 3)")
    tensor_count = reader.rd("Q")
    kv_count = reader.rd("Q")

    # Key-value metadata
    kv: dict[str, Any] = {}
    for _ in range(kv_count):
        k = reader.rs()
        v_type = reader.rd("I")
        v = reader.rv(v_type)
        kv[k] = v

    # Tensor descriptors
    tensors: list[TensorInfo] = []
    for _ in range(tensor_count):
        t_name = reader.rs()
        n_dims = reader.rd("I")
        dims = [reader.rd("Q") for _ in range(n_dims)]
        t_type = reader.rd("I")
        offset = reader.rd("Q")

        nelem = math.prod(dims) if dims else 1
        be, bb = _get_block_spec(t_type)
        nbytes = (nelem // be) * bb

        orig_shape = kv.get(f"comfy.gguf.orig_shape.{t_name}")
        type_name = _get_type_name(t_type)

        tensors.append(
            TensorInfo(
                name=t_name,
                dims=dims,
                type=type_name,
                offset=offset,
                nbytes=nbytes,
                orig_shape=orig_shape,
            )
        )

    # Compute data offset aligned to general.alignment (default 32)
    alignment = int(kv.get("general.alignment", 32))
    header_end = reader.cursor
    data_offset = (header_end + alignment - 1) // alignment * alignment

    return GgufHeader(kv=kv, tensors=tensors, data_offset=data_offset)
