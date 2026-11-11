"""Unit and regression tests for remote GGUF header parser and census tool."""

from __future__ import annotations

import os
import re
import struct
import tempfile
from unittest.mock import MagicMock, patch

import pytest

requests = pytest.importorskip("requests")
gguf = pytest.importorskip("gguf")

import tools.census as census
import tools.gguf_remote as gguf_remote
import tools.hf_range as hf_range


DIT_URL = "https://huggingface.co/realrebelai/FastH3-V2_GGUFs/resolve/main/FastH3-8-Step-V2-Q4_K_M.gguf"
TE_URL = "https://huggingface.co/realrebelai/FastH3-V2_GGUFs/resolve/main/Text%20Encoders/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"


# ==============================================================================
# Step 1: Network-marked tests against Hugging Face endpoints
# ==============================================================================


@pytest.mark.network
def test_header_parses_q4km():
    """Verify DiT GGUF v3 header parsing against realrebelai Q4_K_M model.

    Step 1 assertions:
      1. len(tensors) == 585
      2. kv["general.architecture"] == "wan"
      3. sum of blocks.*.adaln_proj.linear.weight nbytes == 50 * 146_313_216 (±0 bytes)
      4. blocks.0 non-adaln bytes ≈ 238.4 MB (238,480,384 bytes)
    """
    hdr = gguf_remote.header(DIT_URL)
    assert len(hdr.tensors) == 585
    assert hdr.kv.get("general.architecture") == "wan"

    # Exact AdaLN weights check: 50 blocks * 146,313,216 bytes = 7,315,660,800 bytes
    adaln_w_tensors = [
        t for t in hdr.tensors if re.match(r"^blocks\.\d+\.adaln_proj\.linear\.weight$", t.name)
    ]
    assert len(adaln_w_tensors) == 50
    adaln_w_bytes = sum(t.nbytes for t in adaln_w_tensors)
    assert adaln_w_bytes == 50 * 146_313_216
    assert adaln_w_bytes == 7_315_660_800

    # Block 0 non-AdaLN bytes (Trunk per block)
    b0_non_adaln = [
        t for t in hdr.tensors if t.name.startswith("blocks.0.") and "adaln" not in t.name
    ]
    assert len(b0_non_adaln) == 9
    b0_non_adaln_bytes = sum(t.nbytes for t in b0_non_adaln)
    assert b0_non_adaln_bytes == 238_480_384  # 238.48 MB (≈ 238.4 MB)
    assert abs(b0_non_adaln_bytes / 1e6 - 238.48) < 0.1

    # Verify data offset alignment
    assert hdr.data_offset % 32 == 0
    assert hdr.data_offset == 42496


@pytest.mark.network
def test_te_header_parses_q4km():
    """Verify Qwen3-VL Text Encoder header parsing against Hugging Face endpoint."""
    hdr = gguf_remote.header(TE_URL)
    assert len(hdr.tensors) == 902
    assert hdr.kv.get("general.architecture") == "qwen3vl"

    te_stats = census.analyze_te(hdr, TE_URL)
    # Layers < 50: 50 layers * 11 tensors = 550 tensors
    assert te_stats["layers_lt50"]["layer_count"] == 50
    assert te_stats["layers_lt50"]["tensor_count"] == 550
    assert te_stats["layers_lt50"]["bytes"] == 13_783_091_200

    # Layers >= 50: truncated in this distribution
    assert te_stats["layers_ge50"]["tensor_count"] == 0
    assert te_stats["layers_ge50"]["bytes"] == 0

    # Total TE size
    assert te_stats["total_bytes"] == 14_576_902_848


# ==============================================================================
# Unit tests: Local file reading & edge cases in hf_range
# ==============================================================================


def test_hf_range_local_file():
    """Test hf_range.read with local file paths and boundary conditions."""
    payload = b"0123456789ABCDEF" * 1024  # 16 KB

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(payload)
        temp_path = tf.name

    try:
        # Standard slice
        assert hf_range.read(temp_path, 0, 16) == payload[:16]
        assert hf_range.read(temp_path, 100, 200) == payload[100:200]
        # Empty slice
        assert hf_range.read(temp_path, 10, 10) == b""
        # file:// URL scheme
        file_url = "file:///" + os.path.abspath(temp_path).replace("\\", "/")
        assert hf_range.read(file_url, 0, 32) == payload[:32]
        # Read past file size clamped to file end
        past_end = len(payload) + 100
        assert hf_range.read(temp_path, len(payload) - 5, past_end) == payload[-5:]
        assert hf_range.read(temp_path, len(payload), past_end) == b""

        # Errors
        with pytest.raises(ValueError):
            hf_range.read(temp_path, 50, 40)
        with pytest.raises(ValueError):
            hf_range.read(temp_path, -1, 10)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_hf_range_mock_network_resumable():
    """Test hf_range.read HTTP handling with transient errors and retries."""
    fake_data = b"X" * 1024

    mock_resp_fail = MagicMock()
    mock_resp_fail.status_code = 503
    mock_resp_fail.__enter__.return_value = mock_resp_fail
    mock_resp_fail.raise_for_status.side_effect = Exception("503 Service Unavailable")

    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 206
    mock_resp_ok.__enter__.return_value = mock_resp_ok
    mock_resp_ok.iter_content.return_value = [fake_data[:512], fake_data[512:]]

    # First call returns 503, second returns 206
    with patch("requests.get", side_effect=[mock_resp_fail, mock_resp_ok]) as mock_get:
        data = hf_range.read(
            "https://example.com/model.gguf",
            start=0,
            end=1024,
            max_retries=2,
            backoff_factor=0.01,
        )
        assert data == fake_data
        assert mock_get.call_count == 2


# ==============================================================================
# Unit tests: Synthetic GGUF v3 creation and incremental parsing
# ==============================================================================


def _build_synthetic_gguf(alignment: int = 32) -> bytes:
    """Construct a minimal valid GGUF v3 byte stream with known KV and tensors."""
    out = bytearray()
    # Magic + Version + tensor_count + kv_count
    out.extend(b"GGUF")
    out.extend(struct.pack("<IQQ", 3, 2, 2))  # ver=3, 2 tensors, 2 kv

    # Helper serializers
    def add_str(s: str):
        b = s.encode("utf-8")
        out.extend(struct.pack("<Q", len(b)))
        out.extend(b)

    # KV 1: general.architecture = "wan" (type 8: string)
    add_str("general.architecture")
    out.extend(struct.pack("<I", 8))
    add_str("wan")

    # KV 2: general.alignment = alignment (type 4: uint32)
    add_str("general.alignment")
    out.extend(struct.pack("<II", 4, alignment))

    # Tensor 1: "blocks.0.attn.qkv_proj.weight", 2D [256, 128], type 0 (F32), offset 0
    add_str("blocks.0.attn.qkv_proj.weight")
    out.extend(struct.pack("<I", 2))  # ndims = 2
    out.extend(struct.pack("<QQ", 256, 128))
    out.extend(struct.pack("<IQ", 0, 0))  # type=0 (F32), offset=0

    # Tensor 2: "blocks.0.adaln_proj.linear.weight", 1D [1024], type 12 (Q4_K), offset 131072
    add_str("blocks.0.adaln_proj.linear.weight")
    out.extend(struct.pack("<I", 1))  # ndims = 1
    out.extend(struct.pack("<Q", 1024))
    out.extend(struct.pack("<IQ", 12, 131072))  # type=12 (Q4_K), offset=131072

    return bytes(out)


def test_gguf_remote_synthetic():
    """Verify that gguf_remote incrementally parses a synthetic GGUF v3 stream."""
    raw = _build_synthetic_gguf(alignment=64)

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(raw)
        temp_path = tf.name

    try:
        # Use a small chunk_size to test incremental growing logic
        hdr = gguf_remote.header(temp_path, initial_chunk=32, chunk_size=32)
        assert hdr.kv["general.architecture"] == "wan"
        assert hdr.kv["general.alignment"] == 64
        assert len(hdr.tensors) == 2

        t1 = hdr.tensors[0]
        assert t1.name == "blocks.0.attn.qkv_proj.weight"
        assert t1.dims == [256, 128]
        assert t1.type == "F32"
        assert t1.nbytes == 256 * 128 * 4  # 131072 bytes

        t2 = hdr.tensors[1]
        assert t2.name == "blocks.0.adaln_proj.linear.weight"
        assert t2.dims == [1024]
        assert t2.type == "Q4_K"
        # Q4_K is 144 bytes per 256 elements -> 1024 // 256 * 144 = 576 bytes
        assert t2.nbytes == 576

        # Alignment check
        assert hdr.data_offset % 64 == 0
        assert hdr.data_offset >= len(raw)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==============================================================================
# Unit tests: Census analytics & reductions ledger
# ==============================================================================


def test_census_analytics_and_ledger():
    """Verify DiT and TE analysis functions and reductions ledger calculations."""
    raw = _build_synthetic_gguf()
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(raw)
        temp_path = tf.name

    try:
        hdr = gguf_remote.header(temp_path)
        dit_stats = census.analyze_dit(hdr, temp_path)
        assert dit_stats["total_tensors"] == 2
        assert dit_stats["num_blocks"] == 1
        assert dit_stats["adaln"]["weight_bytes"] == 576
        assert dit_stats["trunk"]["total_bytes"] == 131072
        assert dit_stats["trunk"]["max_block_run_bytes"] == 131072
        assert dit_stats["trunk"]["ring_2slot_bytes"] == 2 * 131072

        ledger = census.compute_reductions_ledger(dit_stats)
        assert ledger["q4_k_m_gguf"]["bytes"] == 131072 + 576
        assert ledger["trunk_per_step"]["bytes"] == 131072
        assert ledger["ram_floor"]["ring_2slot_bytes"] == 262144
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
