"""Pytest integration tests for truncated Qwen3-VL text encoder.

Validates parity between HuggingFace text model, NumPy reference implementation,
packed binary formats, prompt cache, and C unit test runner.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools import conform_te, pack_te
from tools.ref import qwen3_ref

FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures" / "tiny_te"
BUILD_DIR = ROOT_DIR / "build"


@pytest.fixture(scope="module")
def tiny_te_fixtures() -> dict[str, Path]:
    """Ensure tiny_te test fixtures exist."""
    gguf_path = FIXTURES_DIR / "tiny_te.gguf"
    trunk_path = FIXTURES_DIR / "te_trunk.bin"
    embed_path = FIXTURES_DIR / "te_embed.bin"
    ids_path = FIXTURES_DIR / "prompt_ids.bin"
    golden_prenorm_path = FIXTURES_DIR / "golden_prenorm.bin"
    golden_normed_path = FIXTURES_DIR / "golden_normed.bin"
    cfg_path = FIXTURES_DIR / "te_config.json"

    if not (gguf_path.exists() and trunk_path.exists() and embed_path.exists()):
        pack_te.make_tiny_te(FIXTURES_DIR, verbose=False)

    return {
        "gguf": gguf_path,
        "trunk": trunk_path,
        "embed": embed_path,
        "ids": ids_path,
        "golden_prenorm": golden_prenorm_path,
        "golden_normed": golden_normed_path,
        "cfg": cfg_path,
    }


def test_fixtures_presence_and_structure(tiny_te_fixtures: dict[str, Path]) -> None:
    """Verify tiny_te fixtures exist and have non-zero sizes."""
    for key, path in tiny_te_fixtures.items():
        assert path.is_file(), f"Fixture missing: {path}"
        assert path.stat().st_size > 0, f"Fixture empty: {path}"

    # Verify trunk binary header
    with open(tiny_te_fixtures["trunk"], "rb") as f:
        hdr = f.read(128)
        magic, ver, num_layers, align, idx_off, idx_bytes, data_off, total_bytes = struct.unpack("<IIIIQQQQ80x", hdr)
        assert magic == pack_te.FH3_PACK_MAGIC
        assert ver == pack_te.FH3_PACK_VERSION
        assert num_layers == 3
        assert align == pack_te.FH3_PACK_ALIGNMENT

    # Verify embed binary header
    with open(tiny_te_fixtures["embed"], "rb") as f:
        hdr = f.read(64)
        magic, ver, quant_type, vocab_sz, hidden, row_bytes, data_offset, reserved = struct.unpack("<IIIIIIQ32s", hdr)
        assert magic == pack_te.FH3_EMBED_MAGIC
        assert ver == pack_te.FH3_EMBED_VERSION
        assert quant_type == 0
        assert vocab_sz == 1000
        assert hidden == 128


def test_qwen3_ref_matches_hf() -> None:
    """Verify NumPy reference forward pass matches HuggingFace Qwen3VLTextModel."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextConfig,
        Qwen3VLTextModel,
    )

    torch.manual_seed(123)
    hf_cfg = Qwen3VLTextConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
    )
    model = Qwen3VLTextModel(hf_cfg)
    model.eval()

    cfg = qwen3_ref.Qwen3TeConfig(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        num_layers=4,
        intermediate_size=128,
        norm_eps=1e-6,
        qk_norm_eps=1e-6,
        rope_theta=5000000.0,
    )

    embed_table = model.embed_tokens.weight.detach().numpy()
    layers = []
    for L in range(3):
        bl = model.layers[L]
        layers.append({
            "input_layernorm.weight": bl.input_layernorm.weight.detach().numpy(),
            "q_proj.weight": bl.self_attn.q_proj.weight.detach().numpy(),
            "k_proj.weight": bl.self_attn.k_proj.weight.detach().numpy(),
            "v_proj.weight": bl.self_attn.v_proj.weight.detach().numpy(),
            "q_norm.weight": bl.self_attn.q_norm.weight.detach().numpy(),
            "k_norm.weight": bl.self_attn.k_norm.weight.detach().numpy(),
            "o_proj.weight": bl.self_attn.o_proj.weight.detach().numpy(),
            "post_attention_layernorm.weight": bl.post_attention_layernorm.weight.detach().numpy(),
            "gate_proj.weight": bl.mlp.gate_proj.weight.detach().numpy(),
            "up_proj.weight": bl.mlp.up_proj.weight.detach().numpy(),
            "down_proj.weight": bl.mlp.down_proj.weight.detach().numpy(),
        })

    token_ids = [7, 42, 19, 88]
    x_ref = qwen3_ref.qwen3_forward(token_ids, embed_table, layers, cfg, stop_layer=3)

    # HF forward returning all hidden states
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    # Output after 3 layers before final norm is hidden_states[3]
    x_hf_prenorm = out.hidden_states[3].squeeze(0).numpy()

    rel_l2 = np.linalg.norm(x_ref - x_hf_prenorm) / np.linalg.norm(x_hf_prenorm)
    assert rel_l2 <= 1e-4, f"Reference mismatch with HF: rel-L2 = {rel_l2:e}"


def test_mrope_text_eq_1d() -> None:
    """Verify 3D M-RoPE on (pos, pos, pos) reduces identically to 1-D RoPE."""
    positions = np.arange(64)
    # Head dim 128
    assert qwen3_ref.verify_mrope_text_eq_1d(positions, head_dim=128)
    # Head dim 32
    assert qwen3_ref.verify_mrope_text_eq_1d(positions, head_dim=32)


def test_prompt_cache(tmp_path: Path) -> None:
    """Verify prompt cache operations: save, hit, miss."""
    cache_dir = tmp_path / "cache"
    token_ids = [101, 202, 303, 404]
    hidden_size = 128

    # Miss before save
    cached = qwen3_ref.prompt_cache_lookup(cache_dir, token_ids, hidden_size)
    assert cached is None

    # Save embedding tensor
    fake_emb = np.random.randn(len(token_ids), hidden_size).astype(np.float32)
    saved_path = qwen3_ref.prompt_cache_save(cache_dir, token_ids, fake_emb)
    assert saved_path.is_file()

    # Hit after save
    cached = qwen3_ref.prompt_cache_lookup(cache_dir, token_ids, hidden_size)
    assert cached is not None
    assert np.array_equal(cached, fake_emb)

    # Different IDs miss
    diff_ids = [101, 202, 303, 999]
    assert qwen3_ref.prompt_cache_lookup(cache_dir, diff_ids, hidden_size) is None


def test_conform_te_tiny() -> None:
    """Run conform_te on tiny fixture layers 0, 1, 2."""
    results = conform_te.conform_te(tiny=True, layers=[0, 1, 2], tol=1e-4, verbose=False)
    assert len(results) == 3
    for L, rel_l2 in results.items():
        assert rel_l2 <= 1e-4, f"Layer {L} rel-L2 {rel_l2:e} exceeds 1e-4"


def test_c_binary_test_te_tiny() -> None:
    """Verify that build/test_te_tiny runs and all tests pass."""
    bin_candidates = [
        BUILD_DIR / "test_te_tiny.exe",
        BUILD_DIR / "test_te_tiny",
        ROOT_DIR / "test_te_tiny.exe",
        ROOT_DIR / "test_te_tiny",
    ]
    binary = None
    for b in bin_candidates:
        if b.is_file():
            binary = b
            break
    assert binary is not None, f"Could not find test_te_tiny executable. Checked {bin_candidates}"

    res = subprocess.run([str(binary)], capture_output=True, text=True, cwd=str(ROOT_DIR))
    assert res.returncode == 0, f"test_te_tiny failed (exit {res.returncode}):\n{res.stdout}\n{res.stderr}"
    assert "All Truncated Text Encoder Tests Passed!" in res.stdout


@pytest.mark.skipif(os.environ.get("FASTH3_SKIP_REMOTE_TESTS") == "1", reason="Remote tests skipped")
def test_conform_te_remote() -> None:
    """Test remote GGUF conformance if remote tests are not skipped."""
    results = conform_te.conform_te(tiny=False, layers=[0, 25, 49], tol=1e-4, verbose=True)
    assert len(results) == 3
    for L, rel_l2 in results.items():
        assert rel_l2 <= 1e-4
