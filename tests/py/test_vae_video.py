"""Pytest integration tests for Video ViT VAE and Y4M writer.

Validates parity between Diffusers AutoencoderKLMiniMaxH3, packed binary formats,
Y4M writer, and C unit test runner.
"""

from __future__ import annotations

import json
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

from tools import pack_video_vae

FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures" / "tiny_vae_video"
BUILD_DIR = ROOT_DIR / "build"


@pytest.fixture(scope="module")
def tiny_vae_video_fixtures() -> dict[str, Path]:
    """Ensure tiny_vae_video test fixtures exist."""
    bin_path = FIXTURES_DIR / "vae_video_dec.bin"
    lat_path = FIXTURES_DIR / "latents.bin"
    golden_path = FIXTURES_DIR / "golden_frames.bin"
    meta_path = FIXTURES_DIR / "meta.json"

    if not (bin_path.exists() and lat_path.exists() and golden_path.exists() and meta_path.exists()):
        pack_video_vae.make_tiny_vae_video(FIXTURES_DIR)

    return {
        "bin": bin_path,
        "lat": lat_path,
        "golden": golden_path,
        "meta": meta_path,
    }


def test_header_packing() -> None:
    """Validate 192-byte header format and constants."""
    hdr = pack_video_vae.pack_header(
        latent_channels=24,
        out_channels=3,
        dim=2048,
        heads=32,
        dim_head=64,
        num_layers=36,
        num_register_tokens=4,
        patch_size_t=4,
        patch_size=16,
        ffn_mult=4,
        clip_length=17,
        token_drop=3,
        rope_dim_ratio=0.75,
        rope_theta=100.0,
        norm_eps=1e-5,
    )
    assert len(hdr) == 192
    fields = struct.unpack("<14I 3f", hdr[:68])
    magic, ver, lc, oc, dim, heads, hd, nl, nr, pt, ps, ffn_m, cl, td = fields[:14]
    rope_ratio, rope_theta, norm_eps = fields[14:]

    assert magic == pack_video_vae.FH3_VAE_VIDEO_MAGIC
    assert ver == pack_video_vae.FH3_VAE_VIDEO_VERSION
    assert lc == 24
    assert oc == 3
    assert dim == 2048
    assert heads == 32
    assert hd == 64
    assert nl == 36
    assert nr == 4
    assert pt == 4
    assert ps == 16
    assert ffn_m == 4
    assert cl == 17
    assert td == 3
    assert abs(rope_ratio - 0.75) < 1e-6
    assert abs(rope_theta - 100.0) < 1e-6
    assert abs(norm_eps - 1e-5) < 1e-9


def test_tiny_fixture_integrity(tiny_vae_video_fixtures: dict[str, Path]) -> None:
    """Verify metadata and vector dimensions in generated tiny fixture."""
    with open(tiny_vae_video_fixtures["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["latent_channels"] == 24
    assert meta["out_channels"] == 3
    assert meta["dim"] == 32
    assert meta["heads"] == 2
    assert meta["dim_head"] == 16
    assert meta["num_layers"] == 2
    assert meta["num_register_tokens"] == 4
    assert meta["patch_size_t"] == 4
    assert meta["patch_size"] == 16
    assert meta["F"] == 7
    assert meta["out_frames"] == 22
    assert meta["out_height"] == 32
    assert meta["out_width"] == 32

    latents = np.fromfile(tiny_vae_video_fixtures["lat"], dtype=np.float32)
    assert latents.size == 24 * 7 * 2 * 2

    golden = np.fromfile(tiny_vae_video_fixtures["golden"], dtype=np.float32)
    assert golden.size == 3 * 22 * 32 * 32
    assert np.all(golden >= 0.0) and np.all(golden <= 1.0)


def test_diffusers_decode_parity(tiny_vae_video_fixtures: dict[str, Path]) -> None:
    """Verify that golden_frames matches fresh diffusers decode."""
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    with open(tiny_vae_video_fixtures["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    torch.manual_seed(42)
    np.random.seed(42)

    model = AutoencoderKLMiniMaxH3(
        in_channels=meta["out_channels"],
        out_channels=meta["out_channels"],
        latent_channels=meta["latent_channels"],
        spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
        temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
        decoder_num_layers=meta["num_layers"],
        decoder_num_attention_heads=meta["heads"],
        decoder_attention_head_dim=meta["dim_head"],
        decoder_num_register_tokens=meta["num_register_tokens"],
        decoder_ffn_mult=meta["ffn_mult"],
        decoder_rope_theta=meta["rope_theta"],
        decoder_rope_dim_ratio=meta["rope_dim_ratio"],
        decoder_norm_eps=meta["norm_eps"],
        clip_length=meta["clip_length"],
        token_drop=meta["token_drop"],
        latents_mean=meta["latents_mean"],
        latents_std=meta["latents_std"],
    )
    model.disable_tiling()
    model.eval()

    z = torch.from_numpy(np.fromfile(tiny_vae_video_fixtures["lat"], dtype=np.float32)).view(
        1, meta["latent_channels"], meta["F"], meta["h"], meta["w"]
    )

    l_mean_t = torch.tensor(meta["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1, 1)
    l_std_t = torch.tensor(meta["latents_std"], dtype=torch.float32).view(1, -1, 1, 1, 1)
    z_denorm = z * l_std_t + l_mean_t

    with torch.no_grad():
        raw_dec = model.decode(z_denorm, return_dict=False)[0]

    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1)
    expected_frames = (raw_dec.float() * pixel_std + pixel_mean).clamp(0.0, 1.0)

    golden = np.fromfile(tiny_vae_video_fixtures["golden"], dtype=np.float32).reshape(
        1, 3, meta["out_frames"], meta["out_height"], meta["out_width"]
    )
    np.testing.assert_allclose(golden, expected_frames.numpy(), rtol=1e-5, atol=1e-5)


def test_temporal_geometry_formula() -> None:
    """Verify that temporal clip math maps 5n+2 latent frames to 17n+5 pixel frames."""
    clip_length = 17
    temporal_ratio = 4
    token_drop = 3

    frame_pre_padding = (-clip_length) % temporal_ratio
    tokens_chunk_size = (clip_length + temporal_ratio - 1) // temporal_ratio
    token_overlap = (-token_drop) % tokens_chunk_size
    frame_overlap = max(token_overlap * temporal_ratio - frame_pre_padding, 0)

    assert frame_pre_padding == 3
    assert tokens_chunk_size == 5
    assert token_overlap == 2
    assert frame_overlap == 5

    # Check mapping for n = 0, 1, 2, 3
    for n in range(4):
        F = 5 * n + 2
        num_tokens = F + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        expected_pixel_frames = 17 * n + 5

        # pad_tokens should be 0 for all 5n+2 frames
        assert pad_tokens == 0
        assert num_chunks == (n if n > 0 else 0)  # for n=0, 2 latents -> 1 chunk with pad or 0?
        # Note: For n >= 1:
        if n >= 1:
            total_frames = num_chunks * 17 + 5
            assert total_frames == expected_pixel_frames


def test_c_unit_test_passes() -> None:
    """Execute compiled C test_vae_video runner and assert exit code 0."""
    exe_name = "test_vae_video.exe" if sys.platform == "win32" else "test_vae_video"
    exe_path = BUILD_DIR / exe_name

    if not exe_path.exists():
        pytest.skip(f"{exe_name} binary not found in {BUILD_DIR}")

    res = subprocess.run([str(exe_path)], capture_output=True, text=True, cwd=str(ROOT_DIR))
    assert res.returncode == 0, f"C unit tests failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "ALL TESTS PASSED" in res.stdout
