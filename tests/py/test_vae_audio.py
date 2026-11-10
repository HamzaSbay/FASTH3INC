"""Pytest integration tests for BigVGAN audio VAE and WAV writer.

Validates parity between Diffusers AutoencoderKLMiniMaxH3Audio, packed binary formats,
WAV writer, and C unit test runner.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools import pack_vae

FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures" / "tiny_vae_audio"
BUILD_DIR = ROOT_DIR / "build"


@pytest.fixture(scope="module")
def tiny_vae_audio_fixtures() -> dict[str, Path]:
    """Ensure tiny_vae_audio test fixtures exist."""
    bin_path = FIXTURES_DIR / "vae_audio_dec.bin"
    lat_path = FIXTURES_DIR / "latents.bin"
    golden_path = FIXTURES_DIR / "golden_pcm.bin"
    meta_path = FIXTURES_DIR / "meta.json"

    if not (bin_path.exists() and lat_path.exists() and golden_path.exists() and meta_path.exists()):
        pack_vae.make_tiny_vae_audio(FIXTURES_DIR)

    return {
        "bin": bin_path,
        "lat": lat_path,
        "golden": golden_path,
        "meta": meta_path,
    }


def test_header_packing() -> None:
    """Validate 192-byte header format and constants."""
    hdr = pack_vae.pack_header(
        latent_channels=32,
        latent_dim=2048,
        decoder_dim=1024,
        rates=(5, 5, 2, 2, 2, 2, 2),
        kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
        resblock_kernels=(3, 7, 11),
        resblock_dilations=(1, 3, 5),
    )
    assert len(hdr) == 192
    magic, ver, lc, ld, dd, nu, nb, nd = struct.unpack("<8I", hdr[:32])
    assert magic == pack_vae.FH3_VAE_AUDIO_MAGIC
    assert ver == pack_vae.FH3_VAE_AUDIO_VERSION
    assert lc == 32
    assert ld == 2048
    assert dd == 1024
    assert nu == 7
    assert nb == 3
    assert nd == 3


def test_tiny_fixture_integrity(tiny_vae_audio_fixtures: dict[str, Path]) -> None:
    """Verify metadata and vector dimensions in generated tiny fixture."""
    with open(tiny_vae_audio_fixtures["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["latent_channels"] == 32
    assert meta["total_upsample"] == 4
    assert meta["num_frames"] == 4

    latents = np.fromfile(tiny_vae_audio_fixtures["lat"], dtype=np.float32)
    assert latents.size == 2 * 4 * 32

    golden = np.fromfile(tiny_vae_audio_fixtures["golden"], dtype=np.float32)
    assert golden.size == 2 * 16
    assert np.all(golden >= -1.0) and np.all(golden <= 1.0)


def test_diffusers_decode_parity(tiny_vae_audio_fixtures: dict[str, Path]) -> None:
    """Verify that golden_pcm matches fresh diffusers decode."""
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

    with open(tiny_vae_audio_fixtures["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    torch.manual_seed(42)
    np.random.seed(42)

    lc = meta["latent_channels"]
    latents_mean = [0.05 * (i % 7 - 3) for i in range(lc)]
    latents_std = [1.0 + 0.05 * (i % 5) for i in range(lc)]

    model = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=8,
        latent_dim=meta["latent_dim"],
        latent_channels=lc,
        decoder_dim=meta["decoder_dim"],
        encoder_rates=(2, 2),
        decoder_rates=tuple(meta["rates"]),
        decoder_kernel_sizes=tuple(meta["kernel_sizes"]),
        resblock_kernel_sizes=tuple(meta["resblock_kernels"]),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        latents_mean=latents_mean,
        latents_std=latents_std,
    )
    model.eval()

    lat = np.fromfile(tiny_vae_audio_fixtures["lat"], dtype=np.float32).reshape(2, 4, lc)
    golden = np.fromfile(tiny_vae_audio_fixtures["golden"], dtype=np.float32).reshape(2, 16)

    mean_t = torch.tensor(latents_mean, dtype=torch.float32).view(1, 1, lc)
    std_t = torch.tensor(latents_std, dtype=torch.float32).view(1, 1, lc)
    lat_denorm = torch.tensor(lat) * std_t + mean_t
    lat_for_diffusers = lat_denorm.permute(0, 2, 1)

    with torch.no_grad():
        pcm_fresh = model.decode(lat_for_diffusers)[0].squeeze(1).numpy()

    rel_l2 = np.linalg.norm(pcm_fresh - golden) / np.linalg.norm(golden)
    assert rel_l2 < 1e-6


def test_c_unit_test_vae_audio() -> None:
    """Run C executable test_vae_audio and assert 0 exit code."""
    exe_name = "test_vae_audio.exe" if os.name == "nt" else "test_vae_audio"
    exe_path = BUILD_DIR / exe_name
    if not exe_path.exists():
        pytest.skip(f"Test executable {exe_path} not found; run cmake build first.")

    res = subprocess.run([str(exe_path)], capture_output=True, text=True, cwd=str(ROOT_DIR))
    assert res.returncode == 0, f"C test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "All Audio VAE Tests Passed" in res.stdout
