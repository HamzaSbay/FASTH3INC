"""Integration tests for FastH3 unified command-line interface (fh3).

Validates:
- End-to-end execution on tiny fixtures (Phase 1, Phase 2, Phase 3 -> out.y4m, out.wav, report.json)
- Custom resolution snapping to multiples of 32 px and 17n + 5 frames
- Low-memory preflight rejection (exit code 3)
- Corrupt/missing model file rejection (exit code 4)
- Subcommand syntax and usage errors (exit code 2)
- Diagnostic doctor checks (CPU, RAM, disk, benchmark) (exit code 0)
- Cancel and resume functionality (--resume, exit code 5 on cancel, exit 0 on resume)
- Phase-by-phase subcommand execution (encode, denoise, decode)
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = ROOT_DIR / "build"
EXE_NAME = "fh3.exe" if sys.platform == "win32" else "fh3"
FH3_BIN = BUILD_DIR / EXE_NAME


def run_fh3(*args, cwd=None):
    """Helper to run the compiled fh3 executable and return CompletedProcess."""
    if not FH3_BIN.exists():
        pytest.skip(f"{EXE_NAME} binary not found at {FH3_BIN}. Build project first.")
    cmd = [str(FH3_BIN)] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd or ROOT_DIR)
    )


def test_cli_help():
    """Verify 'fh3 help' prints usage text and exits 0."""
    res = run_fh3("help")
    assert res.returncode == 0
    assert "FastH3-in-C Engine" in res.stdout
    assert "Usage:" in res.stdout
    assert "fh3 run" in res.stdout


def test_cli_version():
    """Verify 'fh3 version' prints version string and exits 0."""
    res = run_fh3("version")
    assert res.returncode == 0
    assert "FastH3-in-C Engine" in res.stdout
    assert "v0.1.0" in res.stdout


def test_cli_usage_errors():
    """Verify syntax errors and unknown arguments exit with code 2."""
    # Unknown subcommand
    res = run_fh3("nonexistent_subcommand")
    assert res.returncode == 2
    assert "Unknown subcommand" in res.stderr

    # Unknown option
    res = run_fh3("run", "--unknown-flag")
    assert res.returncode == 2
    assert "Unknown argument" in res.stderr

    # Missing argument value
    res = run_fh3("run", "--width")
    assert res.returncode == 2
    assert "requires an integer value" in res.stderr


def test_cli_doctor():
    """Verify 'fh3 doctor' prints hardware diagnostics, verifies model files and exits 0."""
    res = run_fh3("doctor", "--tiny-fixture")
    assert res.returncode == 0
    assert "[CPU]" in res.stdout
    assert "AVX2 SIMD:" in res.stdout
    assert "[Memory]" in res.stdout
    assert "Available RAM:" in res.stdout
    assert "[Disk]" in res.stdout
    assert "[Model Files]" in res.stdout
    assert "[NVMe Read Benchmark]" in res.stdout
    assert "Sequential read speed:" in res.stdout


def test_cli_snaps_resolution():
    """Verify custom dimensions snap up to multiples of 32 and 17n + 5."""
    # 500x300, 30 frames -> 512x320, 39 frames (latent: 32x20, 12 frames)
    res = run_fh3("run", "--width", "500", "--height", "300", "--frames", "30", "--dry-run")
    assert res.returncode == 0
    assert "Resolved size: 512x320, 39 frames" in res.stdout
    assert "latent: 32x20, 12 frames" in res.stdout

    # Small preset
    res2 = run_fh3("run", "--preset", "small", "--dry-run")
    assert res2.returncode == 0
    assert "Resolved size: 640x384, 39 frames" in res2.stdout

    # Preset + override: tiny preset with custom frame count
    res3 = run_fh3("run", "--preset", "tiny", "--frames", "39", "--dry-run")
    assert res3.returncode == 0
    assert "Resolved size: 448x256, 39 frames" in res3.stdout


def test_cli_refuses_low_mem(tmp_path):
    """Verify --mem-gb 0.0001 triggers preflight memory refusal and exits with code 3."""
    out_dir = tmp_path / "low_mem_out"
    res = run_fh3("run", "--preset", "tiny", "--tiny-fixture", "--mem-gb", "0.0001", "--out", str(out_dir))
    assert res.returncode == 3
    assert "below minimum requirement" in res.stderr or "requested memory" in res.stderr


def test_cli_bad_model(tmp_path):
    """Verify truncated or corrupt model file triggers refusal and exits with code 4."""
    bad_model_dir = tmp_path / "bad_models"
    bad_model_dir.mkdir(parents=True, exist_ok=True)

    # Create truncated model file
    with open(bad_model_dir / "te_trunk.bin", "wb") as f:
        f.write(b"CORRUPT_BYTES")

    out_dir = tmp_path / "bad_model_out"
    res = run_fh3("run", "--models-dir", str(bad_model_dir), "--out", str(out_dir))
    assert res.returncode == 4
    assert "error: failed to open text encoder" in res.stderr or "corrupt" in res.stderr


def test_cli_tiny_end_to_end(tmp_path):
    """Run full fh3 pipeline on tiny fixtures and verify out.y4m, out.wav, report.json, exit 0."""
    out_dir = tmp_path / "tiny_e2e"
    json_path = out_dir / "report.json"

    res = run_fh3(
        "run",
        "--preset", "tiny",
        "--tiny-fixture",
        "--prompt", "a red fox running through snow, wind sounds",
        "--out", str(out_dir),
        "--json", str(json_path)
    )

    assert res.returncode == 0, f"fh3 run failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify video output
    y4m_path = out_dir / "out.y4m"
    assert y4m_path.exists()
    assert y4m_path.stat().st_size > 1000
    with open(y4m_path, "rb") as f:
        y4m_header = f.read(40)
        assert y4m_header.startswith(b"YUV4MPEG2")

    # Verify audio output
    wav_path = out_dir / "out.wav"
    assert wav_path.exists()
    assert wav_path.stat().st_size > 44
    with open(wav_path, "rb") as f:
        wav_header = f.read(12)
        assert wav_header[:4] == b"RIFF"
        assert wav_header[8:12] == b"WAVE"

    # Verify telemetry JSON report
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["exit_code"] == 0
    assert report["peak_rss_mb"] > 0
    assert report["pinned_layers"] >= 0
    assert report["timings"]["total_s"] > 0
    assert report["timings"]["encode_s"] >= 0
    assert report["timings"]["denoise_s"] >= 0
    assert report["timings"]["decode_s"] >= 0
    assert report["hardware"]["cpu_cores"] > 0


def test_cli_cancel_resume(tmp_path):
    """Verify intermediate step checkpoint saving on interrupt and seamless --resume."""
    out_dir = tmp_path / "cancel_resume"

    # 1. Run with interrupt after step 3 -> exits 5, saves latents.step3.bin
    res1 = run_fh3(
        "run",
        "--preset", "tiny",
        "--tiny-fixture",
        "--seed", "42",
        "--interrupt-after", "3",
        "--out", str(out_dir)
    )
    assert res1.returncode == 5
    assert (out_dir / "latents.step3.bin").exists()
    assert (out_dir / "latents.step3.bin").stat().st_size > 0

    # Verify out.y4m does NOT exist yet
    assert not (out_dir / "out.y4m").exists()

    # 2. Resume from checkpoint -> completes remaining steps (4..7) and exits 0
    res2 = run_fh3(
        "run",
        "--preset", "tiny",
        "--tiny-fixture",
        "--seed", "42",
        "--resume",
        "--out", str(out_dir)
    )
    assert res2.returncode == 0
    assert "Resuming diffusion from checkpoint 'latents.step3.bin' (starting step 4)" in res2.stdout
    assert (out_dir / "out.y4m").exists()
    assert (out_dir / "out.wav").exists()


def test_cli_phase_isolation_subcommands(tmp_path):
    """Test sequential execution of individual phase subcommands: encode -> denoise -> decode."""
    out_dir = tmp_path / "phase_subcmds"

    # Phase 1: Encode
    res_enc = run_fh3(
        "encode",
        "--preset", "tiny",
        "--tiny-fixture",
        "--prompt", "cinematic nature scene",
        "--out", str(out_dir)
    )
    assert res_enc.returncode == 0
    assert (out_dir / "prompt.emb").exists()
    assert (out_dir / "prompt.emb").stat().st_size > 0

    # Phase 2: Denoise
    res_den = run_fh3(
        "denoise",
        "--preset", "tiny",
        "--tiny-fixture",
        "--seed", "100",
        "--out", str(out_dir)
    )
    assert res_den.returncode == 0
    assert (out_dir / "video_latents.bin").exists()
    assert (out_dir / "audio_latents.bin").exists()

    # Phase 3: Decode
    res_dec = run_fh3(
        "decode",
        "--preset", "tiny",
        "--tiny-fixture",
        "--out", str(out_dir)
    )
    assert res_dec.returncode == 0
    assert (out_dir / "out.y4m").exists()
    assert (out_dir / "out.wav").exists()
