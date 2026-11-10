"""Tests for real-checkpoint conformance and test_real_block."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tools import conform_all

ROOT = Path(__file__).resolve().parent.parent.parent


def test_conform_log_exists_and_passes() -> None:
    """Verify that the committed conform_all_50.log exists and proves 50/50 blocks pass."""
    log_path = ROOT / "tests" / "fixtures" / "gates" / "conform_all_50.log"
    assert log_path.is_file(), f"Expected gate log at {log_path}"

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 51, f"Expected 51 lines in conform log, got {len(lines)}"

    block_pattern = re.compile(r"^Block (\d{2})/50: rel_L2 = ([0-9.eE+-]+) PASS \(misbound rel_L2 = ([0-9.]+) PASS\)$")
    for i in range(50):
        m = block_pattern.match(lines[i])
        assert m is not None, f"Line {i} does not match expected format: '{lines[i]}'"
        block_idx = int(m.group(1))
        assert block_idx == i, f"Expected block {i}, got {block_idx}"
        rel_l2 = float(m.group(2))
        misbound_rel_l2 = float(m.group(3))
        assert rel_l2 <= 1e-4, f"Block {i}: rel_L2 {rel_l2} > 1e-4"
        assert misbound_rel_l2 > 1e-2, f"Block {i}: misbound rel_L2 {misbound_rel_l2} <= 1e-2"

    assert lines[50] == "50/50 blocks PASS", f"Final line mismatch: '{lines[50]}'"


def test_test_real_block_binary_self_test() -> None:
    """Verify test_real_block runs its built-in self-test cleanly when called without arguments."""
    bin_path = conform_all.find_binary()
    assert bin_path.is_file(), f"Binary not found: {bin_path}"
    res = subprocess.run([str(bin_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"Self-test failed with code {res.returncode}: {res.stderr}"
    assert "[PASS]" in res.stdout or "[PASS]" in res.stderr


def test_conform_subset_programmatic(tmp_path: Path) -> None:
    """Verify conform_all helper runs programmatically for a fast subset of blocks."""
    subset_log = tmp_path / "conform_subset.log"
    success = conform_all.conform_all(
        pack_dir=None,
        log_path=subset_log,
        num_blocks=2,
        keep_fixture=False,
        verbose=False,
    )
    assert success is True
    assert subset_log.is_file()
    content = subset_log.read_text(encoding="utf-8")
    assert "2/2 blocks PASS" in content
