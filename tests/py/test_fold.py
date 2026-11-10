"""Test AdaLN folding against live GGUF computation and schedule hash integrity."""

from __future__ import annotations

import os
import struct
from pathlib import Path
import numpy as np
import pytest

from tools import census, fold_adaln, gguf_remote, hf_range, pack_dit
from tools.ref import h3_ref

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pack"


def test_schedule_hash():
    """Verify schedule hash calculation and sensitivity to parameter changes."""
    dmd_steps = [999, 874, 749, 624, 500, 375, 250, 125]
    h1 = fold_adaln.compute_schedule_hash(dmd_steps, 10.0, 3.0)
    # Expected precalculated value
    assert h1 == 0x2a0c8b9c5de6bb1

    # Sensitivity: change in step
    dmd_alt = [999, 874, 749, 624, 500, 375, 250, 120]
    h2 = fold_adaln.compute_schedule_hash(dmd_alt, 10.0, 3.0)
    assert h1 != h2

    # Sensitivity: change in video shift
    h3 = fold_adaln.compute_schedule_hash(dmd_steps, 9.0, 3.0)
    assert h1 != h3

    # Sensitivity: change in audio shift
    h4 = fold_adaln.compute_schedule_hash(dmd_steps, 10.0, 4.0)
    assert h1 != h4


def test_fold_matches_live_synthetic(tmp_path):
    """Test AdaLN folding on synthetic fixture matches live silu(temb) @ W.T + b bit-exact."""
    tiny_gguf = FIXTURE_DIR / "tiny_dit.gguf"
    if not tiny_gguf.exists():
        from tests.fixtures.pack.make_tiny_pack import make_tiny_pack_fixtures
        make_tiny_pack_fixtures(FIXTURE_DIR)

    out_path = fold_adaln.fold_adaln(
        src=str(tiny_gguf),
        out_dir=tmp_path,
        layers=[0, 1],
        verbose=False,
    )

    hdr = gguf_remote.header(str(tiny_gguf))
    t_map = {t.name: t for t in hdr.tensors}

    # Fetch time_embedder live
    w_tin = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(str(tiny_gguf), hdr, t_map["time_embedder.proj_in.weight"]), t_map["time_embedder.proj_in.weight"].type, t_map["time_embedder.proj_in.weight"].dims)
    b_tin = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(str(tiny_gguf), hdr, t_map["time_embedder.proj_in.bias"]), t_map["time_embedder.proj_in.bias"].type, t_map["time_embedder.proj_in.bias"].dims)
    w_tout = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(str(tiny_gguf), hdr, t_map["time_embedder.proj_out.weight"]), t_map["time_embedder.proj_out.weight"].type, t_map["time_embedder.proj_out.weight"].dims)
    b_tout = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(str(tiny_gguf), hdr, t_map["time_embedder.proj_out.bias"]), t_map["time_embedder.proj_out.bias"].type, t_map["time_embedder.proj_out.bias"].dims)

    if w_tin.shape[0] != b_tin.shape[0]: w_tin = w_tin.T
    if w_tout.shape[0] != b_tout.shape[0]: w_tout = w_tout.T
    freq_dim = int(w_tin.shape[1])

    plan = h3_ref.sampler_plan(fold_adaln.DEFAULT_DMD_STEPS, 10.0, 3.0)

    # Read back folded table
    with open(out_path, "rb") as f:
        hdr_data = f.read(128)
        magic, ver, n_steps, n_blocks, pairs, hidden, shash, t_off, t_bytes, f_off, f_bytes = struct.unpack("<IIIIIIQQQQQ", hdr_data[:64])
        assert magic == fold_adaln.FH3_ADALN_MAGIC
        assert ver == fold_adaln.FH3_ADALN_VERSION
        assert n_steps == 8
        assert n_blocks == 2
        assert pairs == 3
        assert hidden == 64

        f.seek(t_off)
        trunk_buf = f.read(t_bytes)
        table = np.frombuffer(trunk_buf, dtype=np.float32).reshape(n_steps, n_blocks, pairs, 6, hidden)

    # Test for block 0 and 1 at step 3
    step = 3
    t_v = float(1.0 - plan[step].sigma_v)
    t_a = float(1.0 - plan[step].sigma_a)

    tf_v = h3_ref.get_timestep_embedding(np.array([t_v], dtype=np.float32), freq_dim, flip_sin_to_cos=True)
    temb_v = fold_adaln._silu(tf_v @ w_tin.T + b_tin) @ w_tout.T + b_tout
    act_v = fold_adaln._silu(temb_v)

    tf_a = h3_ref.get_timestep_embedding(np.array([t_a], dtype=np.float32), freq_dim, flip_sin_to_cos=True)
    temb_a = fold_adaln._silu(tf_a @ w_tin.T + b_tin) @ w_tout.T + b_tout
    act_a = fold_adaln._silu(temb_a)

    for b in [0, 1]:
        wb = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(str(tiny_gguf), hdr, t_map[f"blocks.{b}.adaln_proj.linear.weight"]), t_map[f"blocks.{b}.adaln_proj.linear.weight"].type, t_map[f"blocks.{b}.adaln_proj.linear.weight"].dims)
        bb = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(str(tiny_gguf), hdr, t_map[f"blocks.{b}.adaln_proj.linear.bias"]), t_map[f"blocks.{b}.adaln_proj.linear.bias"].type, t_map[f"blocks.{b}.adaln_proj.linear.bias"].dims)
        if wb.shape[0] != bb.shape[0]: wb = wb.T

        live_proj_v = (act_v @ wb.T + bb).reshape(3, 6, hidden)
        live_proj_a = (act_a @ wb.T + bb).reshape(3, 6, hidden)

        # Pair 0 (video)
        np.testing.assert_array_equal(table[step, b, 0], live_proj_v[0])
        # Pair 1 (text)
        np.testing.assert_array_equal(table[step, b, 1], live_proj_v[1])
        # Pair 2 (audio)
        np.testing.assert_array_equal(table[step, b, 2], live_proj_a[2])


@pytest.mark.skipif(
    os.environ.get("FASTH3_SKIP_REMOTE_TESTS") == "1",
    reason="FASTH3_SKIP_REMOTE_TESTS set",
)
def test_fold_matches_live_adaln(tmp_path):
    """Step 1 requirement: For remote block 0 and 7, step 3, table[...] == silu(temb(t)) @ W.T + b bit-exact in f32."""
    src_url = census.DEFAULT_DIT_URL
    target_blocks = [0, 7]

    out_path = fold_adaln.fold_adaln(
        src=src_url,
        out_dir=tmp_path,
        layers=target_blocks,
        verbose=True,
    )

    hdr = gguf_remote.header(src_url)
    t_map = {t.name: t for t in hdr.tensors}

    # Read back folded table
    with open(out_path, "rb") as f:
        hdr_data = f.read(128)
        magic, ver, n_steps, n_blocks, pairs, hidden, shash, t_off, t_bytes, f_off, f_bytes = struct.unpack("<IIIIIIQQQQQ", hdr_data[:64])
        assert magic == fold_adaln.FH3_ADALN_MAGIC
        assert n_steps == 8
        assert n_blocks == len(target_blocks)
        assert pairs == 3
        assert hidden == 5376

        f.seek(t_off)
        trunk_buf = f.read(t_bytes)
        table = np.frombuffer(trunk_buf, dtype=np.float32).reshape(n_steps, n_blocks, pairs, 6, hidden)

    # Compute live for step 3
    step = 3
    plan = h3_ref.sampler_plan(fold_adaln.DEFAULT_DMD_STEPS, 10.0, 3.0)
    t_v = float(1.0 - plan[step].sigma_v)
    t_a = float(1.0 - plan[step].sigma_a)

    w_tin = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(src_url, hdr, t_map["time_embedder.proj_in.weight"]), t_map["time_embedder.proj_in.weight"].type, t_map["time_embedder.proj_in.weight"].dims)
    b_tin = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(src_url, hdr, t_map["time_embedder.proj_in.bias"]), t_map["time_embedder.proj_in.bias"].type, t_map["time_embedder.proj_in.bias"].dims)
    w_tout = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(src_url, hdr, t_map["time_embedder.proj_out.weight"]), t_map["time_embedder.proj_out.weight"].type, t_map["time_embedder.proj_out.weight"].dims)
    b_tout = fold_adaln._dequantize_tensor(fold_adaln._read_tensor_data(src_url, hdr, t_map["time_embedder.proj_out.bias"]), t_map["time_embedder.proj_out.bias"].type, t_map["time_embedder.proj_out.bias"].dims)

    if w_tin.shape[0] != b_tin.shape[0]: w_tin = w_tin.T
    if w_tout.shape[0] != b_tout.shape[0]: w_tout = w_tout.T
    freq_dim = int(w_tin.shape[1])

    tf_v = h3_ref.get_timestep_embedding(np.array([t_v], dtype=np.float32), freq_dim, flip_sin_to_cos=True)
    temb_v = fold_adaln._silu(tf_v @ w_tin.T + b_tin) @ w_tout.T + b_tout
    act_v = fold_adaln._silu(temb_v)

    tf_a = h3_ref.get_timestep_embedding(np.array([t_a], dtype=np.float32), freq_dim, flip_sin_to_cos=True)
    temb_a = fold_adaln._silu(tf_a @ w_tin.T + b_tin) @ w_tout.T + b_tout
    act_a = fold_adaln._silu(temb_a)

    # Verify both block 0 and block 7 at step 3
    for idx, b_num in enumerate(target_blocks):
        w_b = fold_adaln._dequantize_tensor(
            fold_adaln._read_tensor_data(src_url, hdr, t_map[f"blocks.{b_num}.adaln_proj.linear.weight"]),
            t_map[f"blocks.{b_num}.adaln_proj.linear.weight"].type,
            t_map[f"blocks.{b_num}.adaln_proj.linear.weight"].dims,
            t_map[f"blocks.{b_num}.adaln_proj.linear.weight"].orig_shape,
        )
        b_b = fold_adaln._dequantize_tensor(
            fold_adaln._read_tensor_data(src_url, hdr, t_map[f"blocks.{b_num}.adaln_proj.linear.bias"]),
            t_map[f"blocks.{b_num}.adaln_proj.linear.bias"].type,
            t_map[f"blocks.{b_num}.adaln_proj.linear.bias"].dims,
        )
        if w_b.shape[0] != b_b.shape[0]:
            w_b = w_b.T

        live_proj_v = (act_v @ w_b.T + b_b).reshape(3, 6, hidden)
        live_proj_a = (act_a @ w_b.T + b_b).reshape(3, 6, hidden)

        # Bit-exact assertion on all 6 modulation vectors for each pair
        # Pair 0 (video)
        np.testing.assert_array_equal(table[step, idx, 0], live_proj_v[0])
        # Pair 1 (text)
        np.testing.assert_array_equal(table[step, idx, 1], live_proj_v[1])
        # Pair 2 (audio)
        np.testing.assert_array_equal(table[step, idx, 2], live_proj_a[2])


def test_pack_dit_synthetic(tmp_path):
    """Verify that pack_dit correctly packages layer runs and globals from tiny fixture."""
    tiny_gguf = FIXTURE_DIR / "tiny_dit.gguf"
    if not tiny_gguf.exists():
        from tests.fixtures.pack.make_tiny_pack import make_tiny_pack_fixtures
        make_tiny_pack_fixtures(FIXTURE_DIR)

    trunk_path, globals_path = pack_dit.pack_dit(
        src=str(tiny_gguf),
        out_dir=tmp_path,
        layers=[0, 1],
        write_globals=True,
        verbose=False,
    )
    assert trunk_path.exists()
    assert globals_path.exists()

    with open(trunk_path, "rb") as f:
        hdr_data = f.read(128)
        magic, ver, n_layers, align, idx_off, idx_bytes, d_off, tot_bytes = struct.unpack("<IIIIQQQQ", hdr_data[:48])
        assert magic == pack_dit.FH3_PACK_MAGIC
        assert ver == pack_dit.FH3_PACK_VERSION
        assert n_layers == 2
        assert align == 4096
        assert d_off % 4096 == 0

        # Read layer descriptors
        f.seek(idx_off)
        for L in range(n_layers):
            layer_desc = f.read(1536)
            f_off, s_bytes, n_tensors, _ = struct.unpack("<QQII", layer_desc[:24])
            assert f_off % 4096 == 0
            assert s_bytes % 4096 == 0
            assert n_tensors == 9

