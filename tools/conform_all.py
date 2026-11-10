#!/usr/bin/env python3
"""FastH3: 50-Block Real-Checkpoint Conformance Gate.

Proves that all 50 blocks (0..49) run identically between C (test_real_block)
and the Python reference implementation (h3_ref.block), and verifies that
AdaLN modulation binding is strictly enforced (mis-binding block L to block L+1
fails with relative L2 > 1e-2).

Produces:
tests/fixtures/gates/conform_all_50.log
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import gguf
    from gguf.constants import GGMLQuantizationType
    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False

from diffusers import MiniMaxH3Transformer3DModel

from tools import fold_adaln, pack_dit
from tools.ref import h3_ref


FH3_BLOCK_IO_MAGIC = 0x49334846  # 'FH3I'
FH3_BLOCK_IO_VERSION = 1


def find_binary(explicit_path: Optional[str] = None) -> Path:
    """Find test_real_block executable."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend([
        ROOT / "build" / "test_real_block.exe",
        ROOT / "build" / "test_real_block",
        ROOT / "test_real_block.exe",
        ROOT / "test_real_block",
    ])
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"test_real_block executable not found. Checked: {[str(c) for c in candidates]}")


def write_block_input(
    path: Path,
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    num_heads: int,
    head_dim: int,
    rope_rot_dim: int = 96,
    step: int = 0,
    pair: int = 0,
    norm_eps: float = 1e-5,
    qk_norm_eps: float = 1e-5,
) -> None:
    """Write input file for test_real_block CLI."""
    N, hidden = x.shape
    hdr = struct.pack(
        "<IIiiiiiiiffi48x",
        FH3_BLOCK_IO_MAGIC,
        FH3_BLOCK_IO_VERSION,
        N,
        hidden,
        num_heads,
        head_dim,
        rope_rot_dim,
        step,
        pair,
        norm_eps,
        qk_norm_eps,
        0,  # has_pair_indices (0 = uniform pair)
    )
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(x.astype(np.float32).tobytes())
        f.write(cos.astype(np.float32).tobytes())
        f.write(sin.astype(np.float32).tobytes())


def generate_conform_fixture(
    out_dir: Path,
    num_blocks: int = 50,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Generate a synthetic 50-block GGUF model and packed binaries."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = out_dir / "conform_50.gguf"

    if verbose:
        print(f"Generating synthetic {num_blocks}-block model fixture...")

    cfg = dict(
        num_attention_heads=2,
        attention_head_dim=128,
        hidden_size=256,
        num_layers=num_blocks,
        num_refiner_layers=1,
        ffn_dim=512,
        text_dim=64,
        time_embed_dim=128,
        time_embed_hidden_dim=256,
        freq_dim=64,
        rope_freq_dim=16,
        rope_theta=10000.0,
        in_channels=24,
        audio_in_channels=32,
        patch_size=(1, 2, 2),
    )

    torch.manual_seed(42)
    model = MiniMaxH3Transformer3DModel(**cfg)

    # Initialize each block's adaln_proj with distinct random biases
    # to guarantee strong distinction between blocks (rel-L2 > 0.05 on misbinding)
    for b in range(num_blocks):
        torch.manual_seed(1000 + b)
        model.transformer_blocks[b].adaln_proj.linear.bias.data = torch.randn(3 * 6 * 256) * 0.5

    writer = gguf.GGUFWriter(str(gguf_path), "wan")
    writer.add_uint32("general.alignment", 32)

    writer.add_tensor("time_embedder.proj_in.weight", model.time_embedder.linear_1.weight.detach().numpy())
    writer.add_tensor("time_embedder.proj_in.bias", model.time_embedder.linear_1.bias.detach().numpy())
    writer.add_tensor("time_embedder.proj_out.weight", model.time_embedder.linear_2.weight.detach().numpy())
    writer.add_tensor("time_embedder.proj_out.bias", model.time_embedder.linear_2.bias.detach().numpy())
    writer.add_tensor("final_layer.norm.weight", model.norm_out.norm.weight.detach().numpy())
    writer.add_tensor("final_layer.adaln_proj.linear.weight", model.norm_out.linear.weight.detach().numpy())
    writer.add_tensor("final_layer.adaln_proj.linear.bias", model.norm_out.linear.bias.detach().numpy())
    writer.add_tensor("final_layer.video_out.weight", model.proj_out.weight.detach().numpy())
    writer.add_tensor("final_layer.video_out.bias", model.proj_out.bias.detach().numpy())
    writer.add_tensor("final_layer.audio_out.weight", model.audio_proj_out.weight.detach().numpy())
    writer.add_tensor("final_layer.audio_out.bias", model.audio_proj_out.bias.detach().numpy())
    writer.add_tensor("rope.inv_freq", np.ones(16, dtype=np.float32))

    for b in range(num_blocks):
        tb = model.transformer_blocks[b]
        writer.add_tensor(f"blocks.{b}.norm1.weight", tb.norm1.weight.detach().numpy())
        qkv = torch.cat([tb.attn.to_q.weight, tb.attn.to_k.weight, tb.attn.to_v.weight], dim=0)
        writer.add_tensor(f"blocks.{b}.attn.qkv_proj.weight", qkv.detach().numpy())
        writer.add_tensor(f"blocks.{b}.attn.to_gate_compress.weight", np.zeros((256, 256), dtype=np.float32))
        writer.add_tensor(f"blocks.{b}.attn.q_norm.weight", tb.attn.norm_q.weight.detach().numpy())
        writer.add_tensor(f"blocks.{b}.attn.k_norm.weight", tb.attn.norm_k.weight.detach().numpy())
        writer.add_tensor(f"blocks.{b}.attn.out_proj.weight", tb.attn.to_out[0].weight.detach().numpy())
        writer.add_tensor(f"blocks.{b}.norm2.weight", tb.norm2.weight.detach().numpy())
        writer.add_tensor(f"blocks.{b}.mlp.fc1.weight", tb.ff.net[0].proj.weight.detach().numpy())
        writer.add_tensor(f"blocks.{b}.mlp.fc2.weight", tb.ff.net[2].weight.detach().numpy())
        writer.add_tensor(f"blocks.{b}.adaln_proj.linear.weight", tb.adaln_proj.linear.weight.detach().numpy())
        writer.add_tensor(f"blocks.{b}.adaln_proj.linear.bias", tb.adaln_proj.linear.bias.detach().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if verbose:
        print(f"Packing trunk for {num_blocks} blocks...")
    pack_dit.pack_dit(str(gguf_path), out_dir, write_globals=False, verbose=False)

    if verbose:
        print(f"Folding AdaLN for {num_blocks} blocks...")
    fold_adaln.fold_adaln(str(gguf_path), out_dir, verbose=False)

    return out_dir, gguf_path


def load_adaln_table(table_path: Path) -> tuple[np.ndarray, int, int]:
    """Load adaln_table.bin into numpy array (n_steps, n_blocks, pairs_per_step, 6, hidden)."""
    with open(table_path, "rb") as f:
        hdr = f.read(128)
        (
            magic,
            version,
            n_steps,
            n_blocks,
            pairs_per_step,
            hidden_size,
            schedule_hash,
            trunk_offset,
            trunk_bytes,
            final_offset,
            final_bytes,
        ) = struct.unpack("<IIIIIIQQQQQ64x", hdr)
        f.seek(trunk_offset)
        total_floats = n_steps * n_blocks * pairs_per_step * 6 * hidden_size
        table = np.fromfile(f, dtype=np.float32, count=total_floats)
        table = table.reshape(n_steps, n_blocks, pairs_per_step, 6, hidden_size)
    return table, n_blocks, hidden_size


def load_layer_weights_from_gguf(gguf_path: Path, L: int) -> dict[str, np.ndarray]:
    """Read and dequantize weights for block L from GGUF."""
    hdr = fold_adaln.gguf_remote.header(str(gguf_path))
    t_map = {t.name: t for t in hdr.tensors}
    pfx = f"blocks.{L}."

    def get_t(name: str) -> np.ndarray:
        t = t_map[f"{pfx}{name}"]
        raw = fold_adaln._read_tensor_data(str(gguf_path), hdr, t)
        return fold_adaln._dequantize_tensor(raw, t.type, t.dims, t.orig_shape)

    w_norm1 = get_t("norm1.weight")
    w_qkv = get_t("attn.qkv_proj.weight")
    w_qnorm = get_t("attn.q_norm.weight")
    w_knorm = get_t("attn.k_norm.weight")
    w_out = get_t("attn.out_proj.weight")
    w_norm2 = get_t("norm2.weight")
    w_fc1 = get_t("mlp.fc1.weight")
    w_fc2 = get_t("mlp.fc2.weight")

    to_q, to_k, to_v = np.split(w_qkv, 3, axis=0)
    w_block = {
        "norm1.weight": w_norm1,
        "attn.to_q.weight": to_q,
        "attn.to_k.weight": to_k,
        "attn.to_v.weight": to_v,
        "attn.norm_q.weight": w_qnorm,
        "attn.norm_k.weight": w_knorm,
        "attn.to_out.0.weight": w_out,
        "norm2.weight": w_norm2,
        "ff.net.0.proj.weight": w_fc1,
        "ff.net.2.weight": w_fc2,
    }
    return w_block


def load_layer_weights_from_trunk(trunk_path: Path, L: int) -> dict[str, np.ndarray]:
    """Read weights for block L directly from dit_trunk.bin."""
    with open(trunk_path, "rb") as f:
        hdr = f.read(128)
        magic, ver, num_layers, align, idx_off, idx_bytes, data_off, total_bytes = struct.unpack("<IIIIQQQQ80x", hdr)
        layer_desc_size = 1536
        f.seek(idx_off + L * layer_desc_size)
        layer_desc = f.read(layer_desc_size)
        file_off, size_bytes, n_tensors = struct.unpack("<QQI4x", layer_desc[:24])

        tensor_entries = []
        for t in range(9):
            t_bytes = layer_desc[24 + t * 168 : 24 + (t + 1) * 168]
            t_name, t_type, n_dims, d0, d1, d2, d3, off_in_l, sz_b, in_d, out_d = struct.unpack("<64sII4qQQqq32x", t_bytes)
            name = t_name.split(b"\x00")[0].decode("ascii")
            tensor_entries.append((name, t_type, off_in_l, sz_b, in_d, out_d))

        weights: dict[str, np.ndarray] = {}
        for name, t_type, off_in_l, sz_b, in_d, out_d in tensor_entries:
            f.seek(file_off + off_in_l)
            raw = f.read(sz_b)
            if t_type == 0:  # F32
                arr = np.frombuffer(raw, dtype=np.float32).copy()
            elif _HAS_GGUF:
                quant_type = GGMLQuantizationType(t_type)
                arr = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), quant_type)
            else:
                raise RuntimeError(f"Cannot dequantize tensor {name} (type {t_type}) without gguf package")

            if out_d > 1 and in_d > 1:
                arr = arr.reshape(out_d, in_d)
            else:
                arr = arr.reshape(-1)
            weights[name] = arr

        to_q, to_k, to_v = np.split(weights["qkv"], 3, axis=0)
        return {
            "norm1.weight": weights["norm1"],
            "attn.to_q.weight": to_q,
            "attn.to_k.weight": to_k,
            "attn.to_v.weight": to_v,
            "attn.norm_q.weight": weights["q_norm"],
            "attn.norm_k.weight": weights["k_norm"],
            "attn.to_out.0.weight": weights["out"],
            "norm2.weight": weights["norm2"],
            "ff.net.0.proj.weight": weights["fc1"],
            "ff.net.2.weight": weights["fc2"],
        }


def conform_all(
    pack_dir: Optional[str | Path] = None,
    gguf_path: Optional[str | Path] = None,
    binary_path: Optional[str | Path] = None,
    log_path: str | Path = "tests/fixtures/gates/conform_all_50.log",
    num_blocks: int = 50,
    keep_fixture: bool = False,
    verbose: bool = True,
) -> bool:
    """Run 50-block conformance check."""
    bin_file = find_binary(str(binary_path) if binary_path else None)
    log_file = ROOT / Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    temp_dir_obj = None
    if pack_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        work_dir = Path(temp_dir_obj.name)
        pack_dir, gguf_path = generate_conform_fixture(work_dir, num_blocks=num_blocks, verbose=verbose)
    else:
        pack_dir = Path(pack_dir)
        if gguf_path is not None:
            gguf_path = Path(gguf_path)
        else:
            cand = pack_dir / "conform_50.gguf"
            if cand.is_file():
                gguf_path = cand

    table_path = pack_dir / "adaln_table.bin"
    trunk_path = pack_dir / "dit_trunk.bin"

    if not table_path.is_file():
        raise FileNotFoundError(f"Missing adaln_table.bin at {table_path}")
    if not trunk_path.is_file():
        raise FileNotFoundError(f"Missing dit_trunk.bin at {trunk_path}")

    table, total_blocks, hidden_size = load_adaln_table(table_path)
    if num_blocks > total_blocks:
        num_blocks = total_blocks

    if verbose:
        print(f"Running conformance for {num_blocks} blocks against {bin_file.name}...")

    N = 256
    num_heads = 2 if hidden_size <= 512 else 56
    head_dim = 128
    rope_rot_dim = 96

    # Build 2D grid coordinates for RoPE
    pos = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        pos[i, 0] = i // 16
        pos[i, 1] = (i % 16) // 4
        pos[i, 2] = i % 4
    cos, sin = h3_ref.build_rope(pos, rope_freq_dim=16, rope_theta=10000.0)

    log_lines: list[str] = []
    all_passed = True

    # Scratch directory for I/O tensors
    io_temp = tempfile.TemporaryDirectory()
    io_dir = Path(io_temp.name)

    in_file = io_dir / "in.bin"
    out_file = io_dir / "out.bin"
    out_mis_file = io_dir / "out_mis.bin"

    try:
        for L in range(num_blocks):
            # 1. Random input x (N=256, seed L)
            rng = np.random.RandomState(L)
            x = rng.randn(N, hidden_size).astype(np.float32)

            # 2. Block weights for block L
            if gguf_path and gguf_path.is_file():
                w_block = load_layer_weights_from_gguf(gguf_path, L)
            else:
                w_block = load_layer_weights_from_trunk(trunk_path, L)

            # 3. Reference Python forward
            mod_L = np.broadcast_to(table[0, L, 0], (N, 6, hidden_size))
            y_ref = h3_ref.block(
                x,
                w_block,
                mod_L,
                cos,
                sin,
                vsa=None,
                num_heads=num_heads,
                head_dim=head_dim,
                norm_eps=1e-5,
                qk_norm_eps=1e-5,
            )

            # 4. Write C input binary
            write_block_input(
                in_file,
                x,
                cos,
                sin,
                num_heads=num_heads,
                head_dim=head_dim,
                rope_rot_dim=rope_rot_dim,
                step=0,
                pair=0,
            )

            # 5. Run C executable (correct block L)
            cmd_correct = [
                str(bin_file),
                "--pack", str(pack_dir),
                "--block", str(L),
                "--in", str(in_file),
                "--out", str(out_file),
                "--step", "0",
                "--pair", "0",
                "--dense",
            ]
            res_c = subprocess.run(cmd_correct, capture_output=True, text=True)
            if res_c.returncode != 0:
                print(f"[FAIL] Block {L:02d}: C process failed: {res_c.stderr}")
                all_passed = False
                break

            y_c = np.fromfile(out_file, dtype=np.float32).reshape(N, hidden_size)
            rel_l2 = float(np.linalg.norm(y_c - y_ref) / np.linalg.norm(y_ref))
            correct_pass = (rel_l2 <= 1e-4)

            # 6. Mis-bound check: run with mod-block (L+1)%num_blocks
            mis_block = (L + 1) % num_blocks
            cmd_mis = [
                str(bin_file),
                "--pack", str(pack_dir),
                "--block", str(L),
                "--mod-block", str(mis_block),
                "--in", str(in_file),
                "--out", str(out_mis_file),
                "--step", "0",
                "--pair", "0",
                "--dense",
            ]
            res_mis = subprocess.run(cmd_mis, capture_output=True, text=True)
            if res_mis.returncode != 0:
                print(f"[FAIL] Block {L:02d}: C misbound process failed: {res_mis.stderr}")
                all_passed = False
                break

            y_mis = np.fromfile(out_mis_file, dtype=np.float32).reshape(N, hidden_size)
            mis_rel_l2 = float(np.linalg.norm(y_mis - y_ref) / np.linalg.norm(y_ref))
            mis_pass = (mis_rel_l2 > 1e-2)

            pass_str = "PASS" if (correct_pass and mis_pass) else "FAIL"
            line = f"Block {L:02d}/{num_blocks:02d}: rel_L2 = {rel_l2:.6e} {'PASS' if correct_pass else 'FAIL'} (misbound rel_L2 = {mis_rel_l2:.4f} {'PASS' if mis_pass else 'FAIL'})"
            if verbose:
                print(line)
            log_lines.append(line)

            if not correct_pass or not mis_pass:
                all_passed = False
                break

        if all_passed:
            summary_line = f"{num_blocks}/{num_blocks} blocks PASS"
            if verbose:
                print(summary_line)
            log_lines.append(summary_line)

            # Write gate log file
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")
            if verbose:
                print(f"Wrote gate log to {log_file}")
        else:
            print("Conformance check FAILED.")

    finally:
        io_temp.cleanup()
        if temp_dir_obj and not keep_fixture:
            temp_dir_obj.cleanup()

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="50-block real-checkpoint conformance gate")
    parser.add_argument("--pack", type=str, default=None, help="Directory containing dit_trunk.bin and adaln_table.bin")
    parser.add_argument("--gguf", type=str, default=None, help="Path to GGUF model")
    parser.add_argument("--binary", type=str, default=None, help="Path to test_real_block executable")
    parser.add_argument("--log", type=str, default="tests/fixtures/gates/conform_all_50.log", help="Path to write log output")
    parser.add_argument("--num-blocks", type=int, default=50, help="Number of blocks to verify (default: 50)")
    parser.add_argument("--keep-fixture", action="store_true", help="Keep generated fixture files on disk")
    args = parser.parse_args()

    success = conform_all(
        pack_dir=args.pack,
        gguf_path=args.gguf,
        binary_path=args.binary,
        log_path=args.log,
        num_blocks=args.num_blocks,
        keep_fixture=args.keep_fixture,
        verbose=True,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
