#!/usr/bin/env python3
"""Generate a synthetic tiny GGUF fixture and pack it into FH3P / FH3A fixtures.

Creates:
- tests/fixtures/pack/tiny_dit.gguf
- tests/fixtures/pack/tiny_trunk.bin
- tests/fixtures/pack/tiny_globals.bin
- tests/fixtures/pack/tiny_adaln.bin
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

try:
    import gguf
    from gguf.constants import GGMLQuantizationType
except ImportError:
    sys.exit("Error: 'gguf' Python package not found. Run in project .venv.")

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools import fold_adaln, pack_dit


def make_tiny_pack_fixtures(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = dest_dir / "tiny_dit.gguf"

    writer = gguf.GGUFWriter(str(gguf_path), "wan")
    writer.add_uint32("general.alignment", 32)

    hidden = 64
    inner = 64
    ffn = 128
    time_emb_dim = 16
    freq_dim = 16
    num_layers = 2

    # Deterministic pseudo-random pattern
    rng = np.random.RandomState(42)

    # 1. Globals
    # Time embedder
    w_tin = (rng.randn(32, freq_dim) * 0.1).astype(np.float32)
    b_tin = (rng.randn(32) * 0.1).astype(np.float32)
    w_tout = (rng.randn(time_emb_dim, 32) * 0.1).astype(np.float32)
    b_tout = (rng.randn(time_emb_dim) * 0.1).astype(np.float32)

    writer.add_tensor("time_embedder.proj_in.weight", w_tin)
    writer.add_tensor("time_embedder.proj_in.bias", b_tin)
    writer.add_tensor("time_embedder.proj_out.weight", w_tout)
    writer.add_tensor("time_embedder.proj_out.bias", b_tout)

    # Final layer
    w_f_adaln = (rng.randn(2 * hidden, time_emb_dim) * 0.1).astype(np.float32)
    b_f_adaln = (rng.randn(2 * hidden) * 0.1).astype(np.float32)
    w_f_norm = np.ones(hidden, dtype=np.float32)

    writer.add_tensor("final_layer.adaln_proj.linear.weight", w_f_adaln)
    writer.add_tensor("final_layer.adaln_proj.linear.bias", b_f_adaln)
    writer.add_tensor("final_layer.norm.weight", w_f_norm)
    writer.add_tensor("rope.inv_freq", np.arange(16, dtype=np.float32))
    writer.add_tensor("token_refiner.final_norm.weight", np.ones(hidden, dtype=np.float32))

    # 2. Block layers
    for L in range(num_layers):
        pfx = f"blocks.{L}."
        # Trunk tensors (9 per layer)
        writer.add_tensor(f"{pfx}attn.qkv_proj.weight", (rng.randn(3 * inner, hidden) * 0.1).astype(np.float32))
        writer.add_tensor(f"{pfx}attn.to_gate_compress.weight", (rng.randn(inner, hidden) * 0.1).astype(np.float32))
        writer.add_tensor(f"{pfx}attn.out_proj.weight", (rng.randn(hidden, inner) * 0.1).astype(np.float32))
        writer.add_tensor(f"{pfx}mlp.fc1.weight", (rng.randn(ffn, hidden) * 0.1).astype(np.float32))
        writer.add_tensor(f"{pfx}mlp.fc2.weight", (rng.randn(hidden, ffn // 2) * 0.1).astype(np.float32))
        writer.add_tensor(f"{pfx}norm1.weight", np.ones(hidden, dtype=np.float32))
        writer.add_tensor(f"{pfx}norm2.weight", np.ones(hidden, dtype=np.float32))
        writer.add_tensor(f"{pfx}attn.q_norm.weight", np.ones(hidden, dtype=np.float32))
        writer.add_tensor(f"{pfx}attn.k_norm.weight", np.ones(hidden, dtype=np.float32))

        # AdaLN tensors (3 modalities * 6 chunks * hidden = 18 * hidden = 1152)
        adaln_out = 3 * 6 * hidden
        w_adaln = (rng.randn(adaln_out, time_emb_dim) * 0.1).astype(np.float32)
        b_adaln = (rng.randn(adaln_out) * 0.1).astype(np.float32)
        writer.add_tensor(f"{pfx}adaln_proj.linear.weight", w_adaln)
        writer.add_tensor(f"{pfx}adaln_proj.linear.bias", b_adaln)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Generated GGUF fixture: {gguf_path} ({gguf_path.stat().st_size} bytes)")

    # 3. Pack trunk and fold AdaLN
    pack_dit.pack_dit(
        src=str(gguf_path),
        out_dir=dest_dir,
        layers=range(num_layers),
        write_globals=True,
        verbose=False,
    )

    # Rename dit_trunk.bin / dit_globals.bin to tiny_trunk.bin / tiny_globals.bin if needed,
    # or keep them as tiny_trunk.bin
    trunk_out = dest_dir / "dit_trunk.bin"
    tiny_trunk = dest_dir / "tiny_trunk.bin"
    if trunk_out.exists():
        if tiny_trunk.exists():
            tiny_trunk.unlink()
        trunk_out.rename(tiny_trunk)

    globals_out = dest_dir / "dit_globals.bin"
    tiny_globals = dest_dir / "tiny_globals.bin"
    if globals_out.exists():
        if tiny_globals.exists():
            tiny_globals.unlink()
        globals_out.rename(tiny_globals)

    fold_adaln.fold_adaln(
        src=str(gguf_path),
        out_dir=dest_dir,
        layers=range(num_layers),
        verbose=False,
    )
    adaln_out = dest_dir / "adaln_table.bin"
    tiny_adaln = dest_dir / "tiny_adaln.bin"
    if adaln_out.exists():
        if tiny_adaln.exists():
            tiny_adaln.unlink()
        adaln_out.rename(tiny_adaln)

    print(f"Pack fixtures generated in {dest_dir}:")
    print(f"  tiny_trunk.bin:  {tiny_trunk.stat().st_size} bytes")
    print(f"  tiny_globals.bin:{tiny_globals.stat().st_size} bytes")
    print(f"  tiny_adaln.bin:  {tiny_adaln.stat().st_size} bytes")


if __name__ == "__main__":
    dest = ROOT / "tests" / "fixtures" / "pack"
    make_tiny_pack_fixtures(dest)
