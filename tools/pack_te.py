#!/usr/bin/env python3
"""Repack Qwen3-VL text encoder from GGUF into truncated layer runs and embedding table.

FastH3 conditions on text embeddings from Qwen3-VL-32B, but ONLY layers 0..49
are used. MiniMax-H3 takes the pre-final-norm hidden state after layer 49 (T x 5120).
Layers 50..63, model.norm, and lm_head are NEVER fetched and are dropped.
The vision tower (visual.*) is also dropped.

Writes:
1. te_trunk.bin: header + N 4096-aligned layer runs (layers 0..stop_layer-1).
   Each run contains 11 tensors:
     0: input_layernorm.weight
     1: q_proj.weight
     2: k_proj.weight
     3: v_proj.weight
     4: q_norm.weight
     5: k_norm.weight
     6: o_proj.weight
     7: post_attention_layernorm.weight
     8: gate_proj.weight
     9: up_proj.weight
     10: down_proj.weight
   Each tensor is 32-byte aligned; the run is padded to 4096 bytes.
2. te_embed.bin: model.embed_tokens.weight extracted with a 64-byte header so
   individual rows can be fetched on demand via pread per token ID.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import gguf
    from gguf.constants import GGMLQuantizationType
    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False

try:
    from . import gguf_remote, hf_range
    from .ref import qwen3_ref
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import gguf_remote, hf_range
    from tools.ref import qwen3_ref


FH3_PACK_MAGIC = 0x50334846  # 'FH3P'
FH3_PACK_VERSION = 1
FH3_PACK_ALIGNMENT = 4096
FH3_TE_TENSORS_PER_LAYER = 11

FH3_EMBED_MAGIC = 0x45334846  # 'FH3E'
FH3_EMBED_VERSION = 1
FH3_EMBED_HEADER_SIZE = 64

QUANT_TYPE_MAP = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q8_K": 15,
    "BF16": 30,
}

TE_LAYER_TENSORS = [
    ("input_layernorm.weight", ["model.layers.{L}.input_layernorm.weight", "blk.{L}.attn_norm.weight"]),
    ("q_proj.weight", ["model.layers.{L}.self_attn.q_proj.weight", "blk.{L}.attn_q.weight"]),
    ("k_proj.weight", ["model.layers.{L}.self_attn.k_proj.weight", "blk.{L}.attn_k.weight"]),
    ("v_proj.weight", ["model.layers.{L}.self_attn.v_proj.weight", "blk.{L}.attn_v.weight"]),
    ("q_norm.weight", ["model.layers.{L}.self_attn.q_norm.weight", "blk.{L}.attn_q_norm.weight"]),
    ("k_norm.weight", ["model.layers.{L}.self_attn.k_norm.weight", "blk.{L}.attn_k_norm.weight"]),
    ("o_proj.weight", ["model.layers.{L}.self_attn.o_proj.weight", "blk.{L}.attn_output.weight"]),
    ("post_attention_layernorm.weight", ["model.layers.{L}.post_attention_layernorm.weight", "blk.{L}.ffn_norm.weight"]),
    ("gate_proj.weight", ["model.layers.{L}.mlp.gate_proj.weight", "blk.{L}.ffn_gate.weight"]),
    ("up_proj.weight", ["model.layers.{L}.mlp.up_proj.weight", "blk.{L}.ffn_up.weight"]),
    ("down_proj.weight", ["model.layers.{L}.mlp.down_proj.weight", "blk.{L}.ffn_down.weight"]),
]

EMBED_TENSOR_NAMES = [
    "model.embed_tokens.weight",
    "token_embd.weight",
]


def _read_tensor_bytes(src: str, header: gguf_remote.GgufHeader, t: gguf_remote.TensorInfo) -> bytes:
    abs_offset = header.data_offset + t.offset
    return hf_range.read(src, abs_offset, abs_offset + t.nbytes)


def pack_te(
    src: str,
    out_dir: str | Path,
    stop_layer: int = 50,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Pack truncated text encoder layers into te_trunk.bin and embeddings into te_embed.bin."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trunk_path = out_dir / "te_trunk.bin"
    embed_path = out_dir / "te_embed.bin"

    if verbose:
        print(f"Parsing GGUF header from {src}...")
    hdr = gguf_remote.header(src)
    tensor_map = {t.name: t for t in hdr.tensors}

    # 1. Pack embedding table (model.embed_tokens.weight)
    embed_info: Optional[gguf_remote.TensorInfo] = None
    for name in EMBED_TENSOR_NAMES:
        if name in tensor_map:
            embed_info = tensor_map[name]
            break
    if embed_info is None:
        raise ValueError(f"Could not find embed_tokens tensor in GGUF. Looked for {EMBED_TENSOR_NAMES}")

    if verbose:
        print(f"Reading embedding table ({embed_info.name}, {embed_info.nbytes / 1e6:.1f} MB)...")
    embed_bytes = _read_tensor_bytes(src, hdr, embed_info)

    type_id = QUANT_TYPE_MAP.get(embed_info.type, 0)
    # GGUF dims: [ne0, ne1] where ne0 is hidden_size, ne1 is vocab_size
    hidden_size = embed_info.dims[0] if len(embed_info.dims) >= 1 else 5120
    vocab_size = embed_info.dims[1] if len(embed_info.dims) >= 2 else (len(embed_bytes) // (hidden_size * 4))
    row_bytes = len(embed_bytes) // vocab_size if vocab_size > 0 else (hidden_size * 4)

    embed_hdr = struct.pack(
        "<IIIIIIQ32s",
        FH3_EMBED_MAGIC,
        FH3_EMBED_VERSION,
        type_id,
        vocab_size,
        hidden_size,
        row_bytes,
        FH3_EMBED_HEADER_SIZE,
        b"\x00" * 32,
    )
    assert len(embed_hdr) == FH3_EMBED_HEADER_SIZE

    with open(embed_path, "wb") as f_emb:
        f_emb.write(embed_hdr)
        f_emb.write(embed_bytes)

    if verbose:
        print(f"Wrote {embed_path} ({len(embed_hdr) + len(embed_bytes):,} bytes, vocab={vocab_size}, hidden={hidden_size})")

    # 2. Discover available layers
    discovered_layers: list[int] = []
    for L in range(stop_layer):
        found_all = True
        for _, candidates in TE_LAYER_TENSORS:
            if not any(c.format(L=L) in tensor_map for c in candidates):
                found_all = False
                break
        if found_all:
            discovered_layers.append(L)
        else:
            break

    num_layers = len(discovered_layers)
    if num_layers == 0:
        raise ValueError("No complete text encoder layers found in GGUF.")

    if verbose:
        print(f"Packing {num_layers} layers (stopping at layer {num_layers})...")

    # Calculate trunk layout
    header_size = 128
    layer_desc_size = 24 + FH3_TE_TENSORS_PER_LAYER * 168
    index_offset = header_size
    index_bytes = num_layers * layer_desc_size
    data_offset = (header_size + index_bytes + FH3_PACK_ALIGNMENT - 1) & ~(FH3_PACK_ALIGNMENT - 1)

    layer_descs_data = bytearray()
    cur_file_offset = data_offset
    max_run_bytes = 0

    with open(trunk_path, "wb") as f_trunk:
        f_trunk.seek(data_offset)

        for L in range(num_layers):
            layer_chunk = bytearray()
            tensor_desc_list: list[bytes] = []

            for short_name, candidates in TE_LAYER_TENSORS:
                t_info = None
                for c in candidates:
                    cand_name = c.format(L=L)
                    if cand_name in tensor_map:
                        t_info = tensor_map[cand_name]
                        break
                assert t_info is not None, f"Missing tensor for layer {L}: {candidates}"

                # 32-byte alignment within layer chunk
                cur_len = len(layer_chunk)
                pad_32 = (32 - (cur_len % 32)) % 32
                if pad_32 > 0:
                    layer_chunk.extend(b"\x00" * pad_32)

                offset_in_layer = len(layer_chunk)
                assert offset_in_layer % 32 == 0

                raw_bytes = _read_tensor_bytes(src, hdr, t_info)
                layer_chunk.extend(raw_bytes)
                size_bytes = len(raw_bytes)

                type_id = QUANT_TYPE_MAP.get(t_info.type, 0)
                n_dims = len(t_info.dims)
                shape = [0, 0, 0, 0]
                for d_i, d in enumerate(t_info.dims[:4]):
                    shape[d_i] = d

                if n_dims >= 2:
                    in_dim = t_info.dims[0]
                    out_dim = t_info.dims[1]
                elif n_dims == 1:
                    in_dim = t_info.dims[0]
                    out_dim = 1
                else:
                    in_dim = 1
                    out_dim = 1

                t_name_bytes = short_name.encode("utf-8")[:63]
                t_desc_bytes = struct.pack(
                    "<64sII4qQQqq32s",
                    t_name_bytes,
                    type_id,
                    n_dims,
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[3],
                    offset_in_layer,
                    size_bytes,
                    in_dim,
                    out_dim,
                    b"\x00" * 32,
                )
                assert len(t_desc_bytes) == 168
                tensor_desc_list.append(t_desc_bytes)

            run_unpadded = len(layer_chunk)
            pad_4096 = (FH3_PACK_ALIGNMENT - (run_unpadded % FH3_PACK_ALIGNMENT)) % FH3_PACK_ALIGNMENT
            if pad_4096 > 0:
                layer_chunk.extend(b"\x00" * pad_4096)

            run_size = len(layer_chunk)
            assert run_size % FH3_PACK_ALIGNMENT == 0
            if run_size > max_run_bytes:
                max_run_bytes = run_size

            l_desc_bytes = struct.pack(
                "<QQII",
                cur_file_offset,
                run_size,
                FH3_TE_TENSORS_PER_LAYER,
                0,
            ) + b"".join(tensor_desc_list)
            assert len(l_desc_bytes) == layer_desc_size
            layer_descs_data.extend(l_desc_bytes)

            f_trunk.write(layer_chunk)
            cur_file_offset += run_size

        total_file_bytes = cur_file_offset

        # Seek to beginning and write header & index table
        f_trunk.seek(0)
        hdr_bytes = struct.pack(
            "<IIIIQQQQ80s",
            FH3_PACK_MAGIC,
            FH3_PACK_VERSION,
            num_layers,
            FH3_PACK_ALIGNMENT,
            index_offset,
            index_bytes,
            data_offset,
            total_file_bytes,
            b"\x00" * 80,
        )
        assert len(hdr_bytes) == header_size
        f_trunk.write(hdr_bytes)
        f_trunk.write(layer_descs_data)

    if verbose:
        print(f"Wrote {trunk_path} ({total_file_bytes / 1e9:.3f} GB, {num_layers} layers)")

    return trunk_path, embed_path


def make_tiny_te(
    out_dir: str | Path,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 32,
    num_layers: int = 4,
    stop_layer: int = 3,
    intermediate_size: int = 256,
    vocab_size: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate tiny Qwen3-VL text encoder GGUF, packed binaries, and golden vectors."""
    if not _HAS_GGUF:
        raise RuntimeError("'gguf' package is required to generate tiny_te fixture.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # 1. Initialize weights
    embed_tokens = rng.randn(vocab_size, hidden_size).astype(np.float32) * 0.02
    layers_weights: list[dict[str, np.ndarray]] = []

    for _ in range(num_layers):
        w: dict[str, np.ndarray] = {
            "input_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
            "q_proj.weight": rng.randn(num_heads * head_dim, hidden_size).astype(np.float32) * 0.02,
            "k_proj.weight": rng.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32) * 0.02,
            "v_proj.weight": rng.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32) * 0.02,
            "q_norm.weight": np.ones(head_dim, dtype=np.float32),
            "k_norm.weight": np.ones(head_dim, dtype=np.float32),
            "o_proj.weight": rng.randn(hidden_size, num_heads * head_dim).astype(np.float32) * 0.02,
            "post_attention_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
            "gate_proj.weight": rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02,
            "up_proj.weight": rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02,
            "down_proj.weight": rng.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02,
        }
        layers_weights.append(w)

    final_norm = np.ones(hidden_size, dtype=np.float32)

    # 2. Write tiny GGUF
    gguf_path = out_dir / "tiny_te.gguf"
    writer = gguf.GGUFWriter(str(gguf_path), "qwen3vl")
    writer.add_uint32("qwen3vl.embedding_length", hidden_size)
    writer.add_uint32("qwen3vl.block_count", num_layers)
    writer.add_uint32("qwen3vl.feed_forward_length", intermediate_size)
    writer.add_uint32("qwen3vl.attention.head_count", num_heads)
    writer.add_uint32("qwen3vl.attention.head_count_kv", num_kv_heads)
    writer.add_float32("qwen3vl.attention.layer_norm_rms_epsilon", 1e-6)

    writer.add_tensor("model.embed_tokens.weight", embed_tokens)
    for L, lw in enumerate(layers_weights):
        writer.add_tensor(f"model.layers.{L}.input_layernorm.weight", lw["input_layernorm.weight"])
        writer.add_tensor(f"model.layers.{L}.self_attn.q_proj.weight", lw["q_proj.weight"])
        writer.add_tensor(f"model.layers.{L}.self_attn.k_proj.weight", lw["k_proj.weight"])
        writer.add_tensor(f"model.layers.{L}.self_attn.v_proj.weight", lw["v_proj.weight"])
        writer.add_tensor(f"model.layers.{L}.self_attn.q_norm.weight", lw["q_norm.weight"])
        writer.add_tensor(f"model.layers.{L}.self_attn.k_norm.weight", lw["k_norm.weight"])
        writer.add_tensor(f"model.layers.{L}.self_attn.o_proj.weight", lw["o_proj.weight"])
        writer.add_tensor(f"model.layers.{L}.post_attention_layernorm.weight", lw["post_attention_layernorm.weight"])
        writer.add_tensor(f"model.layers.{L}.mlp.gate_proj.weight", lw["gate_proj.weight"])
        writer.add_tensor(f"model.layers.{L}.mlp.up_proj.weight", lw["up_proj.weight"])
        writer.add_tensor(f"model.layers.{L}.mlp.down_proj.weight", lw["down_proj.weight"])

    writer.add_tensor("model.norm.weight", final_norm)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # 3. Pack trunk and embedding table
    trunk_path, embed_path = pack_te(str(gguf_path), out_dir, stop_layer=stop_layer, verbose=False)

    # 4. Generate golden test vectors
    prompt_ids = np.array([42, 108, 256, 789, 15], dtype=np.int32)
    cfg = qwen3_ref.Qwen3TeConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        norm_eps=1e-6,
        qk_norm_eps=1e-6,
        rope_theta=5000000.0,
    )

    golden_prenorm = qwen3_ref.qwen3_forward(
        prompt_ids, embed_tokens, layers_weights, cfg, stop_layer=stop_layer
    )
    golden_normed = qwen3_ref.rmsnorm(golden_prenorm, final_norm, 1e-6)

    prompt_ids_path = out_dir / "prompt_ids.bin"
    golden_prenorm_path = out_dir / "golden_prenorm.bin"
    golden_normed_path = out_dir / "golden_normed.bin"
    config_json_path = out_dir / "te_config.json"

    prompt_ids_path.write_bytes(prompt_ids.tobytes())
    golden_prenorm_path.write_bytes(golden_prenorm.astype(np.float32).tobytes())
    golden_normed_path.write_bytes(golden_normed.astype(np.float32).tobytes())

    meta = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_layers": stop_layer,
        "total_model_layers": num_layers,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "norm_eps": 1e-6,
        "qk_norm_eps": 1e-6,
        "rope_theta": 5000000.0,
        "num_tokens": len(prompt_ids),
        "prompt_ids": prompt_ids.tolist(),
    }
    config_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack Qwen3-VL text encoder for FastH3")
    parser.add_argument("--src", type=str, required=True, help="Path or URL to Qwen3-VL GGUF")
    parser.add_argument("--out", type=str, default="packed", help="Output directory")
    parser.add_argument("--stop-layer", type=int, default=50, help="Stop layer index (default: 50)")
    parser.add_argument("--make-tiny", action="store_true", help="Generate tiny fixture in output dir")
    args = parser.parse_args()

    if args.make_tiny:
        print(f"Generating tiny Qwen3-VL text encoder fixture in {args.out}...")
        meta = make_tiny_te(args.out)
        print(f"Tiny fixture generated: {meta}")
        return

    pack_te(args.src, args.out, stop_layer=args.stop_layer, verbose=True)


if __name__ == "__main__":
    main()
