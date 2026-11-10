#!/usr/bin/env python3
"""Truncated Qwen3-VL Text Encoder Conformance Gate.

Verifies layers (e.g. 0, 25, 49) of the Qwen3-VL text encoder GGUF against
the pure NumPy reference implementation (tools.ref.qwen3_ref) and HuggingFace
Qwen3VLTextDecoderLayer, proving rel-L2 <= 1e-4.

Supports:
- Remote GGUF over HTTP range requests (default URL from tools.census)
- Local GGUF file
- Fast local tiny fixture (--tiny, testing layers 0, 1, 2)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    import gguf
    from gguf.constants import GGMLQuantizationType
    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import census, gguf_remote, hf_range
from tools.ref import qwen3_ref

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextConfig,
        Qwen3VLTextDecoderLayer,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


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


def compute_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 distance ||a - b||_2 / ||b||_2."""
    diff = np.linalg.norm((a - b).astype(np.float64))
    ref = np.linalg.norm(b.astype(np.float64))
    if ref == 0.0:
        return float(diff)
    return float(diff / ref)


def _read_tensor(
    src: str,
    header: gguf_remote.GgufHeader,
    t_info: gguf_remote.TensorInfo,
) -> np.ndarray:
    """Read and dequantize a tensor from GGUF."""
    abs_offset = header.data_offset + t_info.offset
    raw = hf_range.read(src, abs_offset, abs_offset + t_info.nbytes)

    if t_info.type == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    elif t_info.type == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif t_info.type == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif _HAS_GGUF:
        quant_type = getattr(GGMLQuantizationType, t_info.type, None)
        if quant_type is None:
            raise ValueError(f"Unknown quant type: {t_info.type}")
        u8 = np.frombuffer(raw, dtype=np.uint8)
        arr = gguf.quants.dequantize(u8, quant_type).astype(np.float32)
    else:
        raise RuntimeError(f"Cannot dequantize {t_info.type} without gguf library")

    # Shape restoration: GGUF dims is in reverse order [ne0, ne1, ...]
    if len(t_info.dims) == 1:
        arr = arr.reshape(t_info.dims[0])
    elif len(t_info.dims) == 2:
        arr = arr.reshape(t_info.dims[1], t_info.dims[0])
    return arr


def load_gguf_layer_weights(
    src: str,
    layer_idx: int,
    header: Optional[gguf_remote.GgufHeader] = None,
) -> tuple[dict[str, np.ndarray], gguf_remote.GgufHeader]:
    """Load and dequantize the 11 weights for layer L from GGUF."""
    if header is None:
        header = gguf_remote.header(src)
    tensor_map = {t.name: t for t in header.tensors}

    weights: dict[str, np.ndarray] = {}
    for canon_name, candidates in TE_LAYER_TENSORS:
        found_info: Optional[gguf_remote.TensorInfo] = None
        for cand in candidates:
            cand_name = cand.format(L=layer_idx)
            if cand_name in tensor_map:
                found_info = tensor_map[cand_name]
                break
        if found_info is None:
            raise KeyError(f"Could not find tensor for {canon_name} at layer {layer_idx}")
        weights[canon_name] = _read_tensor(src, header, found_info)

    return weights, header


def conform_layer(
    layer_idx: int,
    weights: Mapping[str, np.ndarray],
    cfg: qwen3_ref.Qwen3TeConfig,
    seq_len: int = 4,
    seed: int = 42,
    tol: float = 1e-4,
    verbose: bool = True,
) -> float:
    """Verify that NumPy reference and HuggingFace layer produce identical outputs.

    Args:
        layer_idx: Index of layer.
        weights: Dictionary of 11 weight tensors.
        cfg: Qwen3TeConfig.
        seq_len: Token sequence length.
        seed: Random seed for test input.
        tol: Tolerance for relative L2 distance.
        verbose: Print comparison details.

    Returns:
        Computed relative L2 distance.
    """
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers package is required for conform_layer")

    rng = np.random.default_rng(seed + layer_idx)
    x_np = rng.standard_normal((seq_len, cfg.hidden_size)).astype(np.float32)

    # 1. NumPy reference forward
    cos, sin = qwen3_ref.build_rope_1d(np.arange(seq_len), cfg.head_dim, cfg.rope_theta)
    ref_out = qwen3_ref.qwen3_layer(x_np, weights, cfg, cos, sin)

    # 2. HuggingFace forward
    hf_cfg = Qwen3VLTextConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.norm_eps,
    )
    hf_layer = Qwen3VLTextDecoderLayer(hf_cfg, layer_idx=layer_idx)

    # Assign weights to HF layer
    with torch.no_grad():
        hf_layer.input_layernorm.weight.copy_(torch.from_numpy(weights["input_layernorm.weight"]))
        hf_layer.self_attn.q_proj.weight.copy_(torch.from_numpy(weights["q_proj.weight"]))
        hf_layer.self_attn.k_proj.weight.copy_(torch.from_numpy(weights["k_proj.weight"]))
        hf_layer.self_attn.v_proj.weight.copy_(torch.from_numpy(weights["v_proj.weight"]))
        hf_layer.self_attn.q_norm.weight.copy_(torch.from_numpy(weights["q_norm.weight"]))
        hf_layer.self_attn.k_norm.weight.copy_(torch.from_numpy(weights["k_norm.weight"]))
        hf_layer.self_attn.o_proj.weight.copy_(torch.from_numpy(weights["o_proj.weight"]))
        hf_layer.post_attention_layernorm.weight.copy_(torch.from_numpy(weights["post_attention_layernorm.weight"]))
        hf_layer.mlp.gate_proj.weight.copy_(torch.from_numpy(weights["gate_proj.weight"]))
        hf_layer.mlp.up_proj.weight.copy_(torch.from_numpy(weights["up_proj.weight"]))
        hf_layer.mlp.down_proj.weight.copy_(torch.from_numpy(weights["down_proj.weight"]))

    x_torch = torch.from_numpy(x_np).unsqueeze(0)  # (1, T, hidden)
    cos_t = torch.from_numpy(cos).unsqueeze(0)
    sin_t = torch.from_numpy(sin).unsqueeze(0)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    hf_layer.eval()
    with torch.no_grad():
        hf_out = hf_layer(
            x_torch,
            attention_mask=causal_mask,
            position_embeddings=(cos_t, sin_t),
        )[0].squeeze(0).numpy()

    rel_l2 = compute_rel_l2(ref_out, hf_out)
    if verbose:
        print(f"Layer {layer_idx:2d}: rel-L2(NumPy, HF) = {rel_l2:e} (tol: {tol:e}) -> {'PASS' if rel_l2 <= tol else 'FAIL'}")

    assert rel_l2 <= tol, f"Layer {layer_idx} conformance failed: rel-L2 = {rel_l2:e} > {tol:e}"
    return rel_l2


def conform_te(
    src: Optional[str] = None,
    layers: Optional[Sequence[int]] = None,
    tiny: bool = False,
    tol: float = 1e-4,
    seq_len: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> dict[int, float]:
    """Run text encoder conformance check on specified layers.

    Args:
        src: Path or URL to GGUF model.
        layers: Layer indices to verify.
        tiny: Use local tiny fixture if True.
        tol: Relative L2 tolerance threshold.
        seq_len: Token sequence length.
        seed: Random seed.
        verbose: Verbose printing.

    Returns:
        Mapping from layer_idx to computed rel-L2 distance.
    """
    if tiny or (src is None and os.environ.get("FASTH3_SKIP_REMOTE_TESTS") == "1"):
        tiny_gguf = ROOT / "tests" / "fixtures" / "tiny_te" / "tiny_te.gguf"
        if not tiny_gguf.is_file():
            if verbose:
                print("Tiny GGUF fixture not found, generating...")
            from tools.pack_te import make_tiny_te
            make_tiny_te(ROOT / "tests" / "fixtures" / "tiny_te", verbose=verbose)
        src = str(tiny_gguf)
        if layers is None:
            layers = [0, 1, 2]
        cfg = qwen3_ref.tiny_te_config()
    else:
        if src is None:
            src = os.environ.get("FASTH3_TE_GGUF", census.DEFAULT_TE_URL)
        if layers is None:
            layers = [0, 25, 49]
        cfg = qwen3_ref.Qwen3TeConfig()

    if verbose:
        print(f"--- Conformance Check: Truncated Text Encoder ---")
        print(f"Source: {src}")
        print(f"Layers to verify: {list(layers)}")

    header = gguf_remote.header(src)
    results: dict[int, float] = {}

    for L in layers:
        weights, _ = load_gguf_layer_weights(src, L, header=header)
        # Adapt config to weights if running on non-standard model
        hidden = weights["input_layernorm.weight"].shape[0]
        q_heads = weights["q_norm.weight"].shape[0] // 128 if weights["q_norm.weight"].ndim == 1 and weights["q_norm.weight"].shape[0] % 128 == 0 else cfg.num_heads
        head_dim = weights["q_norm.weight"].shape[0] // cfg.num_heads if weights["q_norm.weight"].shape[0] % cfg.num_heads == 0 else cfg.head_dim
        intermediate = weights["gate_proj.weight"].shape[0]

        layer_cfg = qwen3_ref.Qwen3TeConfig(
            hidden_size=hidden,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
            intermediate_size=intermediate,
            norm_eps=cfg.norm_eps,
            qk_norm_eps=cfg.qk_norm_eps,
            rope_theta=cfg.rope_theta,
        )

        rel_l2 = conform_layer(
            layer_idx=L,
            weights=weights,
            cfg=layer_cfg,
            seq_len=seq_len,
            seed=seed,
            tol=tol,
            verbose=verbose,
        )
        results[L] = rel_l2

    if verbose:
        print(f"All {len(layers)} tested layers conform (rel-L2 <= {tol:e}).")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Truncated Qwen3-VL Text Encoder Conformance Gate")
    parser.add_argument("--src", type=str, default=None, help="GGUF path or URL (default: census.DEFAULT_TE_URL)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to verify (default: 0 25 49, or 0 1 2 for tiny)")
    parser.add_argument("--tiny", action="store_true", help="Run against local tiny fixture")
    parser.add_argument("--tol", type=float, default=1e-4, help="Relative L2 tolerance (default: 1e-4)")
    parser.add_argument("--seq-len", type=int, default=4, help="Sequence length (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    try:
        conform_te(
            src=args.src,
            layers=args.layers,
            tiny=args.tiny,
            tol=args.tol,
            seq_len=args.seq_len,
            seed=args.seed,
            verbose=True,
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Conformance check failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
