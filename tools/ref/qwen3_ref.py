"""Qwen3-VL Text Encoder: Pure NumPy reference implementation and verification helpers.

FastH3 conditions on text embeddings from Qwen3-VL-32B, but ONLY layers 0..49
are used. MiniMax-H3 takes the pre-final-norm hidden state after layer 49 (T x 5120).
Layers 50..63, model.norm, and lm_head are never needed and are dropped.
The vision tower (visual.*) is also dropped.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Qwen3TeConfig:
    """Configuration for truncated Qwen3-VL text encoder."""

    hidden_size: int = 5120
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    num_layers: int = 50
    intermediate_size: int = 25600
    norm_eps: float = 1e-6
    qk_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0


def tiny_te_config() -> Qwen3TeConfig:
    """Tiny Qwen3-VL text encoder config for fast unit testing."""
    return Qwen3TeConfig(
        hidden_size=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        num_layers=3,  # stop at 3 (out of 4 in fixture)
        intermediate_size=256,
        norm_eps=1e-6,
        qk_norm_eps=1e-6,
        rope_theta=5000000.0,
    )


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm along the last dimension.

    Args:
        x: Input array of shape (..., dim).
        weight: Scale parameter of shape (dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized array of shape (..., dim).
    """
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    normed = x.astype(np.float32) / np.sqrt(var + eps)
    return (normed * weight.astype(np.float32)).astype(x.dtype)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


def build_rope_1d(
    positions: np.ndarray,
    head_dim: int,
    rope_theta: float = 5000000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D Rotary Position Embedding (RoPE) cos and sin tables.

    Args:
        positions: 1-D array of token positions (length T).
        head_dim: Channel dimension per attention head.
        rope_theta: Base frequency (default 5e6 for Qwen3-VL).

    Returns:
        Tuple of (cos, sin) each with shape (T, head_dim).
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.asarray(positions, dtype=np.float32)
    freqs = pos[:, None] * inv_freq[None, :]  # (T, half_dim)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    # Qwen-style rotate_half uses duplicate cos/sin concatenated: [cos, cos]
    cos = np.concatenate([cos, cos], axis=-1).astype(np.float32)
    sin = np.concatenate([sin, sin], axis=-1).astype(np.float32)
    return cos, sin


def build_rope_3d_text(
    positions: np.ndarray,
    head_dim: int,
    rope_theta: float = 5000000.0,
    mrope_section: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """3-axis M-RoPE for text tokens where all three axes (T, H, W) equal position.

    In Qwen3-VL, text tokens are assigned identical position coordinates along
    temporal, height, and width axes: (pos, pos, pos).
    This function explicitly executes the 3-axis M-RoPE algorithm to demonstrate
    mathematical equivalence with 1-D RoPE.
    """
    if mrope_section is None:
        # Default partition: sum(mrope_section) * 2 == head_dim
        # E.g., for head_dim=128: [24, 20, 20] -> (24+20+20)*2 = 128.
        # For general head_dim:
        s0 = head_dim // 6 + (head_dim // 2) % 3
        s1 = head_dim // 6
        s2 = head_dim // 6
        # Adjust so sum(mrope_section) == head_dim // 2
        diff = (head_dim // 2) - (s0 + s1 + s2)
        s0 += diff
        mrope_section = [s0, s1, s2]

    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.asarray(positions, dtype=np.float32)

    # 3 grids: T, H, W, all having identical position IDs
    # freqs shape: (3, T, half_dim)
    freqs_3 = np.stack([pos[:, None] * inv_freq[None, :]] * 3, axis=0)

    # Recomposition frequencies matching Qwen3-VL recomposition_frequencies
    freqs_thw = freqs_3[0].copy()  # (T, half_dim)
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_thw[:, idx] = freqs_3[dim, :, idx]

    cos = np.cos(freqs_thw)
    sin = np.sin(freqs_thw)
    cos = np.concatenate([cos, cos], axis=-1).astype(np.float32)
    sin = np.concatenate([sin, sin], axis=-1).astype(np.float32)
    return cos, sin


def verify_mrope_text_eq_1d(
    positions: np.ndarray,
    head_dim: int,
    rope_theta: float = 5000000.0,
    mrope_section: Optional[Sequence[int]] = None,
) -> bool:
    """Verify that 3D text M-RoPE on (pos, pos, pos) is mathematically identical to 1-D RoPE.

    Returns:
        True if max absolute difference between cos and sin is <= 1e-6.
    """
    cos_1d, sin_1d = build_rope_1d(positions, head_dim, rope_theta)
    cos_3d, sin_3d = build_rope_3d_text(positions, head_dim, rope_theta, mrope_section)
    diff_cos = float(np.max(np.abs(cos_1d - cos_3d)))
    diff_sin = float(np.max(np.abs(sin_1d - sin_3d)))
    return diff_cos <= 1e-6 and diff_sin <= 1e-6


def apply_rotary_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply rotary position embedding using HuggingFace / Qwen style rotate_half.

    Args:
        x: Input tensor of shape (T, heads, head_dim).
        cos: Cosine table of shape (T, head_dim).
        sin: Sine table of shape (T, head_dim).

    Returns:
        Rotated tensor of shape (T, heads, head_dim).
    """
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rx = np.concatenate([-x2, x1], axis=-1)
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    return x * cos_b + rx * sin_b


def gqa_causal_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_heads: int,
    num_kv_heads: int,
) -> np.ndarray:
    """Grouped-Query Causal Self-Attention.

    Args:
        q: Query tensor of shape (T, num_heads, head_dim).
        k: Key tensor of shape (T, num_kv_heads, head_dim).
        v: Value tensor of shape (T, num_kv_heads, head_dim).
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads.

    Returns:
        Attention output tensor of shape (T, num_heads * head_dim).
    """
    T, _, head_dim = q.shape
    group_size = num_heads // num_kv_heads

    # Broadcast KV heads to match query heads
    if group_size > 1:
        k = np.repeat(k, group_size, axis=1)
        v = np.repeat(v, group_size, axis=1)

    # Transpose to (num_heads, T, head_dim)
    q_t = q.transpose(1, 0, 2)
    k_t = k.transpose(1, 0, 2)
    v_t = v.transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(head_dim)
    scores = (q_t @ k_t.transpose(0, 2, 1)) * scale  # (num_heads, T, T)

    # Causal lower-triangular mask
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores[:, mask] = -1e9

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - scores_max)
    probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    attn_out = (probs @ v_t).transpose(1, 0, 2)  # (T, num_heads, head_dim)
    return attn_out.reshape(T, num_heads * head_dim)


def qwen3_layer(
    x: np.ndarray,
    weights: Mapping[str, np.ndarray],
    cfg: Qwen3TeConfig,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Execute one truncated Qwen3-VL text encoder layer.

    Flow:
      1. input_layernorm
      2. q, k, v linear projections
      3. q_norm and k_norm RMSNorm per head
      4. 1-D RoPE on q and k
      5. GQA causal attention
      6. o_proj linear projection + residual addition
      7. post_attention_layernorm
      8. SwiGLU MLP: gate_proj, up_proj, val = up * silu(gate), down_proj + residual addition
    """
    T = x.shape[0]

    # --- Attention sub-layer ---
    residual = x
    normed = rmsnorm(x, weights["input_layernorm.weight"], cfg.norm_eps)

    q = (normed @ weights["q_proj.weight"].T).reshape(T, cfg.num_heads, cfg.head_dim)
    k = (normed @ weights["k_proj.weight"].T).reshape(T, cfg.num_kv_heads, cfg.head_dim)
    v = (normed @ weights["v_proj.weight"].T).reshape(T, cfg.num_kv_heads, cfg.head_dim)

    # Q-norm and K-norm per head
    q = rmsnorm(q, weights["q_norm.weight"], cfg.qk_norm_eps)
    k = rmsnorm(k, weights["k_norm.weight"], cfg.qk_norm_eps)

    # RoPE
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    # Attention + output projection
    attn_out = gqa_causal_attention(q, k, v, cfg.num_heads, cfg.num_kv_heads)
    proj_out = attn_out @ weights["o_proj.weight"].T
    h = residual + proj_out

    # --- MLP sub-layer ---
    residual = h
    normed = rmsnorm(h, weights["post_attention_layernorm.weight"], cfg.norm_eps)

    gate = normed @ weights["gate_proj.weight"].T
    up = normed @ weights["up_proj.weight"].T
    swiglu = up * silu(gate)
    down = swiglu @ weights["down_proj.weight"].T

    return residual + down


def qwen3_forward(
    ids: Sequence[int],
    embed_table: np.ndarray,
    layers: Sequence[Mapping[str, np.ndarray]],
    cfg: Qwen3TeConfig,
    stop_layer: Optional[int] = None,
) -> np.ndarray:
    """Execute truncated Qwen3-VL text encoder forward pass.

    Returns the PRE-FINAL-NORM hidden state (T x hidden_size).
    """
    stop = cfg.num_layers if stop_layer is None else stop_layer
    token_ids = np.asarray(ids, dtype=np.int32)
    T = len(token_ids)

    # 1. Embedding lookup
    x = embed_table[token_ids].astype(np.float32)

    # 2. RoPE tables for positions 0..T-1
    cos, sin = build_rope_1d(np.arange(T), cfg.head_dim, cfg.rope_theta)

    # 3. Stream through stop_layer layers
    for L in range(stop):
        x = qwen3_layer(x, layers[L], cfg, cos, sin)

    # Returns pre-final-norm output (no model.norm applied)
    return x


# --- Prompt Cache Utilities ---


def prompt_cache_hash(ids: Sequence[int]) -> str:
    """Compute SHA-256 hash of token ID sequence."""
    token_bytes = np.asarray(ids, dtype=np.int32).tobytes()
    return hashlib.sha256(token_bytes).hexdigest()


def prompt_cache_path(cache_dir: str | Path, ids: Sequence[int]) -> Path:
    """Get prompt cache file path for given token IDs."""
    h = prompt_cache_hash(ids)
    return Path(cache_dir) / f"{h}.emb"


def prompt_cache_lookup(
    cache_dir: str | Path,
    ids: Sequence[int],
    hidden_size: int,
) -> Optional[np.ndarray]:
    """Look up cached text embedding. Returns (T, hidden_size) or None on miss."""
    p = prompt_cache_path(cache_dir, ids)
    if not p.is_file():
        return None
    data = p.read_bytes()
    expected_bytes = len(ids) * hidden_size * 4
    if len(data) != expected_bytes:
        return None
    return np.frombuffer(data, dtype=np.float32).reshape(len(ids), hidden_size)


def prompt_cache_save(
    cache_dir: str | Path,
    ids: Sequence[int],
    emb: np.ndarray,
) -> Path:
    """Save text embedding tensor to cache directory."""
    p = prompt_cache_path(cache_dir, ids)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(emb.astype(np.float32).tobytes())
    return p
