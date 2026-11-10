"""VSA-H3: Reference CPU implementation of Video Sparse Attention for MiniMax-H3 / FastH3.

Implements the exact VSA-H3 sparse attention contract:
- 3D spatial-temporal tiling: 64-token tiles with (t, h, w) = (4, 4, 4).
- Prefix segments (text, audio, image references) chunked into 64-token tiles.
- Mean-pooling of queries and keys over valid tokens per tile.
- Tile-level scoring: S_tile = (Q_pool @ K_pool.T) / sqrt(dim).
- Top-k selection at configured sparsity (default 80% -> top 20% video tiles kept).
- Prefix queries are dense (attend to all key tiles).
- Prefix keys are exempt (all queries attend to all prefix keys).
- Gated coarse compression branch:
  If `to_gate_compress` gate is active:
    V_pool = mean(V) per tile
    A_coarse = softmax(S_tile)
    O_coarse = A_coarse @ V_pool
    O = O_fine + O_coarse * gate
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VsaCfg:
    """Configuration for VSA-H3 sparse attention."""

    tile_size: int = 64
    sparsity: float = 0.8
    exempt: bool = True
    prefix_segments: tuple[int, ...] = ()
    video_grid: tuple[int, int, int] = (0, 0, 0)  # (t, h, w) in patches/tokens


@dataclass(frozen=True)
class VsaGeometry:
    """Precomputed tile partitioning and token mappings for a sequence."""

    tile_size: int
    tile_shape: tuple[int, int, int]
    prefix_segments: tuple[int, ...]
    video_grid: tuple[int, int, int]
    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    num_tiles: int
    tile_tokens: list[list[int]]  # list of sequence token indices in each tile
    variable_block_sizes: np.ndarray  # valid token counts per tile
    tok_to_tile: np.ndarray  # (N,) map from sequence token index to tile index


def compute_topk(sparsity: float, num_blocks: int) -> int:
    """Number of tiles to keep under a given sparsity level, clamped to [1, num_blocks]."""
    if num_blocks <= 0:
        return 0
    return max(1, min(int(math.ceil((1.0 - sparsity) * num_blocks)), num_blocks))


def build_vsa_geometry(
    prefix_segments: tuple[int, ...],
    video_grid: tuple[int, int, int],
    tile_size: int = 64,
) -> VsaGeometry:
    """Build tile geometry for a sequence with `prefix_segments` and a 3D `video_grid`."""
    if tile_size != 64:
        raise ValueError(f"Standard FastH3 tile size is 64 (4x4x4), got {tile_size}")

    ts_t, ts_h, ts_w = (4, 4, 4)
    t, h, w = video_grid
    total_video_tokens = t * h * w if all(x > 0 for x in video_grid) else 0
    total_prefix_tokens = sum(prefix_segments)
    total_seq_length = total_prefix_tokens + total_video_tokens

    tile_tokens: list[list[int]] = []

    # 1. Prefix tiles (dense chunks per segment)
    cursor = 0
    for seg_len in prefix_segments:
        c = cursor
        while c < cursor + seg_len:
            sz = min(tile_size, cursor + seg_len - c)
            tile_tokens.append(list(range(c, c + sz)))
            c += sz
        cursor += seg_len
    num_prefix_tiles = len(tile_tokens)

    # 2. Video 3D tiles: (tt, hh, ww)
    if total_video_tokens > 0:
        n_t = int(math.ceil(t / ts_t))
        n_h = int(math.ceil(h / ts_h))
        n_w = int(math.ceil(w / ts_w))
        for tt in range(n_t):
            for hh in range(n_h):
                for ww in range(n_w):
                    toks: list[int] = []
                    for dt in range(tt * ts_t, min((tt + 1) * ts_t, t)):
                        for dh in range(hh * ts_h, min((hh + 1) * ts_h, h)):
                            for dw in range(ww * ts_w, min((ww + 1) * ts_w, w)):
                                tok_idx = cursor + (dt * h * w + dh * w + dw)
                                toks.append(tok_idx)
                    tile_tokens.append(toks)

    num_tiles = len(tile_tokens)
    num_video_tiles = num_tiles - num_prefix_tiles
    variable_block_sizes = np.array([len(toks) for toks in tile_tokens], dtype=np.int32)

    tok_to_tile = np.zeros(total_seq_length, dtype=np.int32)
    for tile_idx, toks in enumerate(tile_tokens):
        for tok in toks:
            tok_to_tile[tok] = tile_idx

    return VsaGeometry(
        tile_size=tile_size,
        tile_shape=(ts_t, ts_h, ts_w),
        prefix_segments=prefix_segments,
        video_grid=video_grid,
        total_seq_length=total_seq_length,
        num_prefix_tiles=num_prefix_tiles,
        num_video_tiles=num_video_tiles,
        num_tiles=num_tiles,
        tile_tokens=tile_tokens,
        variable_block_sizes=variable_block_sizes,
        tok_to_tile=tok_to_tile,
    )


def vsa_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cfg_or_geom: VsaCfg | VsaGeometry,
    sparsity: float | None = None,
    exempt: bool | None = None,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    """Compute VSA-H3 sparse attention.

    Args:
        q: (N, heads, dim) query tensor.
        k: (N, heads, dim) key tensor.
        v: (N, heads, dim) value tensor.
        cfg_or_geom: VsaCfg configuration or precomputed VsaGeometry.
        sparsity: Attention sparsity in [0, 1) (overrides cfg if given).
        exempt: Whether prefix keys are exempt from top-k (overrides cfg if given).
        gate: (N, heads, dim) learned gating tensor from `to_gate_compress`.

    Returns:
        (N, heads, dim) attention output tensor.
    """
    N, heads, dim = q.shape

    if isinstance(cfg_or_geom, VsaGeometry):
        geom = cfg_or_geom
        eff_sparsity = sparsity if sparsity is not None else 0.8
        eff_exempt = exempt if exempt is not None else True
    else:
        cfg = cfg_or_geom
        eff_sparsity = sparsity if sparsity is not None else cfg.sparsity
        eff_exempt = exempt if exempt is not None else cfg.exempt
        geom = build_vsa_geometry(cfg.prefix_segments, cfg.video_grid, cfg.tile_size)

    if geom.total_seq_length != N:
        raise ValueError(
            f"VSA geometry expects seq_len={geom.total_seq_length}, got q.shape[0]={N}"
        )

    # 1. Pool Q and K per tile: mean over valid tokens in each tile
    q_pool = np.zeros((heads, geom.num_tiles, dim), dtype=np.float32)
    k_pool = np.zeros((heads, geom.num_tiles, dim), dtype=np.float32)
    for i, toks in enumerate(geom.tile_tokens):
        q_pool[:, i, :] = np.mean(q[toks, :, :], axis=0)
        k_pool[:, i, :] = np.mean(k[toks, :, :], axis=0)

    # 2. Tile scores: S = (Q_pool @ K_pool.T) / sqrt(dim)
    scale = 1.0 / np.sqrt(dim)
    scores = (q_pool @ k_pool.transpose(0, 2, 1)) * scale  # (heads, num_tiles, num_tiles)

    # 3. Build tile mask: (heads, num_tiles, num_tiles)
    k_video = compute_topk(eff_sparsity, geom.num_video_tiles)

    tile_mask = np.zeros((heads, geom.num_tiles, geom.num_tiles), dtype=bool)
    # Prefix queries are always dense across all keys
    tile_mask[:, : geom.num_prefix_tiles, :] = True

    # Video queries
    if eff_sparsity <= 0.0 or k_video == geom.num_video_tiles:
        # Full dense selection
        tile_mask[:, :, :] = True
    elif eff_exempt or geom.num_prefix_tiles == 0:
        # Prefix keys are exempt: always selected
        tile_mask[:, geom.num_prefix_tiles :, : geom.num_prefix_tiles] = True
        # Video keys: top-k selected per video query tile
        for head in range(heads):
            for i in range(geom.num_prefix_tiles, geom.num_tiles):
                vid_scores = scores[head, i, geom.num_prefix_tiles :]
                topk_idx = (
                    np.argsort(-vid_scores)[:k_video] + geom.num_prefix_tiles
                )
                tile_mask[head, i, topk_idx] = True
    else:
        # Compete mode: prefix keys compete with video keys for top-k
        k_total = min(k_video + geom.num_prefix_tiles, geom.num_tiles)
        for head in range(heads):
            for i in range(geom.num_prefix_tiles, geom.num_tiles):
                topk_idx = np.argsort(-scores[head, i, :])[:k_total]
                tile_mask[head, i, topk_idx] = True

    # 4. Fine Attention per query tile
    out_fine = np.zeros((N, heads, dim), dtype=np.float32)
    for head in range(heads):
        for i, q_toks in enumerate(geom.tile_tokens):
            selected_tiles = np.where(tile_mask[head, i, :])[0]
            k_toks: list[int] = []
            for t_idx in selected_tiles:
                k_toks.extend(geom.tile_tokens[t_idx])
            k_toks_arr = np.array(k_toks, dtype=np.int32)

            q_sub = q[q_toks, head, :]  # (len(q_toks), dim)
            k_sub = k[k_toks_arr, head, :]  # (len(k_toks), dim)
            v_sub = v[k_toks_arr, head, :]  # (len(k_toks), dim)

            attn_scores = (q_sub @ k_sub.T) * scale
            exp_s = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
            attn_p = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

            out_fine[q_toks, head, :] = attn_p @ v_sub

    # 5. Coarse Gated Attention (if gate provided)
    if gate is not None:
        v_pool = np.zeros((heads, geom.num_tiles, dim), dtype=np.float32)
        for i, toks in enumerate(geom.tile_tokens):
            v_pool[:, i, :] = np.mean(v[toks, :, :], axis=0)

        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        coarse_attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        out_c = coarse_attn @ v_pool  # (heads, num_tiles, dim)
        out_c = out_c.transpose(1, 0, 2)  # (num_tiles, heads, dim)

        out_total = np.zeros_like(out_fine)
        for i, toks in enumerate(geom.tile_tokens):
            out_total[toks, :, :] = (
                out_fine[toks, :, :] + out_c[i, None, :, :] * gate[toks, :, :]
            )
        return out_total

    return out_fine
