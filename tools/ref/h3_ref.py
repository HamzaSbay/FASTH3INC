"""MiniMax-H3 / FastH3: Reference CPU / NumPy implementation.

Covers:
- AdaLayerNorm modulation: per-row table indexed by (timestep_index * 3 + modality_tag).
- 3-axis Rotary Position Embedding (RoPE).
- SwiGLU Feed-Forward Networks.
- DiT Transformer block with optional VSA sparse attention.
- Token Refiner block.
- Final AdaLN Layer and projection heads.
- DMD2 8-step sampler schedule and data-ward Euler / renoising step updates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tools.ref.vsa_ref import VsaCfg, VsaGeometry, vsa_attention


# MiniMax-H3 modality tags: 0 = video, 1 = text, 2 = audio
MINIMAX_H3_MODALITY_NUM = 3


@dataclass(frozen=True)
class StepPlan:
    """One denoising iteration plan."""

    sigma_v: float
    sigma_v_next: float
    sigma_a: float
    sigma_a_next: float
    unique_t: np.ndarray  # sorted distinct timesteps for the DiT forward
    text_t: float  # timestep assigned to text rows (equals 1 - sigma_v)


def sampler_plan(
    dmd_steps: list[int] | tuple[int, ...],
    video_shift: float = 10.0,
    audio_shift: float = 3.0,
) -> list[StepPlan]:
    """Compute the multi-step sampler plan for DMD2 few-step models.

    Args:
        dmd_steps: Explicit unshifted training-clock rungs (e.g. [999, 874, 749, 624, 500, 375, 250, 125]).
        video_shift: Exponential shift for video latents (default 10.0 for FastH3 V2).
        audio_shift: Exponential shift for audio latents (default 3.0).

    Returns:
        List of StepPlan objects, one per transformer forward.
    """
    base = np.array([step / 1000.0 for step in dmd_steps] + [0.0], dtype=np.float64)
    # Float64 exponential shift: sigma' = s*sigma / (1 + (s-1)*sigma), then cast to float32
    v_sigmas = (video_shift * base / (1.0 + (video_shift - 1.0) * base)).astype(np.float32)
    a_sigmas = (audio_shift * base / (1.0 + (audio_shift - 1.0) * base)).astype(np.float32)

    plan: list[StepPlan] = []
    num_forwards = len(dmd_steps)
    for i in range(num_forwards):
        sigma_v = float(v_sigmas[i])
        sigma_v_next = float(v_sigmas[i + 1])
        sigma_a = float(a_sigmas[i])
        sigma_a_next = float(a_sigmas[i + 1])

        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - sigma_a)
        unique_t = np.sort(np.unique(np.array([t_v, t_a], dtype=np.float32)))
        text_t = t_v

        plan.append(
            StepPlan(
                sigma_v=sigma_v,
                sigma_v_next=sigma_v_next,
                sigma_a=sigma_a,
                sigma_a_next=sigma_a_next,
                unique_t=unique_t,
                text_t=text_t,
            )
        )
    return plan


def euler_step(
    sample: np.ndarray,
    model_output: np.ndarray,
    sigma: float,
    sigma_next: float,
    timestep: float,
) -> np.ndarray:
    """Take one rectified-flow Euler step with data-ward velocity.

    MiniMax-H3's DiT predicts data-ward velocity v, where:
        x0 = xt + (1 - t) * v = xt + sigma * v
    The Euler blend (eta = 0) is:
        x_next = r * xt + (1 - r) * x0, with r = sigma_next / sigma
    """
    sigma_from_t = 1.0 - timestep
    denoised = sample + sigma_from_t * model_output
    ratio = sigma_next / sigma
    prev_sample = ratio * sample.astype(np.float32) + (1.0 - ratio) * denoised.astype(np.float32)
    return prev_sample.astype(sample.dtype)


def stochastic_step(
    sample: np.ndarray,
    model_output: np.ndarray,
    sigma: float,
    sigma_next: float,
    noise: np.ndarray,
) -> np.ndarray:
    """Take one few-step re-noising step.

    Predicts clean sample x0 = xt + sigma * v, then re-noises to sigma_next with fresh noise.
    """
    denoised = sample.astype(np.float32) + sigma * model_output.astype(np.float32)
    prev_sample = (1.0 - sigma_next) * denoised + sigma_next * noise.astype(np.float32)
    return prev_sample.astype(sample.dtype)


# -----------------------------------------------------------------------------
# Math primitives
# -----------------------------------------------------------------------------


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation function."""
    return x / (1.0 + np.exp(-x))


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Root Mean Square Layer Normalization without bias."""
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)) * weight


def get_timestep_embedding(
    timesteps: np.ndarray,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    max_period: int = 10000,
) -> np.ndarray:
    """Create sinusoidal timestep embeddings matching diffusers."""
    timesteps = np.asarray(timesteps, dtype=np.float32).reshape(-1)
    half_dim = embedding_dim // 2
    exponent = (
        -math.log(max_period)
        * np.arange(half_dim, dtype=np.float32)
        / (half_dim - downscale_freq_shift)
    )
    emb = np.exp(exponent)
    emb = timesteps[:, None] * emb[None, :]
    if flip_sin_to_cos:
        return np.concatenate([np.cos(emb), np.sin(emb)], axis=-1)
    else:
        return np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)


def compute_temb(
    timesteps: np.ndarray,
    weights: dict[str, Any],
    freq_dim: int = 256,
) -> np.ndarray:
    """Compute shared timestep embedding (num_timesteps, time_embed_dim)."""
    t_freq = get_timestep_embedding(
        timesteps, freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
    )
    h = silu(
        t_freq @ weights["time_embedder.linear_1.weight"].T
        + weights["time_embedder.linear_1.bias"]
    )
    temb = (
        h @ weights["time_embedder.linear_2.weight"].T
        + weights["time_embedder.linear_2.bias"]
    )
    return temb


def build_rope(
    position_ids: np.ndarray,
    rope_freq_dim: int = 16,
    rope_theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 3-axis rotary embedding cos and sin tables.

    position_ids: (N, 3) in float64 coordinates (t, h, w).
    Returns cos, sin of shape (N, 2 * 3 * rope_freq_dim).
    """
    pos = position_ids.astype(np.float32)
    inv_freq = 1.0 / (
        rope_theta
        ** (np.arange(0, 2 * rope_freq_dim, 2, dtype=np.float32) / (2 * rope_freq_dim))
    )
    freqs = pos[:, :, None] * inv_freq[None, None, :]  # (N, 3, rope_freq_dim)
    freqs_t, freqs_h, freqs_w = freqs[:, 0, :], freqs[:, 1, :], freqs[:, 2, :]
    freqs = np.concatenate([freqs_t, freqs_h, freqs_w], axis=-1)
    freqs = np.concatenate([freqs, freqs], axis=-1)  # (N, 2 * 3 * rope_freq_dim)
    return np.cos(freqs), np.sin(freqs)


def apply_rotary_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate leading channels of each attention head.

    x: (N, heads, dim)
    cos, sin: (N, rotary_dim) where rotary_dim <= dim
    """
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    x1, x2 = np.split(x_rot, 2, axis=-1)
    x_rotated = np.concatenate([-x2, x1], axis=-1)

    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    x_rot = x_rot * cos_b + x_rotated * sin_b
    return np.concatenate([x_rot, x_pass], axis=-1)


def dense_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Full self-attention over sequence.

    q, k, v: (N, heads, dim)
    Returns out: (N, heads, dim)
    """
    dim = q.shape[-1]
    scale = 1.0 / np.sqrt(dim)

    # Transpose to (heads, N, dim)
    q_t = q.transpose(1, 0, 2)
    k_t = k.transpose(1, 0, 2)
    v_t = v.transpose(1, 0, 2)

    scores = (q_t @ k_t.transpose(0, 2, 1)) * scale
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_p = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    out = attn_p @ v_t  # (heads, N, dim)
    return out.transpose(1, 0, 2)


def swiglu_ffn(x: np.ndarray, w: dict[str, Any], prefix: str = "") -> np.ndarray:
    """SwiGLU feed-forward network with value-first packed projection.

    Formula: out = (val * silu(gate)) @ W2.T
    """
    pfx = f"{prefix}." if prefix and not prefix.endswith(".") else prefix
    w_proj = w[f"{pfx}ff.net.0.proj.weight"]
    w_out = w[f"{pfx}ff.net.2.weight"]

    proj = x @ w_proj.T
    val, gate = np.split(proj, 2, axis=-1)
    act = val * silu(gate)
    return act @ w_out.T


# -----------------------------------------------------------------------------
# Blocks and Layers
# -----------------------------------------------------------------------------


def block(
    x: np.ndarray,
    w: dict[str, Any],
    mod: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    vsa: VsaCfg | VsaGeometry | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
    norm_eps: float = 1e-5,
    qk_norm_eps: float = 1e-5,
) -> np.ndarray:
    """Execute one MiniMax-H3 DiT block.

    Args:
        x: (N, hidden) input sequence.
        w: Dictionary of weights for this block (e.g. norm1.weight, attn.to_q.weight, etc.).
        mod: (N, 6, hidden) AdaLN modulation vectors (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        cos, sin: (N, rotary_dim) RoPE tables.
        vsa: Optional VSA configuration or geometry. If None, runs dense attention.
        num_heads: Number of attention heads (inferred from weights if None).
        head_dim: Dimension per head (inferred from weights if None).
        norm_eps: Epsilon for pre-norm RMSNorm.
        qk_norm_eps: Epsilon for per-head query/key RMSNorm.

    Returns:
        (N, hidden) output sequence.
    """
    N, hidden = x.shape

    shift_msa = mod[:, 0, :]
    scale_msa = mod[:, 1, :]
    gate_msa = mod[:, 2, :]
    shift_mlp = mod[:, 3, :]
    scale_mlp = mod[:, 4, :]
    gate_mlp = mod[:, 5, :]

    if num_heads is None or head_dim is None:
        inner_dim = w["attn.to_q.weight"].shape[0]
        head_dim = w["attn.norm_q.weight"].shape[0]
        num_heads = inner_dim // head_dim
    else:
        inner_dim = num_heads * head_dim

    # 1. Self-Attention branch
    residual = x
    normed = rms_norm(x, w["norm1.weight"], eps=norm_eps)
    normed = normed * (1.0 + scale_msa) + shift_msa

    q = normed @ w["attn.to_q.weight"].T
    k = normed @ w["attn.to_k.weight"].T
    v = normed @ w["attn.to_v.weight"].T

    q = q.reshape(N, num_heads, head_dim)
    k = k.reshape(N, num_heads, head_dim)
    v = v.reshape(N, num_heads, head_dim)

    q = rms_norm(q, w["attn.norm_q.weight"], eps=qk_norm_eps)
    k = rms_norm(k, w["attn.norm_k.weight"], eps=qk_norm_eps)

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    if vsa is not None:
        gate_w = w.get("attn.to_gate_compress.weight")
        if gate_w is not None and not np.all(gate_w == 0):
            gate = (normed @ gate_w.T).reshape(N, num_heads, head_dim)
        else:
            gate = None
        attn_out = vsa_attention(q, k, v, vsa, gate=gate)
    else:
        attn_out = dense_attention(q, k, v)

    attn_out = attn_out.reshape(N, inner_dim) @ w["attn.to_out.0.weight"].T
    x = residual + gate_msa * attn_out

    # 2. Feed-Forward branch
    residual = x
    normed = rms_norm(x, w["norm2.weight"], eps=norm_eps)
    normed = normed * (1.0 + scale_mlp) + shift_mlp
    ff_out = swiglu_ffn(normed, w, prefix="")
    x = residual + gate_mlp * ff_out

    return x


def refiner_block(
    x: np.ndarray,
    w: dict[str, Any],
    prefix: str = "",
    num_heads: int | None = None,
    head_dim: int | None = None,
    norm_eps: float = 1e-5,
    qk_norm_eps: float = 1e-5,
) -> np.ndarray:
    """Execute one token refiner block (no AdaLN, no rotary)."""
    pfx = f"{prefix}." if prefix and not prefix.endswith(".") else prefix
    N, hidden = x.shape

    if num_heads is None or head_dim is None:
        inner_dim = w[f"{pfx}attn.to_q.weight"].shape[0]
        head_dim = w[f"{pfx}attn.norm_q.weight"].shape[0]
        num_heads = inner_dim // head_dim
    else:
        inner_dim = num_heads * head_dim

    # 1. Attention
    residual = x
    normed = rms_norm(x, w[f"{pfx}norm1.weight"], eps=norm_eps)
    q = (normed @ w[f"{pfx}attn.to_q.weight"].T).reshape(N, num_heads, head_dim)
    k = (normed @ w[f"{pfx}attn.to_k.weight"].T).reshape(N, num_heads, head_dim)
    v = (normed @ w[f"{pfx}attn.to_v.weight"].T).reshape(N, num_heads, head_dim)

    q = rms_norm(q, w[f"{pfx}attn.norm_q.weight"], eps=qk_norm_eps)
    k = rms_norm(k, w[f"{pfx}attn.norm_k.weight"], eps=qk_norm_eps)

    attn_out = dense_attention(q, k, v).reshape(N, inner_dim) @ w[f"{pfx}attn.to_out.0.weight"].T
    x = residual + attn_out

    # 2. FFN
    residual = x
    normed = rms_norm(x, w[f"{pfx}norm2.weight"], eps=norm_eps)
    ff_out = swiglu_ffn(normed, w, prefix=prefix)
    x = residual + ff_out

    return x


def final_layer(
    x: np.ndarray,
    w: dict[str, Any],
    mod: np.ndarray | tuple[np.ndarray, np.ndarray],
    video_indices: np.ndarray | None = None,
    audio_indices: np.ndarray | None = None,
    final_norm_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Apply final norm_out modulation and optional output heads."""
    if isinstance(mod, tuple):
        shift, scale = mod
    elif mod.ndim == 3 and mod.shape[1] == 2:
        shift = mod[:, 0, :]
        scale = mod[:, 1, :]
    else:
        raise ValueError(f"Unexpected shape for final layer mod: {mod.shape}")

    normed = rms_norm(x, w["norm_out.norm.weight"], eps=final_norm_eps)
    normed = normed * (1.0 + scale) + shift

    if "proj_out.weight" in w and "audio_proj_out.weight" in w:
        v_rows = normed[video_indices] if video_indices is not None else normed
        a_rows = normed[audio_indices] if audio_indices is not None else normed

        v_out = v_rows @ w["proj_out.weight"].T + w.get("proj_out.bias", 0)
        a_out = a_rows @ w["audio_proj_out.weight"].T + w.get("audio_proj_out.bias", 0)
        return v_out, a_out

    return normed


# -----------------------------------------------------------------------------
# Full DiT Forward Reference
# -----------------------------------------------------------------------------


def dit_forward(
    weights: dict[str, Any],
    video_latents: np.ndarray,
    audio_latents: np.ndarray,
    text_embeds: np.ndarray,
    timesteps: np.ndarray,
    timestep_indices: np.ndarray,
    token_tags: np.ndarray,
    position_ids: np.ndarray,
    video_indices: np.ndarray,
    audio_indices: np.ndarray,
    text_indices: np.ndarray,
    num_layers: int = 50,
    num_refiner_layers: int = 2,
    num_heads: int = 56,
    head_dim: int = 128,
    hidden_size: int = 5376,
    freq_dim: int = 256,
    rope_freq_dim: int = 16,
    rope_theta: float = 10000.0,
    norm_eps: float = 1e-5,
    qk_norm_eps: float = 1e-5,
    final_norm_eps: float = 1e-5,
    vsa: VsaCfg | VsaGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full forward pass of the MiniMax-H3 / FastH3 Transformer."""
    seq_len = position_ids.shape[0]

    # 1. Modality Input Projections
    v_in = video_latents @ weights["proj_in.weight"].T + weights["proj_in.bias"]
    a_in = audio_latents @ weights["audio_proj_in.weight"].T + weights["audio_proj_in.bias"]
    t_in = text_embeds @ weights["context_embedder.weight"].T + weights["context_embedder.bias"]

    # 2. Token Refiner for text
    for r in range(num_refiner_layers):
        rw = {
            k.replace(f"token_refiner.refiner_blocks.{r}.", ""): v
            for k, v in weights.items()
            if k.startswith(f"token_refiner.refiner_blocks.{r}.")
        }
        t_in = refiner_block(
            t_in,
            rw,
            num_heads=num_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
        )
    t_in = rms_norm(t_in, weights["token_refiner.final_norm.weight"], eps=final_norm_eps)

    # 3. Assemble packed sequence
    packed = np.zeros((seq_len, hidden_size), dtype=np.float32)
    packed[text_indices] = t_in
    packed[audio_indices] = a_in
    packed[video_indices] = v_in

    # 4. Timestep embedding and RoPE
    temb = compute_temb(timesteps, weights, freq_dim=freq_dim)
    cos, sin = build_rope(position_ids, rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)

    # 5. Blocks
    adaln_idx = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
    for b in range(num_layers):
        bw = {
            k.replace(f"transformer_blocks.{b}.", ""): v
            for k, v in weights.items()
            if k.startswith(f"transformer_blocks.{b}.")
        }
        # AdaLN modulation table
        w_linear = weights[f"transformer_blocks.{b}.adaln_proj.linear.weight"]
        b_linear = weights[f"transformer_blocks.{b}.adaln_proj.linear.bias"]
        proj_adaln = silu(temb) @ w_linear.T + b_linear
        proj_adaln = proj_adaln.reshape(-1, 6, hidden_size)
        mod_b = proj_adaln[adaln_idx]

        packed = block(
            packed,
            bw,
            mod_b,
            cos,
            sin,
            vsa=vsa,
            num_heads=num_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
        )

    # 6. Final Layer
    out_linear = silu(temb) @ weights["norm_out.linear.weight"].T + weights["norm_out.linear.bias"]
    shift_out, scale_out = np.split(out_linear, 2, axis=-1)
    mod_out = np.stack(
        [shift_out[timestep_indices], scale_out[timestep_indices]], axis=1
    )

    v_out, a_out = final_layer(
        packed,
        weights,
        mod_out,
        video_indices=video_indices,
        audio_indices=audio_indices,
        final_norm_eps=final_norm_eps,
    )
    return v_out, a_out
