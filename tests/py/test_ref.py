"""Tests for NumPy reference implementations: h3_ref and vsa_ref."""

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
from diffusers import MiniMaxH3Transformer3DModel

from tools.ref.h3_ref import (
    StepPlan,
    block,
    dit_forward,
    euler_step,
    final_layer,
    refiner_block,
    sampler_plan,
    stochastic_step,
)
from tools.ref.vsa_ref import (
    VsaCfg,
    build_vsa_geometry,
    compute_topk,
    vsa_attention,
)


def test_ref_matches_diffusers():
    """Verify that h3_ref matches diffusers MiniMaxH3Transformer3DModel within rel-L2 <= 1e-5."""
    torch.manual_seed(0)
    np.random.seed(0)

    config = dict(
        num_attention_heads=2,
        attention_head_dim=128,
        hidden_size=256,
        num_layers=3,
        num_refiner_layers=2,
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
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
    )

    model = MiniMaxH3Transformer3DModel(**config)
    model.eval()

    batch_size = 1
    num_video_tokens = 8
    video_patch_dim = 24 * 1 * 2 * 2  # 96
    num_audio_tokens = 4
    num_text_tokens = 6
    seq_len = num_text_tokens + num_audio_tokens + num_video_tokens

    hidden_states = torch.randn(batch_size, num_video_tokens, video_patch_dim, dtype=torch.float32)
    audio_hidden_states = torch.randn(batch_size, num_audio_tokens, 32, dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch_size, num_text_tokens, 64, dtype=torch.float32)

    text_indices = torch.arange(0, num_text_tokens, dtype=torch.long)
    audio_indices = torch.arange(num_text_tokens, num_text_tokens + num_audio_tokens, dtype=torch.long)
    video_indices = torch.arange(num_text_tokens + num_audio_tokens, seq_len, dtype=torch.long)

    token_tags = torch.empty(seq_len, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timesteps = torch.tensor([0.5, 0.8], dtype=torch.float32)
    timestep_indices = torch.empty(seq_len, dtype=torch.long)
    timestep_indices[text_indices] = 1
    timestep_indices[video_indices] = 1
    timestep_indices[audio_indices] = 0

    position_ids = torch.randn(seq_len, 3, dtype=torch.float64)

    with torch.no_grad():
        torch_out = model(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            return_dict=True,
        )

    torch_video = torch_out.sample.squeeze(0).numpy()
    torch_audio = torch_out.audio_sample.squeeze(0).numpy()

    weights = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}

    ref_video, ref_audio = dit_forward(
        weights=weights,
        video_latents=hidden_states.squeeze(0).numpy(),
        audio_latents=audio_hidden_states.squeeze(0).numpy(),
        text_embeds=encoder_hidden_states.squeeze(0).numpy(),
        timesteps=timesteps.numpy(),
        timestep_indices=timestep_indices.numpy(),
        token_tags=token_tags.numpy(),
        position_ids=position_ids.numpy(),
        video_indices=video_indices.numpy(),
        audio_indices=audio_indices.numpy(),
        text_indices=text_indices.numpy(),
        num_layers=config["num_layers"],
        num_refiner_layers=config["num_refiner_layers"],
        num_heads=config["num_attention_heads"],
        head_dim=config["attention_head_dim"],
        hidden_size=config["hidden_size"],
        freq_dim=config["freq_dim"],
        rope_freq_dim=config["rope_freq_dim"],
        rope_theta=config["rope_theta"],
        norm_eps=config["norm_eps"],
        qk_norm_eps=config["qk_norm_eps"],
        final_norm_eps=config["final_norm_eps"],
    )

    rel_l2_video = np.linalg.norm(ref_video - torch_video) / np.linalg.norm(torch_video)
    rel_l2_audio = np.linalg.norm(ref_audio - torch_audio) / np.linalg.norm(torch_audio)

    assert rel_l2_video <= 1e-5, f"Video rel-L2 {rel_l2_video} > 1e-5"
    assert rel_l2_audio <= 1e-5, f"Audio rel-L2 {rel_l2_audio} > 1e-5"


def test_vsa_ref_matches_fastvideo():
    """Test VSA sparse attention on N=512 random q/k/v against FastVideo's PyTorch masked attention reference."""
    np.random.seed(123)
    torch.manual_seed(123)

    # N = 512 tokens: 128 prefix tokens (2 tiles of 64), 384 video tokens (grid 4x8x12 = 6 tiles of 64)
    prefix_segments = (128,)
    video_grid = (4, 8, 12)
    N = 512
    heads = 4
    dim = 64
    sparsity = 0.8

    q = np.random.randn(N, heads, dim).astype(np.float32)
    k = np.random.randn(N, heads, dim).astype(np.float32)
    v = np.random.randn(N, heads, dim).astype(np.float32)

    cfg = VsaCfg(tile_size=64, sparsity=sparsity, exempt=True, prefix_segments=prefix_segments, video_grid=video_grid)
    vsa_out = vsa_attention(q, k, v, cfg)

    # Ground-truth PyTorch masked attention reference (matching FastVideo test_block_sparse_blackwell.py oracle)
    geom = build_vsa_geometry(prefix_segments, video_grid, tile_size=64)
    assert geom.num_tiles == 8
    assert geom.num_prefix_tiles == 2
    assert geom.num_video_tiles == 6

    q_t = torch.from_numpy(q).permute(1, 0, 2)  # (heads, N, dim)
    k_t = torch.from_numpy(k).permute(1, 0, 2)
    v_t = torch.from_numpy(v).permute(1, 0, 2)
    scale = 1.0 / np.sqrt(dim)

    q_pool = torch.stack([q_t[:, toks, :].mean(dim=1) for toks in geom.tile_tokens], dim=1)
    k_pool = torch.stack([k_t[:, toks, :].mean(dim=1) for toks in geom.tile_tokens], dim=1)
    tile_scores = torch.bmm(q_pool, k_pool.transpose(1, 2)) * scale

    keep_mask = torch.zeros(heads, N, N, dtype=torch.bool)
    # Prefix queries attend to all keys
    for qb in range(geom.num_prefix_tiles):
        keep_mask[:, geom.tile_tokens[qb], :] = True

    # Video queries attend to all prefix keys
    for qb in range(geom.num_prefix_tiles, geom.num_tiles):
        for kb in range(geom.num_prefix_tiles):
            for q_tok in geom.tile_tokens[qb]:
                keep_mask[:, q_tok, geom.tile_tokens[kb]] = True

    # Video queries attend to top-k video keys
    k_video = compute_topk(sparsity, geom.num_video_tiles)
    assert k_video == 2  # ceil(0.2 * 6) = 2

    for h in range(heads):
        for qb in range(geom.num_prefix_tiles, geom.num_tiles):
            vid_scores = tile_scores[h, qb, geom.num_prefix_tiles :]
            topk = torch.topk(vid_scores, k=k_video).indices + geom.num_prefix_tiles
            for kb in topk:
                for q_tok in geom.tile_tokens[qb]:
                    keep_mask[h, q_tok, geom.tile_tokens[kb]] = True

    scores = torch.bmm(q_t, k_t.transpose(1, 2)) * scale
    scores.masked_fill_(~keep_mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    ref_out = torch.bmm(p, v_t).permute(1, 0, 2).numpy()

    rel_l2 = np.linalg.norm(vsa_out - ref_out) / np.linalg.norm(ref_out)
    assert rel_l2 <= 1e-5, f"VSA vs PyTorch Mask reference rel-L2 {rel_l2} > 1e-5"


def test_vsa_sparsity0_equals_dense():
    """Verify that at sparsity 0.0, VSA attention equals standard dense attention."""
    np.random.seed(42)
    prefix_segments = (64,)
    video_grid = (4, 4, 8)  # 128 video tokens
    N = 64 + 128
    heads = 2
    dim = 32

    q = np.random.randn(N, heads, dim).astype(np.float32)
    k = np.random.randn(N, heads, dim).astype(np.float32)
    v = np.random.randn(N, heads, dim).astype(np.float32)

    cfg = VsaCfg(tile_size=64, sparsity=0.0, exempt=True, prefix_segments=prefix_segments, video_grid=video_grid)
    vsa_out = vsa_attention(q, k, v, cfg)

    # Dense attention
    scale = 1.0 / np.sqrt(dim)
    scores = (q.transpose(1, 0, 2) @ k.transpose(1, 0, 2).transpose(0, 2, 1)) * scale
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    p = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    dense_out = (p @ v.transpose(1, 0, 2)).transpose(1, 0, 2)

    rel_diff = np.linalg.norm(vsa_out - dense_out) / np.linalg.norm(dense_out)
    assert rel_diff <= 1e-5, f"Sparsity 0 vs dense rel-diff {rel_diff} > 1e-5"


def test_vsa_gated_compression():
    """Verify that gate_compress modifies output when gate is nonzero and is inactive when gate is None."""
    np.random.seed(99)
    prefix_segments = (64,)
    video_grid = (4, 4, 4)  # 64 video tokens
    N = 128
    heads = 2
    dim = 32

    q = np.random.randn(N, heads, dim).astype(np.float32)
    k = np.random.randn(N, heads, dim).astype(np.float32)
    v = np.random.randn(N, heads, dim).astype(np.float32)

    cfg = VsaCfg(tile_size=64, sparsity=0.8, prefix_segments=prefix_segments, video_grid=video_grid)
    out_fine = vsa_attention(q, k, v, cfg, gate=None)

    gate_zero = np.zeros_like(q)
    out_gate_zero = vsa_attention(q, k, v, cfg, gate=gate_zero)
    np.testing.assert_allclose(out_fine, out_gate_zero, atol=1e-6)

    gate_rand = np.random.randn(*q.shape).astype(np.float32)
    out_gated = vsa_attention(q, k, v, cfg, gate=gate_rand)
    assert not np.allclose(out_fine, out_gated, atol=1e-3)


def test_sampler_plan():
    """Verify DMD2 8-step sampler schedule and data-ward Euler update."""
    dmd_steps = [999, 874, 749, 624, 500, 375, 250, 125]
    plan = sampler_plan(dmd_steps, video_shift=10.0, audio_shift=3.0)

    assert len(plan) == 8

    base = np.array([step / 1000.0 for step in dmd_steps] + [0.0], dtype=np.float64)
    v_sigmas = (10.0 * base / (1.0 + 9.0 * base)).astype(np.float32)
    a_sigmas = (3.0 * base / (1.0 + 2.0 * base)).astype(np.float32)

    for i, step in enumerate(plan):
        assert math.isclose(step.sigma_v, v_sigmas[i], abs_tol=1e-6)
        assert math.isclose(step.sigma_v_next, v_sigmas[i + 1], abs_tol=1e-6)
        assert math.isclose(step.sigma_a, a_sigmas[i], abs_tol=1e-6)
        assert math.isclose(step.sigma_a_next, a_sigmas[i + 1], abs_tol=1e-6)
        assert math.isclose(step.text_t, 1.0 - v_sigmas[i], abs_tol=1e-6)

    assert plan[-1].sigma_v_next == 0.0
    assert plan[-1].sigma_a_next == 0.0

    # Test Euler step data-ward update: x_next = r * xt + (1 - r) * (xt + sigma * v)
    xt = np.array([1.0, 2.0], dtype=np.float32)
    v = np.array([0.5, -0.5], dtype=np.float32)
    sigma = 0.8
    sigma_next = 0.4
    t = 1.0 - sigma

    next_x = euler_step(xt, v, sigma, sigma_next, t)
    # Expected: x0 = xt + sigma * v = [1.4, 1.6]
    # next_x = 0.5 * xt + 0.5 * x0 = [1.2, 1.8]
    expected = 0.5 * xt + 0.5 * (xt + sigma * v)
    np.testing.assert_allclose(next_x, expected, atol=1e-6)



def test_individual_layers():
    """Explicitly verify individual layer interfaces: block, refiner_block, final_layer, stochastic_step."""
    np.random.seed(7)
    N = 16
    hidden = 256
    heads = 2
    head_dim = 128
    rotary_dim = 96

    x = np.random.randn(N, hidden).astype(np.float32)
    mod = np.random.randn(N, 6, hidden).astype(np.float32)
    cos = np.ones((N, rotary_dim), dtype=np.float32)
    sin = np.zeros((N, rotary_dim), dtype=np.float32)

    # 1. block()
    block_weights = {
        "norm1.weight": np.ones(hidden, dtype=np.float32),
        "attn.to_q.weight": np.random.randn(heads * head_dim, hidden).astype(np.float32) * 0.02,
        "attn.to_k.weight": np.random.randn(heads * head_dim, hidden).astype(np.float32) * 0.02,
        "attn.to_v.weight": np.random.randn(heads * head_dim, hidden).astype(np.float32) * 0.02,
        "attn.norm_q.weight": np.ones(head_dim, dtype=np.float32),
        "attn.norm_k.weight": np.ones(head_dim, dtype=np.float32),
        "attn.to_out.0.weight": np.random.randn(hidden, heads * head_dim).astype(np.float32) * 0.02,
        "norm2.weight": np.ones(hidden, dtype=np.float32),
        "ff.net.0.proj.weight": np.random.randn(2 * 512, hidden).astype(np.float32) * 0.02,
        "ff.net.2.weight": np.random.randn(hidden, 512).astype(np.float32) * 0.02,
    }

    out_block = block(x, block_weights, mod, cos, sin, vsa=None, num_heads=heads, head_dim=head_dim)
    assert out_block.shape == (N, hidden)
    assert not np.isnan(out_block).any()

    # 2. refiner_block()
    refiner_weights = {
        "norm1.weight": np.ones(hidden, dtype=np.float32),
        "attn.to_q.weight": np.random.randn(heads * head_dim, hidden).astype(np.float32) * 0.02,
        "attn.to_k.weight": np.random.randn(heads * head_dim, hidden).astype(np.float32) * 0.02,
        "attn.to_v.weight": np.random.randn(heads * head_dim, hidden).astype(np.float32) * 0.02,
        "attn.norm_q.weight": np.ones(head_dim, dtype=np.float32),
        "attn.norm_k.weight": np.ones(head_dim, dtype=np.float32),
        "attn.to_out.0.weight": np.random.randn(hidden, heads * head_dim).astype(np.float32) * 0.02,
        "norm2.weight": np.ones(hidden, dtype=np.float32),
        "ff.net.0.proj.weight": np.random.randn(2 * 512, hidden).astype(np.float32) * 0.02,
        "ff.net.2.weight": np.random.randn(hidden, 512).astype(np.float32) * 0.02,
    }

    out_refiner = refiner_block(x, refiner_weights, num_heads=heads, head_dim=head_dim)
    assert out_refiner.shape == (N, hidden)
    assert not np.isnan(out_refiner).any()

    # 3. final_layer()
    final_weights = {
        "norm_out.norm.weight": np.ones(hidden, dtype=np.float32),
        "proj_out.weight": np.random.randn(96, hidden).astype(np.float32) * 0.02,
        "proj_out.bias": np.zeros(96, dtype=np.float32),
        "audio_proj_out.weight": np.random.randn(32, hidden).astype(np.float32) * 0.02,
        "audio_proj_out.bias": np.zeros(32, dtype=np.float32),
    }
    final_mod = np.random.randn(N, 2, hidden).astype(np.float32)
    vid_idx = np.arange(8, 16)
    aud_idx = np.arange(4, 8)
    v_out, a_out = final_layer(x, final_weights, final_mod, video_indices=vid_idx, audio_indices=aud_idx)
    assert v_out.shape == (8, 96)
    assert a_out.shape == (4, 32)

    # 4. stochastic_step()
    sample = np.array([1.0, 2.0], dtype=np.float32)
    m_out = np.array([0.5, -0.5], dtype=np.float32)
    noise = np.array([0.1, -0.1], dtype=np.float32)
    stoch = stochastic_step(sample, m_out, sigma=0.5, sigma_next=0.2, noise=noise)
    # x0 = [1.25, 1.75]; prev = 0.8 * x0 + 0.2 * noise = [1.02, 1.38]
    expected_stoch = 0.8 * (sample + 0.5 * m_out) + 0.2 * noise
    np.testing.assert_allclose(stoch, expected_stoch, atol=1e-6)
