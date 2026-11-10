#!/usr/bin/env python3
"""Emit reference fixtures for 8-step sampler and noise parity in C."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3 import before_denoise

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ref import h3_ref


def emit_sched_fixtures(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

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

    num_text = 6
    num_audio = 4
    num_video = 8
    seq_len = num_text + num_audio + num_video

    pos, tags, v_idx, a_idx, t_idx, _, _ = before_denoise.MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.ones(num_text, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
    )

    # Load initial latents from tiny_h3 fixtures
    tiny_dir = ROOT / "tests" / "fixtures" / "tiny_h3"
    with open(tiny_dir / "x_video.bin", "rb") as f:
        x_v_init = np.frombuffer(f.read(), dtype=np.float32).reshape(num_video, 96).copy()
    with open(tiny_dir / "x_audio.bin", "rb") as f:
        x_a_init = np.frombuffer(f.read(), dtype=np.float32).reshape(num_audio, 32).copy()
    with open(tiny_dir / "text_emb.bin", "rb") as f:
        text_emb = torch.from_numpy(np.frombuffer(f.read(), dtype=np.float32).reshape(1, num_text, 64))

    dmd_steps = [999, 874, 749, 624, 500, 375, 250, 125]
    plan = h3_ref.sampler_plan(dmd_steps, video_shift=10.0, audio_shift=3.0)

    # -------------------------------------------------------------------------
    # 1. Stochastic renoising 8-step rollout
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(2026)
    x_v_stoch = x_v_init.copy()
    x_a_stoch = x_a_init.copy()

    noise_v_steps = []
    noise_a_steps = []

    for i in range(8):
        sp = plan[i]
        t_v = float(1.0 - sp.sigma_v)
        t_a = float(1.0 - sp.sigma_a)
        unique_t = np.sort(np.unique(np.array([t_v, t_a], dtype=np.float32)))
        idx_v = int(np.searchsorted(unique_t, t_v))
        idx_a = int(np.searchsorted(unique_t, t_a))

        timestep_indices = torch.empty(seq_len, dtype=torch.long)
        timestep_indices[t_idx] = idx_v
        timestep_indices[v_idx] = idx_v
        timestep_indices[a_idx] = idx_a

        with torch.no_grad():
            out = model(
                hidden_states=torch.from_numpy(x_v_stoch).unsqueeze(0),
                audio_hidden_states=torch.from_numpy(x_a_stoch).unsqueeze(0),
                encoder_hidden_states=text_emb,
                timestep=torch.from_numpy(unique_t),
                timestep_indices=timestep_indices,
                token_tags=tags,
                position_ids=pos,
                video_indices=v_idx,
                audio_indices=a_idx,
                text_indices=t_idx,
                return_dict=True,
            )
        v_v = out.sample.squeeze(0).numpy().astype(np.float32)
        v_a = out.audio_sample.squeeze(0).numpy().astype(np.float32)

        if i < 7:
            # Generate fresh noise for step i+1
            nv = rng.standard_normal((num_video, 96), dtype=np.float32)
            na = rng.standard_normal((num_audio, 32), dtype=np.float32)
            noise_v_steps.append(nv)
            noise_a_steps.append(na)

            with open(out_dir / f"noise_v_step{i}.bin", "wb") as f:
                f.write(nv.tobytes())
            with open(out_dir / f"noise_a_step{i}.bin", "wb") as f:
                f.write(na.tobytes())

            x_v_stoch = h3_ref.stochastic_step(x_v_stoch, v_v, sp.sigma_v, sp.sigma_v_next, nv)
            x_a_stoch = h3_ref.stochastic_step(x_a_stoch, v_a, sp.sigma_a, sp.sigma_a_next, na)
        else:
            # Terminal step: returns x0
            x_v_stoch = x_v_stoch + sp.sigma_v * v_v
            x_a_stoch = x_a_stoch + sp.sigma_a * v_a

    # Write stochastic outputs
    with open(out_dir / "final_video_latents.bin", "wb") as f:
        f.write(x_v_stoch.tobytes())
    with open(out_dir / "final_audio_latents.bin", "wb") as f:
        f.write(x_a_stoch.tobytes())

    # Combined noise.bin for convenience: concatenate noise_v_step0..6 and noise_a_step0..6
    with open(out_dir / "noise.bin", "wb") as f:
        for nv, na in zip(noise_v_steps, noise_a_steps):
            f.write(nv.tobytes())
            f.write(na.tobytes())

    # -------------------------------------------------------------------------
    # 2. Deterministic Euler 8-step rollout
    # -------------------------------------------------------------------------
    x_v_euler = x_v_init.copy()
    x_a_euler = x_a_init.copy()

    for i in range(8):
        sp = plan[i]
        t_v = float(1.0 - sp.sigma_v)
        t_a = float(1.0 - sp.sigma_a)
        unique_t = np.sort(np.unique(np.array([t_v, t_a], dtype=np.float32)))
        idx_v = int(np.searchsorted(unique_t, t_v))
        idx_a = int(np.searchsorted(unique_t, t_a))

        timestep_indices = torch.empty(seq_len, dtype=torch.long)
        timestep_indices[t_idx] = idx_v
        timestep_indices[v_idx] = idx_v
        timestep_indices[a_idx] = idx_a

        with torch.no_grad():
            out = model(
                hidden_states=torch.from_numpy(x_v_euler).unsqueeze(0),
                audio_hidden_states=torch.from_numpy(x_a_euler).unsqueeze(0),
                encoder_hidden_states=text_emb,
                timestep=torch.from_numpy(unique_t),
                timestep_indices=timestep_indices,
                token_tags=tags,
                position_ids=pos,
                video_indices=v_idx,
                audio_indices=a_idx,
                text_indices=t_idx,
                return_dict=True,
            )
        v_v = out.sample.squeeze(0).numpy().astype(np.float32)
        v_a = out.audio_sample.squeeze(0).numpy().astype(np.float32)

        if i < 7:
            x_v_euler = h3_ref.euler_step(x_v_euler, v_v, sp.sigma_v, sp.sigma_v_next, t_v)
            x_a_euler = h3_ref.euler_step(x_a_euler, v_a, sp.sigma_a, sp.sigma_a_next, t_a)
        else:
            x_v_euler = x_v_euler + sp.sigma_v * v_v
            x_a_euler = x_a_euler + sp.sigma_a * v_a

    with open(out_dir / "euler_final_video_latents.bin", "wb") as f:
        f.write(x_v_euler.tobytes())
    with open(out_dir / "euler_final_audio_latents.bin", "wb") as f:
        f.write(x_a_euler.tobytes())

    print(f"Emitted 8-step sampler fixtures to {out_dir}")
    print(f"  Stochastic final latents: video L2={np.linalg.norm(x_v_stoch):.4f}, audio L2={np.linalg.norm(x_a_stoch):.4f}")
    print(f"  Euler final latents:      video L2={np.linalg.norm(x_v_euler):.4f}, audio L2={np.linalg.norm(x_a_euler):.4f}")


if __name__ == "__main__":
    out = ROOT / "tests" / "fixtures" / "tiny_h3"
    emit_sched_fixtures(out)
