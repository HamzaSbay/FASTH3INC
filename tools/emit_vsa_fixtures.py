#!/usr/bin/env python3
"""Emit reference fixtures for VSA-H3 sparse attention in C."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tools.ref.vsa_ref import VsaCfg, vsa_attention

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "vsa"


def emit_fixture(
    name: str,
    prefix_segments: tuple[int, ...],
    video_grid: tuple[int, int, int],
    heads: int,
    dim: int,
    sparsity: float,
    has_gate: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)

    t, h, w = video_grid
    num_video = t * h * w if all(x > 0 for x in video_grid) else 0
    num_text = prefix_segments[0] if len(prefix_segments) > 0 else 0
    num_audio = prefix_segments[1] if len(prefix_segments) > 1 else 0
    n = num_text + num_audio + num_video

    q = (rng.standard_normal((n, heads, dim)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((n, heads, dim)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((n, heads, dim)) * 0.5).astype(np.float32)
    gate = (rng.standard_normal((n, heads, dim)) * 0.5).astype(np.float32) if has_gate else None

    cfg = VsaCfg(
        tile_size=64,
        sparsity=sparsity,
        exempt=True,
        prefix_segments=prefix_segments,
        video_grid=video_grid,
    )

    ref = vsa_attention(q, k, v, cfg, gate=gate).astype(np.float32)

    bin_path = FIXTURES_DIR / f"{name}.bin"
    with open(bin_path, "wb") as f:
        # Header: 10 uint32, 1 float32
        header = struct.pack(
            "<IIIIIIIIIfI",
            n,
            num_text,
            num_audio,
            num_video,
            t,
            h,
            w,
            heads,
            dim,
            sparsity,
            1 if has_gate else 0,
        )
        f.write(header)
        f.write(q.tobytes())
        f.write(k.tobytes())
        f.write(v.tobytes())
        if has_gate and gate is not None:
            f.write(gate.tobytes())
        f.write(ref.tobytes())

    json_path = FIXTURES_DIR / f"{name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "n": n,
                "num_text": num_text,
                "num_audio": num_audio,
                "num_video": num_video,
                "video_grid": list(video_grid),
                "heads": heads,
                "head_dim": dim,
                "sparsity": sparsity,
                "has_gate": has_gate,
                "ref_sample": ref[0, 0, :8].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Emitted {name}: N={n} (T={num_text}, A={num_audio}, V={num_video}, grid={video_grid}), heads={heads}, dim={dim}, sparsity={sparsity} -> {bin_path}")


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. vsa_n784: grid 7x8x14 (N=784)
    emit_fixture(
        name="vsa_n784",
        prefix_segments=(),
        video_grid=(7, 8, 14),
        heads=4,
        dim=64,
        sparsity=0.8,
        has_gate=True,
        seed=42,
    )

    # 2. vsa_padded: grid length not a multiple of 64, with prefix segments
    emit_fixture(
        name="vsa_padded",
        prefix_segments=(35, 23),
        video_grid=(5, 6, 7),  # 210 video tokens -> N = 35 + 23 + 210 = 268
        heads=4,
        dim=64,
        sparsity=0.8,
        has_gate=True,
        seed=123,
    )

    # 3. vsa_sparsity0: sparsity=0.0
    emit_fixture(
        name="vsa_sparsity0",
        prefix_segments=(20, 16),
        video_grid=(4, 4, 8),  # 128 video tokens -> N = 20 + 16 + 128 = 164
        heads=4,
        dim=64,
        sparsity=0.0,
        has_gate=True,
        seed=999,
    )

    # 4. tiny_h3 VSA fixtures (step 0 and step 7)
    tiny_dir = REPO_ROOT / "tests" / "fixtures" / "tiny_h3"
    if (tiny_dir / "tiny_h3.gguf").exists():
        import torch
        from diffusers import MiniMaxH3Transformer3DModel
        from tools.ref import h3_ref

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

        x_v = np.fromfile(tiny_dir / "x_video.bin", dtype=np.float32).reshape(8, 96)
        x_a = np.fromfile(tiny_dir / "x_audio.bin", dtype=np.float32).reshape(4, 32)
        text_emb = np.fromfile(tiny_dir / "text_emb.bin", dtype=np.float32).reshape(6, 64)
        pos = np.fromfile(tiny_dir / "pos.bin", dtype=np.float64).reshape(18, 3)

        dmd_steps = [999, 874, 749, 624, 500, 375, 250, 125]
        plan = h3_ref.sampler_plan(dmd_steps, 10.0, 3.0)
        v_idx = np.arange(10, 18)
        a_idx = np.arange(6, 10)
        t_idx = np.arange(0, 6)
        weights = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
        cfg = VsaCfg(tile_size=64, sparsity=0.8, exempt=True, prefix_segments=(6, 4), video_grid=(1, 1, 8))

        for step in [0, 7]:
            sp = plan[step]
            tv = float(1.0 - sp.sigma_v)
            ta = float(1.0 - sp.sigma_a)
            unique_t = np.sort(np.unique(np.array([tv, ta], dtype=np.float32)))
            idx_v = int(np.searchsorted(unique_t, tv))
            idx_a = int(np.searchsorted(unique_t, ta))
            timesteps = unique_t
            timestep_indices = np.array([idx_v]*6 + [idx_a]*4 + [idx_v]*8, dtype=np.int64)
            token_tags = np.array([1]*6 + [2]*4 + [0]*8, dtype=np.int64)

            rv, ra = h3_ref.dit_forward(
                weights=weights,
                video_latents=x_v,
                audio_latents=x_a,
                text_embeds=text_emb,
                timesteps=timesteps,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=pos,
                video_indices=v_idx,
                audio_indices=a_idx,
                text_indices=t_idx,
                num_layers=3,
                num_refiner_layers=2,
                num_heads=2,
                head_dim=128,
                hidden_size=256,
                freq_dim=64,
                vsa=cfg,
            )
            rv.astype(np.float32).tofile(tiny_dir / f"video_v_vsa_step{step}.bin")
            ra.astype(np.float32).tofile(tiny_dir / f"audio_v_vsa_step{step}.bin")
            print(f"Emitted tiny_h3 video_v_vsa_step{step}.bin and audio_v_vsa_step{step}.bin")


if __name__ == "__main__":
    main()
