#!/usr/bin/env python3
"""Generate tiny MiniMax-H3 / FastH3 DiT oracle fixture and packed binaries.

Creates:
- tests/fixtures/tiny_h3/dit_trunk.bin
- tests/fixtures/tiny_h3/dit_globals.bin
- tests/fixtures/tiny_h3/adaln_table.bin
- tests/fixtures/tiny_h3/text_emb.bin
- tests/fixtures/tiny_h3/text_rows.bin
- tests/fixtures/tiny_h3/x_video.bin
- tests/fixtures/tiny_h3/x_audio.bin
- tests/fixtures/tiny_h3/pos.bin
- tests/fixtures/tiny_h3/video_v_step0.bin
- tests/fixtures/tiny_h3/audio_v_step0.bin
- tests/fixtures/tiny_h3/video_v_step7.bin
- tests/fixtures/tiny_h3/audio_v_step7.bin
- tests/fixtures/tiny_h3/meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.exit("Error: 'gguf' Python package not found. Run in project .venv.")

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3 import before_denoise

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import fold_adaln, pack_dit
from tools.ref import h3_ref


def make_tiny_h3(out_dir: str | Path) -> None:
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

    video_patch_dim = 24 * 1 * 2 * 2  # 96
    audio_patch_dim = 32
    text_dim = 64

    x_video = torch.randn(1, num_video, video_patch_dim, dtype=torch.float32)
    x_audio = torch.randn(1, num_audio, audio_patch_dim, dtype=torch.float32)
    text_emb = torch.randn(1, num_text, text_dim, dtype=torch.float32)

    # Compute text_rows (condition_proj + token_refiner)
    with torch.no_grad():
        t_in = model.context_embedder(text_emb)
        text_rows = model.token_refiner(t_in)

    # 1. Write GGUF with realrebelai tensor names
    gguf_path = out_dir / "tiny_h3.gguf"
    writer = gguf.GGUFWriter(str(gguf_path), "wan")
    writer.add_uint32("general.alignment", 32)

    # Globals
    writer.add_tensor("video_patch_proj.weight", model.proj_in.weight.detach().cpu().numpy())
    writer.add_tensor("video_patch_proj.bias", model.proj_in.bias.detach().cpu().numpy())
    writer.add_tensor("audio_patch_proj.weight", model.audio_proj_in.weight.detach().cpu().numpy())
    writer.add_tensor("audio_patch_proj.bias", model.audio_proj_in.bias.detach().cpu().numpy())
    writer.add_tensor("condition_proj.weight", model.context_embedder.weight.detach().cpu().numpy())
    writer.add_tensor("condition_proj.bias", model.context_embedder.bias.detach().cpu().numpy())

    writer.add_tensor("time_embedder.proj_in.weight", model.time_embedder.linear_1.weight.detach().cpu().numpy())
    writer.add_tensor("time_embedder.proj_in.bias", model.time_embedder.linear_1.bias.detach().cpu().numpy())
    writer.add_tensor("time_embedder.proj_out.weight", model.time_embedder.linear_2.weight.detach().cpu().numpy())
    writer.add_tensor("time_embedder.proj_out.bias", model.time_embedder.linear_2.bias.detach().cpu().numpy())

    writer.add_tensor("final_layer.norm.weight", model.norm_out.norm.weight.detach().cpu().numpy())
    writer.add_tensor("final_layer.adaln_proj.linear.weight", model.norm_out.linear.weight.detach().cpu().numpy())
    writer.add_tensor("final_layer.adaln_proj.linear.bias", model.norm_out.linear.bias.detach().cpu().numpy())
    writer.add_tensor("final_layer.video_out.weight", model.proj_out.weight.detach().cpu().numpy())
    writer.add_tensor("final_layer.video_out.bias", model.proj_out.bias.detach().cpu().numpy())
    writer.add_tensor("final_layer.audio_out.weight", model.audio_proj_out.weight.detach().cpu().numpy())
    writer.add_tensor("final_layer.audio_out.bias", model.audio_proj_out.bias.detach().cpu().numpy())

    inv_freq = 1.0 / (10000.0 ** (np.arange(0, 32, 2, dtype=np.float32) / 32.0))
    writer.add_tensor("rope.inv_freq", inv_freq)
    writer.add_tensor("token_refiner.final_norm.weight", model.token_refiner.final_norm.weight.detach().cpu().numpy())

    # Refiner blocks
    for r in range(config["num_refiner_layers"]):
        rb = model.token_refiner.refiner_blocks[r]
        writer.add_tensor(f"token_refiner.blocks.{r}.norm1.weight", rb.norm1.weight.detach().cpu().numpy())
        qkv = torch.cat([rb.attn.to_q.weight, rb.attn.to_k.weight, rb.attn.to_v.weight], dim=0)
        writer.add_tensor(f"token_refiner.blocks.{r}.attn.qkv_proj.weight", qkv.detach().cpu().numpy())
        writer.add_tensor(f"token_refiner.blocks.{r}.attn.q_norm.weight", rb.attn.norm_q.weight.detach().cpu().numpy())
        writer.add_tensor(f"token_refiner.blocks.{r}.attn.k_norm.weight", rb.attn.norm_k.weight.detach().cpu().numpy())
        writer.add_tensor(f"token_refiner.blocks.{r}.attn.out_proj.weight", rb.attn.to_out[0].weight.detach().cpu().numpy())
        writer.add_tensor(f"token_refiner.blocks.{r}.norm2.weight", rb.norm2.weight.detach().cpu().numpy())
        writer.add_tensor(f"token_refiner.blocks.{r}.mlp.fc1.weight", rb.ff.net[0].proj.weight.detach().cpu().numpy())
        writer.add_tensor(f"token_refiner.blocks.{r}.mlp.fc2.weight", rb.ff.net[2].weight.detach().cpu().numpy())

    # Trunk blocks
    for b in range(config["num_layers"]):
        tb = model.transformer_blocks[b]
        writer.add_tensor(f"blocks.{b}.norm1.weight", tb.norm1.weight.detach().cpu().numpy())
        qkv = torch.cat([tb.attn.to_q.weight, tb.attn.to_k.weight, tb.attn.to_v.weight], dim=0)
        writer.add_tensor(f"blocks.{b}.attn.qkv_proj.weight", qkv.detach().cpu().numpy())
        gate_comp = np.zeros((config["hidden_size"], config["hidden_size"]), dtype=np.float32)
        writer.add_tensor(f"blocks.{b}.attn.to_gate_compress.weight", gate_comp)
        writer.add_tensor(f"blocks.{b}.attn.q_norm.weight", tb.attn.norm_q.weight.detach().cpu().numpy())
        writer.add_tensor(f"blocks.{b}.attn.k_norm.weight", tb.attn.norm_k.weight.detach().cpu().numpy())
        writer.add_tensor(f"blocks.{b}.attn.out_proj.weight", tb.attn.to_out[0].weight.detach().cpu().numpy())
        writer.add_tensor(f"blocks.{b}.norm2.weight", tb.norm2.weight.detach().cpu().numpy())
        writer.add_tensor(f"blocks.{b}.mlp.fc1.weight", tb.ff.net[0].proj.weight.detach().cpu().numpy())
        writer.add_tensor(f"blocks.{b}.mlp.fc2.weight", tb.ff.net[2].weight.detach().cpu().numpy())
        writer.add_tensor(f"blocks.{b}.adaln_proj.linear.weight", tb.adaln_proj.linear.weight.detach().cpu().numpy())
        writer.add_tensor(f"blocks.{b}.adaln_proj.linear.bias", tb.adaln_proj.linear.bias.detach().cpu().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote GGUF to {gguf_path}")

    # 2. Run pack_dit and fold_adaln
    pack_dit.pack_dit(str(gguf_path), out_dir, verbose=False)
    fold_adaln.fold_adaln(str(gguf_path), out_dir, verbose=False)
    print(f"Packed trunk, globals, and adaln table in {out_dir}")

    # 3. Compute golden forward passes for steps 0 and 7
    dmd_steps = [999, 874, 749, 624, 500, 375, 250, 125]
    plan = h3_ref.sampler_plan(dmd_steps, 10.0, 3.0)

    golden_outputs = {}
    for step in [0, 7]:
        sp = plan[step]
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
                hidden_states=x_video,
                audio_hidden_states=x_audio,
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
        golden_outputs[step] = (v_v, v_a)

        # Cross-verify with h3_ref
        weights = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
        ref_v, ref_a = h3_ref.dit_forward(
            weights=weights,
            video_latents=x_video.squeeze(0).numpy(),
            audio_latents=x_audio.squeeze(0).numpy(),
            text_embeds=text_emb.squeeze(0).numpy(),
            timesteps=unique_t,
            timestep_indices=timestep_indices.numpy(),
            token_tags=tags.numpy(),
            position_ids=pos.numpy(),
            video_indices=v_idx.numpy(),
            audio_indices=a_idx.numpy(),
            text_indices=t_idx.numpy(),
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
        diff_v = np.linalg.norm(ref_v - v_v) / np.linalg.norm(v_v)
        diff_a = np.linalg.norm(ref_a - v_a) / np.linalg.norm(v_a)
        assert diff_v < 1e-5, f"Step {step} ref_v mismatch: {diff_v}"
        assert diff_a < 1e-5, f"Step {step} ref_a mismatch: {diff_a}"

    # 4. Save raw binary fixtures
    with open(out_dir / "x_video.bin", "wb") as f:
        f.write(x_video.squeeze(0).numpy().tobytes())
    with open(out_dir / "x_audio.bin", "wb") as f:
        f.write(x_audio.squeeze(0).numpy().tobytes())
    with open(out_dir / "text_emb.bin", "wb") as f:
        f.write(text_emb.squeeze(0).numpy().tobytes())
    with open(out_dir / "text_rows.bin", "wb") as f:
        f.write(text_rows.squeeze(0).numpy().tobytes())
    with open(out_dir / "pos.bin", "wb") as f:
        f.write(pos.numpy().tobytes())

    with open(out_dir / "video_v_step0.bin", "wb") as f:
        f.write(golden_outputs[0][0].tobytes())
    with open(out_dir / "audio_v_step0.bin", "wb") as f:
        f.write(golden_outputs[0][1].tobytes())

    with open(out_dir / "video_v_step7.bin", "wb") as f:
        f.write(golden_outputs[7][0].tobytes())
    with open(out_dir / "audio_v_step7.bin", "wb") as f:
        f.write(golden_outputs[7][1].tobytes())

    meta = {
        "num_text": num_text,
        "num_audio": num_audio,
        "num_video": num_video,
        "seq_len": seq_len,
        "hidden_size": config["hidden_size"],
        "num_heads": config["num_attention_heads"],
        "head_dim": config["attention_head_dim"],
        "num_layers": config["num_layers"],
        "num_refiner_layers": config["num_refiner_layers"],
        "ffn_dim": config["ffn_dim"],
        "text_dim": config["text_dim"],
        "video_patch_dim": video_patch_dim,
        "audio_patch_dim": audio_patch_dim,
        "patch_size": list(config["patch_size"]),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"All fixtures successfully generated in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tiny DiT model oracle fixtures")
    parser.add_argument("--out", default=str(ROOT / "tests" / "fixtures" / "tiny_h3"), help="Output directory")
    args = parser.parse_args()
    make_tiny_h3(args.out)


if __name__ == "__main__":
    main()
