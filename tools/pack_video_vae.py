#!/usr/bin/env python3
"""Video ViT VAE decoder packing tool and tiny fixture generator for FastH3.

Packs the MiniMax-H3 Video ViT decoder weights into a compact contiguous binary format:
1. vae_video_dec.bin:
   - 192-byte binary header with magic 'FH3V', version 1, and architecture metadata.
   - latents_mean and latents_std normalization vectors (24 floats each).
   - post_quant_conv weights and bias.
   - decoder.proj_in weights and bias.
   - decoder.register_tokens.
   - Layer transformer blocks (norm1, to_q, to_k, to_v, to_out, scale1, norm2, ff1, ff2, scale2).
   - Final norm (LayerNorm) and proj_out linear weights and bias.
2. Supports --tiny to export a self-contained diffusers test fixture in
   tests/fixtures/tiny_vae_video/ with input latents and golden decoded frames.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

try:
    from . import hf_range
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import hf_range


FH3_VAE_VIDEO_MAGIC = 0x56334846  # 'FH3V' in little-endian uint32
FH3_VAE_VIDEO_VERSION = 1

DEFAULT_VAE_INDEX_URL = (
    "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/raw/main/vae/diffusion_pytorch_model.safetensors.index.json"
)
DEFAULT_CONFIG_URL = (
    "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/raw/main/vae/config.json"
)
DEFAULT_BASE_URL = (
    "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/resolve/main/vae"
)

# Standard FastH3 Video latent normalizer constants from vae/config.json
DEFAULT_LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]

DEFAULT_LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]


def pack_header(
    latent_channels: int,
    out_channels: int,
    dim: int,
    heads: int,
    dim_head: int,
    num_layers: int,
    num_register_tokens: int,
    patch_size_t: int,
    patch_size: int,
    ffn_mult: int,
    clip_length: int,
    token_drop: int,
    rope_dim_ratio: float,
    rope_theta: float,
    norm_eps: float,
) -> bytes:
    """Pack 192-byte binary header."""
    reserved = [0] * 31
    header_bytes = struct.pack(
        "<14I 3f 31I",
        FH3_VAE_VIDEO_MAGIC,
        FH3_VAE_VIDEO_VERSION,
        latent_channels,
        out_channels,
        dim,
        heads,
        dim_head,
        num_layers,
        num_register_tokens,
        patch_size_t,
        patch_size,
        ffn_mult,
        clip_length,
        token_drop,
        rope_dim_ratio,
        rope_theta,
        norm_eps,
        *reserved,
    )
    assert len(header_bytes) == 192, f"Expected header length 192, got {len(header_bytes)}"
    return header_bytes


def write_packed_vae_video(
    out_path: str | Path,
    latent_channels: int,
    out_channels: int,
    dim: int,
    heads: int,
    dim_head: int,
    num_layers: int,
    num_register_tokens: int,
    patch_size_t: int,
    patch_size: int,
    ffn_mult: int,
    clip_length: int,
    token_drop: int,
    rope_dim_ratio: float,
    rope_theta: float,
    norm_eps: float,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    tensor_getter: Callable[[str], np.ndarray],
) -> Path:
    """Write packed Video ViT decoder weights to binary file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap getter to ensure proper float32 unpacking (including bfloat16 bitcast)
    _orig_getter = tensor_getter
    def tensor_getter(k: str) -> np.ndarray:
        t = _orig_getter(k)
        if t.dtype == np.uint16:
            return (t.astype(np.uint32) << 16).view(np.float32)
        elif t.dtype != np.float32:
            return t.astype(np.float32)
        return t

    header_bytes = pack_header(
        latent_channels=latent_channels,
        out_channels=out_channels,
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        num_layers=num_layers,
        num_register_tokens=num_register_tokens,
        patch_size_t=patch_size_t,
        patch_size=patch_size,
        ffn_mult=ffn_mult,
        clip_length=clip_length,
        token_drop=token_drop,
        rope_dim_ratio=rope_dim_ratio,
        rope_theta=rope_theta,
        norm_eps=norm_eps,
    )

    with open(out_path, "wb") as f:
        f.write(header_bytes)

        # 1. latents_mean and latents_std
        mean_arr = np.array(latents_mean, dtype=np.float32)
        std_arr = np.array(latents_std, dtype=np.float32)
        assert len(mean_arr) == latent_channels
        assert len(std_arr) == latent_channels
        f.write(mean_arr.tobytes())
        f.write(std_arr.tobytes())

        # 2. post_quant_conv (1x1 conv3d: weight [latent_channels, latent_channels], bias [latent_channels])
        w_pq = tensor_getter("post_quant_conv.weight")
        if w_pq.ndim == 5:
            w_pq = w_pq.squeeze(-1).squeeze(-1).squeeze(-1)
        b_pq = tensor_getter("post_quant_conv.bias")
        assert w_pq.shape == (latent_channels, latent_channels), f"Bad pq shape: {w_pq.shape}"
        assert b_pq.shape == (latent_channels,), f"Bad pq bias shape: {b_pq.shape}"
        f.write(w_pq.astype(np.float32).tobytes())
        f.write(b_pq.astype(np.float32).tobytes())

        # 3. decoder.proj_in (Linear: weight [dim, latent_channels], bias [dim])
        w_proj_in = tensor_getter("decoder.proj_in.weight")
        b_proj_in = tensor_getter("decoder.proj_in.bias")
        assert w_proj_in.shape == (dim, latent_channels), f"Bad proj_in shape: {w_proj_in.shape}"
        assert b_proj_in.shape == (dim,), f"Bad proj_in bias shape: {b_proj_in.shape}"
        f.write(w_proj_in.astype(np.float32).tobytes())
        f.write(b_proj_in.astype(np.float32).tobytes())

        # 4. decoder.register_tokens [num_register_tokens, dim]
        reg_tokens = tensor_getter("decoder.register_tokens")
        if reg_tokens.ndim == 3:
            reg_tokens = reg_tokens.squeeze(0)
        assert reg_tokens.shape == (num_register_tokens, dim), f"Bad register_tokens shape: {reg_tokens.shape}"
        f.write(reg_tokens.astype(np.float32).tobytes())

        # 5. Transformer blocks
        ffn_dim = dim * ffn_mult
        for l in range(num_layers):
            prefix = f"decoder.transformer_blocks.{l}"

            norm1_w = tensor_getter(f"{prefix}.norm1.weight")
            assert norm1_w.shape == (dim,)
            f.write(norm1_w.astype(np.float32).tobytes())

            to_q_w = tensor_getter(f"{prefix}.attn.to_q.weight")
            to_q_b = tensor_getter(f"{prefix}.attn.to_q.bias")
            assert to_q_w.shape == (dim, dim)
            assert to_q_b.shape == (dim,)
            f.write(to_q_w.astype(np.float32).tobytes())
            f.write(to_q_b.astype(np.float32).tobytes())

            to_k_w = tensor_getter(f"{prefix}.attn.to_k.weight")
            to_k_b = tensor_getter(f"{prefix}.attn.to_k.bias")
            assert to_k_w.shape == (dim, dim)
            assert to_k_b.shape == (dim,)
            f.write(to_k_w.astype(np.float32).tobytes())
            f.write(to_k_b.astype(np.float32).tobytes())

            to_v_w = tensor_getter(f"{prefix}.attn.to_v.weight")
            to_v_b = tensor_getter(f"{prefix}.attn.to_v.bias")
            assert to_v_w.shape == (dim, dim)
            assert to_v_b.shape == (dim,)
            f.write(to_v_w.astype(np.float32).tobytes())
            f.write(to_v_b.astype(np.float32).tobytes())

            to_out_w = tensor_getter(f"{prefix}.attn.to_out.0.weight")
            to_out_b = tensor_getter(f"{prefix}.attn.to_out.0.bias")
            assert to_out_w.shape == (dim, dim)
            assert to_out_b.shape == (dim,)
            f.write(to_out_w.astype(np.float32).tobytes())
            f.write(to_out_b.astype(np.float32).tobytes())

            scale1 = tensor_getter(f"{prefix}.scale1")
            assert scale1.shape == (dim,)
            f.write(scale1.astype(np.float32).tobytes())

            norm2_w = tensor_getter(f"{prefix}.norm2.weight")
            assert norm2_w.shape == (dim,)
            f.write(norm2_w.astype(np.float32).tobytes())

            ff1_w = tensor_getter(f"{prefix}.ff.net.0.proj.weight")
            ff1_b = tensor_getter(f"{prefix}.ff.net.0.proj.bias")
            assert ff1_w.shape == (2 * ffn_dim, dim)
            assert ff1_b.shape == (2 * ffn_dim,)
            f.write(ff1_w.astype(np.float32).tobytes())
            f.write(ff1_b.astype(np.float32).tobytes())

            ff2_w = tensor_getter(f"{prefix}.ff.net.2.weight")
            ff2_b = tensor_getter(f"{prefix}.ff.net.2.bias")
            assert ff2_w.shape == (dim, ffn_dim)
            assert ff2_b.shape == (dim,)
            f.write(ff2_w.astype(np.float32).tobytes())
            f.write(ff2_b.astype(np.float32).tobytes())

            scale2 = tensor_getter(f"{prefix}.scale2")
            assert scale2.shape == (dim,)
            f.write(scale2.astype(np.float32).tobytes())

        # 6. Final LayerNorm (norm_out)
        norm_out_w = tensor_getter("decoder.norm_out.weight")
        norm_out_b = tensor_getter("decoder.norm_out.bias")
        assert norm_out_w.shape == (dim,)
        assert norm_out_b.shape == (dim,)
        f.write(norm_out_w.astype(np.float32).tobytes())
        f.write(norm_out_b.astype(np.float32).tobytes())

        # 7. Patch projection (proj_out: Linear -> out_channels * patch_size_t * patch_size * patch_size)
        out_features = out_channels * patch_size_t * patch_size * patch_size
        proj_out_w = tensor_getter("decoder.proj_out.weight")
        proj_out_b = tensor_getter("decoder.proj_out.bias")
        assert proj_out_w.shape == (out_features, dim), f"Bad proj_out shape: {proj_out_w.shape}"
        assert proj_out_b.shape == (out_features,), f"Bad proj_out bias shape: {proj_out_b.shape}"
        f.write(proj_out_w.astype(np.float32).tobytes())
        f.write(proj_out_b.astype(np.float32).tobytes())

    return out_path


def make_tiny_vae_video(
    out_dir: str | Path,
    seed: int = 42,
    F: int = 7,
    h: int = 2,
    w: int = 2,
) -> dict[str, Any]:
    """Generate tiny diffusers Video ViT VAE fixture with packed weights and golden frames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    torch.manual_seed(seed)
    np.random.seed(seed)

    latent_channels = 24
    out_channels = 3
    num_layers = 2
    heads = 2
    dim_head = 16
    dim = heads * dim_head  # 32
    ffn_mult = 2
    rope_dim_ratio = 0.75
    rope_theta = 100.0
    norm_eps = 1e-5
    num_register_tokens = 4
    clip_length = 17
    token_drop = 3

    latents_mean = [0.05 * (i % 7 - 3) for i in range(latent_channels)]
    latents_std = [1.0 + 0.05 * (i % 5) for i in range(latent_channels)]

    model = AutoencoderKLMiniMaxH3(
        in_channels=out_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
        temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
        decoder_num_layers=num_layers,
        decoder_num_attention_heads=heads,
        decoder_attention_head_dim=dim_head,
        decoder_num_register_tokens=num_register_tokens,
        decoder_ffn_mult=ffn_mult,
        decoder_rope_theta=rope_theta,
        decoder_rope_dim_ratio=rope_dim_ratio,
        decoder_norm_eps=norm_eps,
        clip_length=clip_length,
        token_drop=token_drop,
        latents_mean=latents_mean,
        latents_std=latents_std,
    )
    model.disable_tiling()
    model.eval()

    state = model.state_dict()

    def get_tiny_tensor(name: str) -> np.ndarray:
        t = state[name]
        return t.detach().cpu().float().numpy()

    bin_path = out_dir / "vae_video_dec.bin"
    write_packed_vae_video(
        out_path=bin_path,
        latent_channels=latent_channels,
        out_channels=out_channels,
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        num_layers=num_layers,
        num_register_tokens=num_register_tokens,
        patch_size_t=4,
        patch_size=16,
        ffn_mult=ffn_mult,
        clip_length=clip_length,
        token_drop=token_drop,
        rope_dim_ratio=rope_dim_ratio,
        rope_theta=rope_theta,
        norm_eps=norm_eps,
        latents_mean=latents_mean,
        latents_std=latents_std,
        tensor_getter=get_tiny_tensor,
    )

    # Generate synthetic input latents [1, 24, F, h, w]
    z = torch.randn(1, latent_channels, F, h, w, dtype=torch.float32)

    # Denormalize latents for diffusers decode
    l_mean_t = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    l_std_t = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    z_denorm = z * l_std_t + l_mean_t

    with torch.no_grad():
        raw_dec = model.decode(z_denorm, return_dict=False)[0]

    # Apply ImageNet un-normalization: (x * std + mean).clamp(0, 1)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1)
    golden_frames = (raw_dec.float() * pixel_std + pixel_mean).clamp(0.0, 1.0)

    lat_path = out_dir / "latents.bin"
    golden_path = out_dir / "golden_frames.bin"

    z.numpy().astype(np.float32).tofile(str(lat_path))
    golden_frames.numpy().astype(np.float32).tofile(str(golden_path))

    meta = {
        "latent_channels": latent_channels,
        "out_channels": out_channels,
        "dim": dim,
        "heads": heads,
        "dim_head": dim_head,
        "num_layers": num_layers,
        "num_register_tokens": num_register_tokens,
        "patch_size_t": 4,
        "patch_size": 16,
        "ffn_mult": ffn_mult,
        "clip_length": clip_length,
        "token_drop": token_drop,
        "rope_dim_ratio": rope_dim_ratio,
        "rope_theta": rope_theta,
        "norm_eps": norm_eps,
        "F": F,
        "h": h,
        "w": w,
        "out_frames": int(golden_frames.shape[2]),
        "out_height": int(golden_frames.shape[3]),
        "out_width": int(golden_frames.shape[4]),
        "latents_mean": list(latents_mean),
        "latents_std": list(latents_std),
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def pack_remote_or_local_safetensors(
    src_dir_or_url: str,
    out_path: str | Path,
    config_url: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """Fetch decoder weights from remote or local safetensors into packed format."""
    import requests
    from safetensors import safe_open

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fetch or load config
    if config_url and os.path.exists(config_url):
        with open(config_url, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif config_url and config_url.startswith("http"):
        resp = requests.get(config_url, timeout=30)
        resp.raise_for_status()
        cfg = resp.json()
    else:
        # Try local vae/config.json or remote default
        loc_cfg = Path(src_dir_or_url) / "config.json" if os.path.isdir(src_dir_or_url) else None
        if loc_cfg and loc_cfg.exists():
            with open(loc_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            resp = requests.get(DEFAULT_CONFIG_URL, timeout=30)
            resp.raise_for_status()
            cfg = resp.json()

    latent_channels = cfg.get("latent_channels", 24)
    out_channels = cfg.get("out_channels", 3)
    dim = cfg.get("decoder_num_attention_heads", 32) * cfg.get("decoder_attention_head_dim", 64)
    heads = cfg.get("decoder_num_attention_heads", 32)
    dim_head = cfg.get("decoder_attention_head_dim", 64)
    num_layers = cfg.get("decoder_num_layers", 36)
    num_register_tokens = cfg.get("decoder_num_register_tokens", 4)
    patch_size_t = math.prod(cfg.get("temporal_downsample_factors", (1, 2, 2, 1, 1, 1)))
    patch_size = math.prod(cfg.get("spatial_downsample_factors", (2, 2, 2, 2, 1, 1)))
    ffn_mult = cfg.get("decoder_ffn_mult", 4)
    clip_length = cfg.get("clip_length", 17)
    token_drop = cfg.get("token_drop", 3)
    rope_dim_ratio = float(cfg.get("decoder_rope_dim_ratio", 0.75))
    rope_theta = float(cfg.get("decoder_rope_theta", 100.0))
    norm_eps = float(cfg.get("decoder_norm_eps", 1e-5))
    latents_mean = cfg.get("latents_mean", DEFAULT_LATENTS_MEAN)
    latents_std = cfg.get("latents_std", DEFAULT_LATENTS_STD)

    # 2. Check for local safetensors files
    if os.path.isdir(src_dir_or_url):
        # Local dir with safetensors
        shards = list(Path(src_dir_or_url).glob("*.safetensors"))
        handles = [safe_open(str(s), framework="np") for s in shards]

        def get_tensor(name: str) -> np.ndarray:
            for h in handles:
                if name in h.keys():
                    return h.get_tensor(name)
            raise KeyError(f"Tensor {name} not found in {src_dir_or_url}")

        return write_packed_vae_video(
            out_path=out_path,
            latent_channels=latent_channels,
            out_channels=out_channels,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            num_layers=num_layers,
            num_register_tokens=num_register_tokens,
            patch_size_t=patch_size_t,
            patch_size=patch_size,
            ffn_mult=ffn_mult,
            clip_length=clip_length,
            token_drop=token_drop,
            rope_dim_ratio=rope_dim_ratio,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            latents_mean=latents_mean,
            latents_std=latents_std,
            tensor_getter=get_tensor,
        )

    # Remote range fetch via index json
    index_url = DEFAULT_VAE_INDEX_URL
    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()
    index_data = resp.json()
    weight_map = index_data["weight_map"]

    # Cache safetensors headers for shards
    shard_headers: dict[str, dict[str, Any]] = {}

    def get_remote_tensor(name: str) -> np.ndarray:
        if name not in weight_map:
            raise KeyError(f"Tensor {name} not in index")
        shard_file = weight_map[name]
        shard_url = f"{DEFAULT_BASE_URL}/{shard_file}"

        if shard_file not in shard_headers:
            # Read 8 bytes header size
            hdr_len_bytes = hf_range.read(shard_url, 0, 8)
            hdr_len = struct.unpack("<Q", hdr_len_bytes)[0]
            hdr_json_bytes = hf_range.read(shard_url, 8, 8 + hdr_len)
            shard_headers[shard_file] = (json.loads(hdr_json_bytes.decode("utf-8")), 8 + hdr_len)

        hdr, data_base = shard_headers[shard_file]
        tensor_info = hdr[name]
        dtype_str = tensor_info["dtype"]
        shape = tensor_info["shape"]
        offsets = tensor_info["data_offsets"]

        raw = hf_range.read(shard_url, data_base + offsets[0], data_base + offsets[1])

        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.uint16,
        }
        arr = np.frombuffer(raw, dtype=dtype_map.get(dtype_str, np.float32)).reshape(shape)
        if dtype_str == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    return write_packed_vae_video(
        out_path=out_path,
        latent_channels=latent_channels,
        out_channels=out_channels,
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        num_layers=num_layers,
        num_register_tokens=num_register_tokens,
        patch_size_t=patch_size_t,
        patch_size=patch_size,
        ffn_mult=ffn_mult,
        clip_length=clip_length,
        token_drop=token_drop,
        rope_dim_ratio=rope_dim_ratio,
        rope_theta=rope_theta,
        norm_eps=norm_eps,
        latents_mean=latents_mean,
        latents_std=latents_std,
        tensor_getter=get_remote_tensor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack Video ViT VAE decoder weights into binary")
    parser.add_argument("--src", default=DEFAULT_BASE_URL, help="Source safetensors directory or URL")
    parser.add_argument("--out", default="packed/vae_video_dec.bin", help="Output packed binary path")
    parser.add_argument("--config", default=None, help="Optional config.json URL or path")
    parser.add_argument("--tiny", action="store_true", help="Generate tiny diffusers fixture in tests/fixtures/tiny_vae_video")
    parser.add_argument("--tiny-dir", default="tests/fixtures/tiny_vae_video", help="Output directory for tiny fixture")
    args = parser.parse_args()

    if args.tiny:
        print(f"Generating tiny diffusers Video VAE fixture in {args.tiny_dir}...")
        meta = make_tiny_vae_video(args.tiny_dir)
        print(f"Tiny Video VAE fixture generated successfully: {meta}")
    else:
        pack_remote_or_local_safetensors(args.src, args.out, config_url=args.config)


if __name__ == "__main__":
    main()
