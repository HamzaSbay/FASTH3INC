#!/usr/bin/env python3
"""Audio VAE decoder packing tool and tiny fixture generator for FastH3.

Packs the BigVGAN audio decoder weights into a compact contiguous binary format:
1. vae_audio_dec.bin:
   - 192-byte binary header with magic 'FH3A', version 1, and architecture metadata.
   - latents_mean and latents_std normalization vectors.
   - Pre-folded weight-normed weights (W = g * v / ||v||) and biases in execution order.
2. Supports --tiny to export a self-contained diffusers test fixture with input
   latents and golden PCM output for bit-accurate parity testing.
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
    from . import hf_range
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import hf_range


FH3_VAE_AUDIO_MAGIC = 0x41334846  # 'FH3A' in little-endian uint32
FH3_VAE_AUDIO_VERSION = 1

DEFAULT_AUDIO_VAE_URL = (
    "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/resolve/main/audio_vae/diffusion_pytorch_model.safetensors"
)
DEFAULT_CONFIG_URL = (
    "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/raw/main/audio_vae/config.json"
)

# Standard FastH3 audio latent normalizer constants
DEFAULT_LATENTS_MEAN = [
    -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373,
    0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
    -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
    -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391,
    -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614,
    -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797
]

DEFAULT_LATENTS_STD = [
    1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828,
    1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
    2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
    1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
    1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363
]


def fold_weight(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fold weight_g and weight_v into effective weight W = g * (v / ||v||)."""
    norm_v = np.linalg.norm(v, axis=tuple(range(1, v.ndim)), keepdims=True)
    return (g * (v / (norm_v + 1e-12))).astype(np.float32)


def pack_header(
    latent_channels: int,
    latent_dim: int,
    decoder_dim: int,
    rates: Sequence[int],
    kernel_sizes: Sequence[int],
    resblock_kernels: Sequence[int],
    resblock_dilations: Sequence[int],
) -> bytes:
    """Pack 192-byte binary header."""
    num_upsamples = len(rates)
    num_resblocks = len(resblock_kernels)
    num_dilations = len(resblock_dilations)
    total_upsample = math.prod(rates)

    rates_padded = list(rates) + [0] * (8 - len(rates))
    kernel_sizes_padded = list(kernel_sizes) + [0] * (8 - len(kernel_sizes))
    res_k_padded = list(resblock_kernels) + [0] * (8 - len(resblock_kernels))
    res_d_padded = list(resblock_dilations) + [0] * (8 - len(resblock_dilations))
    reserved = [0] * 7

    fmt = "<8I 8I 8I 8I 8I I 7I"
    header_bytes = struct.pack(
        fmt,
        FH3_VAE_AUDIO_MAGIC,
        FH3_VAE_AUDIO_VERSION,
        latent_channels,
        latent_dim,
        decoder_dim,
        num_upsamples,
        num_resblocks,
        num_dilations,
        *rates_padded,
        *kernel_sizes_padded,
        *res_k_padded,
        *res_d_padded,
        total_upsample,
        *reserved,
    )
    assert len(header_bytes) == 192, f"Expected header length 192, got {len(header_bytes)}"
    return header_bytes


def write_packed_vae_audio(
    out_path: str | Path,
    latent_channels: int,
    latent_dim: int,
    decoder_dim: int,
    rates: Sequence[int],
    kernel_sizes: Sequence[int],
    resblock_kernels: Sequence[int],
    resblock_dilations: Sequence[int],
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    tensor_getter: Any,
) -> Path:
    """Write packed BigVGAN audio decoder weights to out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_bytes = pack_header(
        latent_channels=latent_channels,
        latent_dim=latent_dim,
        decoder_dim=decoder_dim,
        rates=rates,
        kernel_sizes=kernel_sizes,
        resblock_kernels=resblock_kernels,
        resblock_dilations=resblock_dilations,
    )

    with open(out_path, "wb") as f:
        f.write(header_bytes)

        # latents_mean and latents_std
        mean_arr = np.array(latents_mean, dtype=np.float32)
        std_arr = np.array(latents_std, dtype=np.float32)
        assert len(mean_arr) == latent_channels
        assert len(std_arr) == latent_channels
        f.write(mean_arr.tobytes())
        f.write(std_arr.tobytes())

        # 1. dec_in_proj (Conv1d 1x1: weight [latent_dim, latent_channels], bias [latent_dim])
        w_in = tensor_getter("dec_in_proj.weight")
        if w_in.ndim == 3:
            w_in = w_in.squeeze(-1)
        b_in = tensor_getter("dec_in_proj.bias")
        f.write(w_in.astype(np.float32).tobytes())
        f.write(b_in.astype(np.float32).tobytes())

        # 2. conv_pre (Conv1d: weight [decoder_dim, latent_dim, 7], bias [decoder_dim])
        w_pre = tensor_getter("decoder.conv_pre.weight")
        b_pre = tensor_getter("decoder.conv_pre.bias")
        f.write(w_pre.astype(np.float32).tobytes())
        f.write(b_pre.astype(np.float32).tobytes())

        # 3. Upsample stages
        for i in range(len(rates)):
            w_up = tensor_getter(f"decoder.ups.{i}.0.weight")
            b_up = tensor_getter(f"decoder.ups.{i}.0.bias")
            f.write(w_up.astype(np.float32).tobytes())
            f.write(b_up.astype(np.float32).tobytes())

            for j in range(len(resblock_kernels)):
                res_idx = i * len(resblock_kernels) + j
                for d in range(len(resblock_dilations)):
                    a1 = tensor_getter(f"decoder.resblocks.{res_idx}.activations.{2*d}.act.alpha")
                    beta1 = tensor_getter(f"decoder.resblocks.{res_idx}.activations.{2*d}.act.beta")
                    w1 = tensor_getter(f"decoder.resblocks.{res_idx}.convs1.{d}.weight")
                    bias1 = tensor_getter(f"decoder.resblocks.{res_idx}.convs1.{d}.bias")

                    a2 = tensor_getter(f"decoder.resblocks.{res_idx}.activations.{2*d+1}.act.alpha")
                    beta2 = tensor_getter(f"decoder.resblocks.{res_idx}.activations.{2*d+1}.act.beta")
                    w2 = tensor_getter(f"decoder.resblocks.{res_idx}.convs2.{d}.weight")
                    bias2 = tensor_getter(f"decoder.resblocks.{res_idx}.convs2.{d}.bias")

                    f.write(a1.astype(np.float32).tobytes())
                    f.write(beta1.astype(np.float32).tobytes())
                    f.write(w1.astype(np.float32).tobytes())
                    f.write(bias1.astype(np.float32).tobytes())

                    f.write(a2.astype(np.float32).tobytes())
                    f.write(beta2.astype(np.float32).tobytes())
                    f.write(w2.astype(np.float32).tobytes())
                    f.write(bias2.astype(np.float32).tobytes())

        # 4. Post-processing
        a_post = tensor_getter("decoder.activation_post.act.alpha")
        beta_post = tensor_getter("decoder.activation_post.act.beta")
        w_post = tensor_getter("decoder.conv_post.weight")
        f.write(a_post.astype(np.float32).tobytes())
        f.write(beta_post.astype(np.float32).tobytes())
        f.write(w_post.astype(np.float32).tobytes())

    return out_path


def make_tiny_vae_audio(out_dir: str | Path, seed: int = 42, L: int = 4) -> dict[str, Any]:
    """Generate tiny diffusers audio VAE fixture with packed weights and golden PCM."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

    torch.manual_seed(seed)
    np.random.seed(seed)

    latent_channels = 32
    latent_dim = 32
    decoder_dim = 16
    rates = (2, 2)
    kernels = (4, 4)
    res_kernels = (3, 7, 11)
    res_dilations = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

    latents_mean = [0.05 * (i % 7 - 3) for i in range(latent_channels)]
    latents_std = [1.0 + 0.05 * (i % 5) for i in range(latent_channels)]

    model = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=8,
        latent_dim=latent_dim,
        latent_channels=latent_channels,
        decoder_dim=decoder_dim,
        encoder_rates=(2, 2),
        decoder_rates=rates,
        decoder_kernel_sizes=kernels,
        resblock_kernel_sizes=res_kernels,
        resblock_dilation_sizes=res_dilations,
        latents_mean=latents_mean,
        latents_std=latents_std,
    )
    model.eval()

    state = model.state_dict()

    def get_tiny_tensor(name: str) -> np.ndarray:
        if name + "_g" in state and name + "_v" in state:
            g = state[name + "_g"].numpy()
            v = state[name + "_v"].numpy()
            return fold_weight(g, v)
        elif name in state:
            return state[name].numpy()
        raise KeyError(f"Tensor {name} not found in model state dict")

    # 1. Pack weights to vae_audio_dec.bin
    bin_path = out_dir / "vae_audio_dec.bin"
    write_packed_vae_audio(
        out_path=bin_path,
        latent_channels=latent_channels,
        latent_dim=latent_dim,
        decoder_dim=decoder_dim,
        rates=rates,
        kernel_sizes=kernels,
        resblock_kernels=res_kernels,
        resblock_dilations=(1, 3, 5),
        latents_mean=latents_mean,
        latents_std=latents_std,
        tensor_getter=get_tiny_tensor,
    )

    # 2. Generate test input latents: shape [2, L, latent_channels]
    lat_norm = torch.randn(2, L, latent_channels, dtype=torch.float32)
    lat_path = out_dir / "latents.bin"
    lat_norm.numpy().astype(np.float32).tofile(lat_path)

    # 3. Generate golden PCM with diffusers
    mean_t = torch.tensor(latents_mean, dtype=torch.float32).view(1, 1, latent_channels)
    std_t = torch.tensor(latents_std, dtype=torch.float32).view(1, 1, latent_channels)
    lat_denorm = lat_norm * std_t + mean_t
    lat_for_diffusers = lat_denorm.permute(0, 2, 1)  # [2, 32, L]

    with torch.no_grad():
        pcm_out = model.decode(lat_for_diffusers)[0]  # [2, 1, L * 4]

    pcm_planar = pcm_out.squeeze(1).numpy().astype(np.float32)  # [2, L * 4]
    pcm_path = out_dir / "golden_pcm.bin"
    pcm_planar.tofile(pcm_path)

    meta = {
        "latent_channels": latent_channels,
        "latent_dim": latent_dim,
        "decoder_dim": decoder_dim,
        "rates": list(rates),
        "kernel_sizes": list(kernels),
        "resblock_kernels": list(res_kernels),
        "resblock_dilations": [1, 3, 5],
        "total_upsample": math.prod(rates),
        "num_frames": L,
        "latents_shape": list(lat_norm.shape),
        "pcm_shape": list(pcm_planar.shape),
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def pack_remote_or_local_safetensors(
    src: str,
    out_path: str | Path,
    config_url: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """Pack real BigVGAN audio decoder weights from remote or local safetensors."""
    out_path = Path(out_path)
    if out_path.is_dir() or str(out_path).endswith(("\\", "/")):
        out_path = out_path / "vae_audio_dec.bin"

    if verbose:
        print(f"Reading safetensors header from {src}...")

    # Read 8-byte header size
    hdr_len_bytes = hf_range.read(src, 0, 8)
    hdr_len = struct.unpack("<Q", hdr_len_bytes)[0]
    hdr_bytes = hf_range.read(src, 8, 8 + hdr_len)
    st_hdr = json.loads(hdr_bytes.decode("utf-8"))

    data_base = 8 + hdr_len

    # Load configuration
    latents_mean = DEFAULT_LATENTS_MEAN
    latents_std = DEFAULT_LATENTS_STD
    rates = (5, 5, 2, 2, 2, 2, 2)
    kernel_sizes = (9, 9, 4, 4, 4, 4, 4)
    res_kernels = (3, 7, 11)
    res_dilations = (1, 3, 5)
    latent_channels = 32
    latent_dim = 2048
    decoder_dim = 1024

    cfg_url = config_url or DEFAULT_CONFIG_URL
    try:
        if verbose:
            print(f"Attempting to read audio VAE config from {cfg_url}...")
        import urllib.request
        with urllib.request.urlopen(cfg_url, timeout=10.0) as resp:
            cfg = json.loads(resp.read().decode("utf-8"))
            if "latents_mean" in cfg and cfg["latents_mean"] is not None:
                latents_mean = cfg["latents_mean"]
            if "latents_std" in cfg and cfg["latents_std"] is not None:
                latents_std = cfg["latents_std"]
            if verbose:
                print("Loaded latents_mean and latents_std from config!")
    except Exception as e:
        if verbose:
            print(f"Using default FastH3 latents_mean / std ({e})")

    tensor_cache: dict[str, np.ndarray] = {}

    def fetch_tensor(name: str) -> np.ndarray:
        if name in tensor_cache:
            return tensor_cache[name]
        info = st_hdr[name]
        start = data_base + info["data_offsets"][0]
        end = data_base + info["data_offsets"][1]
        raw = hf_range.read(src, start, end)
        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.uint16,
        }
        dt = dtype_map.get(info["dtype"], np.float32)
        arr = np.frombuffer(raw, dtype=dt).reshape(info["shape"])
        if dt != np.float32:
            arr = arr.astype(np.float32)
        tensor_cache[name] = arr
        return arr

    def tensor_getter(name: str) -> np.ndarray:
        if name + "_g" in st_hdr and name + "_v" in st_hdr:
            g = fetch_tensor(name + "_g")
            v = fetch_tensor(name + "_v")
            return fold_weight(g, v)
        elif name in st_hdr:
            return fetch_tensor(name)
        raise KeyError(f"Tensor {name} not found in safetensors header")

    if verbose:
        print(f"Packing audio VAE decoder tensors into {out_path}...")

    write_packed_vae_audio(
        out_path=out_path,
        latent_channels=latent_channels,
        latent_dim=latent_dim,
        decoder_dim=decoder_dim,
        rates=rates,
        kernel_sizes=kernel_sizes,
        resblock_kernels=res_kernels,
        resblock_dilations=res_dilations,
        latents_mean=latents_mean,
        latents_std=latents_std,
        tensor_getter=tensor_getter,
    )

    if verbose:
        sz = out_path.stat().st_size
        print(f"Successfully wrote {out_path} ({sz / 1e6:.2f} MB)")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack audio VAE (BigVGAN) decoder tensors into binary")
    parser.add_argument("--src", default=DEFAULT_AUDIO_VAE_URL, help="Source safetensors file or URL")
    parser.add_argument("--out", default="packed/vae_audio_dec.bin", help="Output packed binary path")
    parser.add_argument("--config", default=None, help="Optional config.json URL or path")
    parser.add_argument("--tiny", action="store_true", help="Generate tiny diffusers fixture in tests/fixtures/tiny_vae_audio")
    parser.add_argument("--tiny-dir", default="tests/fixtures/tiny_vae_audio", help="Output directory for tiny fixture")
    args = parser.parse_args()

    if args.tiny:
        print(f"Generating tiny diffusers audio VAE fixture in {args.tiny_dir}...")
        meta = make_tiny_vae_audio(args.tiny_dir)
        print(f"Tiny fixture generated successfully: {meta}")
    else:
        pack_remote_or_local_safetensors(args.src, args.out, config_url=args.config)


if __name__ == "__main__":
    main()
