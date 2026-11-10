#!/usr/bin/env python3
"""Fold AdaLayerNorm (AdaLN) linear projections for fixed DMD2 timesteps.

Evaluates adaln_proj.linear across all DiT blocks for the 8 fixed DMD2 steps
and folds them into adaln_table.bin (f32).

For remote URL sources, weights are fetched block by block into RAM, folded,
and dropped immediately, avoiding writing ~7.33 GB of raw AdaLN weights to disk.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import gguf
    from gguf.constants import GGMLQuantizationType
    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False

try:
    from . import gguf_remote, hf_range
    from .ref import h3_ref
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import gguf_remote, hf_range
    from tools.ref import h3_ref


# Format constants matching fh3_pack.h
FH3_ADALN_MAGIC = 0x41334846  # 'FH3A'
FH3_ADALN_VERSION = 1
DEFAULT_DMD_STEPS = (999, 874, 749, 624, 500, 375, 250, 125)
DEFAULT_VIDEO_SHIFT = 10.0
DEFAULT_AUDIO_SHIFT = 3.0
DEFAULT_HIDDEN_SIZE = 5376
DEFAULT_FREQ_DIM = 256
PAIRS_PER_STEP = 3  # 0: video, 1: text, 2: audio


def compute_schedule_hash(
    dmd_steps: Sequence[int],
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    audio_shift: float = DEFAULT_AUDIO_SHIFT,
) -> int:
    """Compute 64-bit FNV-1a hash of the denoising schedule configuration."""
    n_steps = len(dmd_steps)
    buf = struct.pack(f"<i{n_steps}idd", n_steps, *dmd_steps, float(video_shift), float(audio_shift))
    h = 0xcbf29ce484222325
    for b in buf:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _dequantize_tensor(raw_bytes: bytes, type_name: str, dims: list[int], orig_shape: Optional[list[int]] = None) -> np.ndarray:
    """Dequantize raw tensor bytes into float32 array."""
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    if type_name == "F32":
        arr = data.view(np.float32)
    elif type_name == "F16":
        arr = data.view(np.float16).astype(np.float32)
    elif type_name == "BF16":
        u16 = data.view(np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif _HAS_GGUF:
        qtype = getattr(GGMLQuantizationType, type_name, None)
        if qtype is None:
            raise ValueError(f"Unsupported quantization type: {type_name}")
        arr = gguf.quants.dequantize(data, qtype)
    else:
        raise RuntimeError(f"Cannot dequantize {type_name} without 'gguf' package")

    # Reshape
    if orig_shape:
        arr = arr.reshape(orig_shape)
    elif len(dims) > 1:
        # GGUF stored dims are fast-axis first [ne0, ne1, ...] -> reshape [ne1, ne0]
        arr = arr.reshape(dims[::-1])
    else:
        arr = arr.reshape(-1)
    return arr


def _read_tensor_data(src: str, header: gguf_remote.GgufHeader, t: gguf_remote.TensorInfo) -> bytes:
    abs_offset = header.data_offset + t.offset
    return hf_range.read(src, abs_offset, abs_offset + t.nbytes)


def fold_adaln(
    src: str,
    out_dir: str | Path,
    layers: Optional[Sequence[int]] = None,
    dmd_steps: Sequence[int] = DEFAULT_DMD_STEPS,
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    audio_shift: float = DEFAULT_AUDIO_SHIFT,
    verbose: bool = True,
) -> Path:
    """Fold AdaLN linear projections into adaln_table.bin."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adaln_table.bin"

    if verbose:
        print(f"Parsing GGUF header from {src}...")
    hdr = gguf_remote.header(src)
    tensor_map = {t.name: t for t in hdr.tensors}

    # Discover available transformer blocks
    block_indices: list[int] = []
    for t_name in tensor_map:
        if t_name.startswith("blocks.") and t_name.endswith(".adaln_proj.linear.weight"):
            parts = t_name.split(".")
            block_indices.append(int(parts[1]))
    block_indices = sorted(list(set(block_indices)))

    if layers is not None:
        block_indices = [b for b in block_indices if b in layers]

    n_blocks = len(block_indices)
    n_steps = len(dmd_steps)
    sched_hash = compute_schedule_hash(dmd_steps, video_shift, audio_shift)

    # 1. Fetch time_embedder weights
    if verbose:
        print("Fetching time_embedder weights...")
    t_in_w = tensor_map["time_embedder.proj_in.weight"]
    t_in_b = tensor_map["time_embedder.proj_in.bias"]
    t_out_w = tensor_map["time_embedder.proj_out.weight"]
    t_out_b = tensor_map["time_embedder.proj_out.bias"]

    w_proj_in = _dequantize_tensor(_read_tensor_data(src, hdr, t_in_w), t_in_w.type, t_in_w.dims, t_in_w.orig_shape)
    b_proj_in = _dequantize_tensor(_read_tensor_data(src, hdr, t_in_b), t_in_b.type, t_in_b.dims, t_in_b.orig_shape)
    w_proj_out = _dequantize_tensor(_read_tensor_data(src, hdr, t_out_w), t_out_w.type, t_out_w.dims, t_out_w.orig_shape)
    b_proj_out = _dequantize_tensor(_read_tensor_data(src, hdr, t_out_b), t_out_b.type, t_out_b.dims, t_out_b.orig_shape)

    # In GGUF / PyTorch, Linear(in_features, out_features) weight is (out_features, in_features).
    # If loaded as (out, in), out = x @ w.T + b. Let's ensure proper 2D shape.
    if w_proj_in.shape[0] != b_proj_in.shape[0] and w_proj_in.shape[1] == b_proj_in.shape[0]:
        w_proj_in = w_proj_in.T
    if w_proj_out.shape[0] != b_proj_out.shape[0] and w_proj_out.shape[1] == b_proj_out.shape[0]:
        w_proj_out = w_proj_out.T

    # Determine freq_dim dynamically from proj_in input dimension
    freq_dim = int(w_proj_in.shape[1])

    # 2. Compute sampler plan and timesteps
    plan = h3_ref.sampler_plan(dmd_steps, video_shift=video_shift, audio_shift=audio_shift)

    def eval_temb(t_val: float) -> np.ndarray:
        t_freq = h3_ref.get_timestep_embedding(
            np.array([t_val], dtype=np.float32), freq_dim, flip_sin_to_cos=True
        )
        h = _silu(t_freq @ w_proj_in.T + b_proj_in)
        temb = h @ w_proj_out.T + b_proj_out
        return temb

    # Precompute silu(temb) for each step
    # For each step: act_v for video & text, act_a for audio
    step_acts: list[tuple[np.ndarray, np.ndarray]] = []
    for s in range(n_steps):
        sp = plan[s]
        t_v = float(1.0 - sp.sigma_v)
        t_a = float(1.0 - sp.sigma_a)
        act_v = _silu(eval_temb(t_v))  # shape (1, 2688)
        act_a = _silu(eval_temb(t_a))  # shape (1, 2688)
        step_acts.append((act_v, act_a))

    # Determine hidden_size from block 0 norm1 or bias
    b0_norm = tensor_map.get("blocks.0.norm1.weight")
    hidden_size = b0_norm.dims[0] if b0_norm else DEFAULT_HIDDEN_SIZE

    # Header calculations
    header_size = 128
    trunk_offset = header_size
    trunk_elements = n_steps * n_blocks * PAIRS_PER_STEP * 6 * hidden_size
    trunk_bytes = trunk_elements * 4

    has_final = "final_layer.adaln_proj.linear.weight" in tensor_map
    final_offset = trunk_offset + trunk_bytes
    final_elements = (n_steps * PAIRS_PER_STEP * 2 * hidden_size) if has_final else 0
    final_bytes = final_elements * 4
    total_file_bytes = final_offset + final_bytes

    if verbose:
        print(f"Writing {out_path} ({n_blocks} blocks, {n_steps} steps, total ~{total_file_bytes / 1e9:.2f} GB)...")

    # Open binary output file
    with open(out_path, "wb") as f:
        # Write header placeholder
        hdr_bytes = struct.pack(
            "<IIIIIIQQQQQ64s",
            FH3_ADALN_MAGIC,
            FH3_ADALN_VERSION,
            n_steps,
            n_blocks,
            PAIRS_PER_STEP,
            hidden_size,
            sched_hash,
            trunk_offset,
            trunk_bytes,
            final_offset,
            final_bytes,
            b"\x00" * 64,
        )
        assert len(hdr_bytes) == header_size
        f.write(hdr_bytes)

        # Allocate buffer for trunk table: [n_steps, n_blocks, PAIRS_PER_STEP, 6, hidden_size]
        # We process block by block to minimize RAM footprint:
        # For each block: fetch weights, compute across all steps, store in [n_steps, PAIRS_PER_STEP, 6, hidden_size]
        block_tables: list[np.ndarray] = []  # shape per block: (n_steps, 3, 6, hidden_size)

        for b_idx, block_num in enumerate(block_indices):
            w_name = f"blocks.{block_num}.adaln_proj.linear.weight"
            b_name = f"blocks.{block_num}.adaln_proj.linear.bias"
            t_w = tensor_map[w_name]
            t_b = tensor_map[b_name]

            if verbose and (b_idx % 10 == 0 or b_idx == n_blocks - 1):
                print(f"  Folding block {block_num} ({b_idx + 1}/{n_blocks})...")

            w_raw = _read_tensor_data(src, hdr, t_w)
            b_raw = _read_tensor_data(src, hdr, t_b)

            w_lin = _dequantize_tensor(w_raw, t_w.type, t_w.dims, t_w.orig_shape)
            b_lin = _dequantize_tensor(b_raw, t_b.type, t_b.dims, t_b.orig_shape)
            del w_raw, b_raw

            if w_lin.shape[0] != b_lin.shape[0] and w_lin.shape[1] == b_lin.shape[0]:
                w_lin = w_lin.T

            # Pre-evaluate for all 8 steps for this block
            # Shape for this block: (n_steps, PAIRS_PER_STEP, 6, hidden_size)
            block_data = np.zeros((n_steps, PAIRS_PER_STEP, 6, hidden_size), dtype=np.float32)

            for s in range(n_steps):
                act_v, act_a = step_acts[s]
                # proj_v: shape (1, 96768) -> (3, 6, hidden_size)
                proj_v = (act_v @ w_lin.T + b_lin).reshape(3, 6, hidden_size)
                # proj_a: shape (1, 96768) -> (3, 6, hidden_size)
                proj_a = (act_a @ w_lin.T + b_lin).reshape(3, 6, hidden_size)

                # Pair 0: video (modality 0)
                block_data[s, 0, :, :] = proj_v[0]
                # Pair 1: text (modality 1)
                block_data[s, 1, :, :] = proj_v[1]
                # Pair 2: audio (modality 2)
                block_data[s, 2, :, :] = proj_a[2]

            block_tables.append(block_data)
            del w_lin, b_lin

        # Transpose / assemble into [n_steps, n_blocks, PAIRS_PER_STEP, 6, hidden_size]
        if n_blocks > 0:
            trunk_arr = np.stack(block_tables, axis=1)  # (n_steps, n_blocks, 3, 6, hidden_size)
            f.write(trunk_arr.tobytes())
            del trunk_arr, block_tables

        # Process final layer adaln table: [n_steps, PAIRS_PER_STEP, 2, hidden_size]
        if has_final:
            if verbose:
                print("  Folding final_layer.adaln_proj.linear...")
            tf_w = tensor_map["final_layer.adaln_proj.linear.weight"]
            tf_b = tensor_map["final_layer.adaln_proj.linear.bias"]
            wf_lin = _dequantize_tensor(_read_tensor_data(src, hdr, tf_w), tf_w.type, tf_w.dims, tf_w.orig_shape)
            bf_lin = _dequantize_tensor(_read_tensor_data(src, hdr, tf_b), tf_b.type, tf_b.dims, tf_b.orig_shape)

            if wf_lin.shape[0] != bf_lin.shape[0] and wf_lin.shape[1] == bf_lin.shape[0]:
                wf_lin = wf_lin.T

            final_data = np.zeros((n_steps, PAIRS_PER_STEP, 2, hidden_size), dtype=np.float32)
            for s in range(n_steps):
                act_v, act_a = step_acts[s]
                proj_fv = (act_v @ wf_lin.T + bf_lin).reshape(2, hidden_size)
                proj_fa = (act_a @ wf_lin.T + bf_lin).reshape(2, hidden_size)

                # Pair 0: video
                final_data[s, 0, :, :] = proj_fv
                # Pair 1: text
                final_data[s, 1, :, :] = proj_fv
                # Pair 2: audio
                final_data[s, 2, :, :] = proj_fa

            f.write(final_data.tobytes())
            del wf_lin, bf_lin, final_data

    actual_size = os.path.getsize(out_path)
    if verbose:
        print(f"wrote adaln_table.bin ≈ {actual_size / 1e9:.2f} GB ({actual_size} bytes)")
    return out_path


def parse_layers_arg(layers_arg: Optional[str]) -> Optional[list[int]]:
    if not layers_arg or layers_arg.lower() in ("all", "none"):
        return None
    if ".." in layers_arg:
        start_s, end_s = layers_arg.split("..", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(x.strip()) for x in layers_arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold AdaLN projections into adaln_table.bin")
    parser.add_argument("--src", default=gguf_remote.DEFAULT_DIT_URL if hasattr(gguf_remote, "DEFAULT_DIT_URL") else "https://huggingface.co/realrebelai/FastH3-V2_GGUFs/resolve/main/FastH3-8-Step-V2-Q4_K_M.gguf", help="Path or URL to source GGUF")
    parser.add_argument("--out", default="packed", help="Output directory (default: packed/)")
    parser.add_argument("--layers", default=None, help="Layer range, e.g. 0..50, 0..2 (default: all)")
    parser.add_argument("--video-shift", type=float, default=DEFAULT_VIDEO_SHIFT, help="Video shift factor")
    parser.add_argument("--audio-shift", type=float, default=DEFAULT_AUDIO_SHIFT, help="Audio shift factor")
    args = parser.parse_args()

    layers = parse_layers_arg(args.layers)
    fold_adaln(
        src=args.src,
        out_dir=args.out,
        layers=layers,
        video_shift=args.video_shift,
        audio_shift=args.audio_shift,
        verbose=True,
    )


if __name__ == "__main__":
    main()
