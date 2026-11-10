#!/usr/bin/env python3
"""Repack the DiT from GGUF format into contiguous 4096-aligned layer runs.

Writes:
1. dit_trunk.bin: header + N layer runs.
   Each run contains [qkv, gate_compress, out, fc1, fc2, norm1, norm2, q_norm, k_norm]
   in that exact order, each tensor 32-byte aligned, the run padded to 4096 alignment.
   Scatter-gathers qkv weights from the front of the GGUF with block weights into
   unified layer chunks.
2. dit_globals.bin: refiner blocks, condition_proj, patch projections, output heads,
   final norm, rope inv_freq, time_embedder.
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
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import gguf_remote, hf_range


FH3_PACK_MAGIC = 0x50334846  # 'FH3P'
FH3_PACK_VERSION = 1
FH3_PACK_ALIGNMENT = 4096
FH3_PACK_TENSORS_PER_LAYER = 9

LAYER_TENSOR_SUFFIXES = [
    ("qkv", "attn.qkv_proj.weight"),
    ("gate_compress", "attn.to_gate_compress.weight"),
    ("out", "attn.out_proj.weight"),
    ("fc1", "mlp.fc1.weight"),
    ("fc2", "mlp.fc2.weight"),
    ("norm1", "norm1.weight"),
    ("norm2", "norm2.weight"),
    ("q_norm", "attn.q_norm.weight"),
    ("k_norm", "attn.k_norm.weight"),
]

QUANT_TYPE_MAP = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q8_K": 15,
    "BF16": 30,
}


def _read_tensor_bytes(src: str, header: gguf_remote.GgufHeader, t: gguf_remote.TensorInfo) -> bytes:
    abs_offset = header.data_offset + t.offset
    return hf_range.read(src, abs_offset, abs_offset + t.nbytes)


def pack_dit(
    src: str,
    out_dir: str | Path,
    layers: Optional[Sequence[int]] = None,
    write_globals: bool = True,
    verbose: bool = True,
) -> tuple[Path, Optional[Path]]:
    """Pack DiT model layers into dit_trunk.bin and globals into dit_globals.bin."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trunk_path = out_dir / "dit_trunk.bin"
    globals_path = out_dir / "dit_globals.bin" if write_globals else None

    if verbose:
        print(f"Parsing GGUF header from {src}...")
    hdr = gguf_remote.header(src)
    tensor_map = {t.name: t for t in hdr.tensors}

    # Discover layers
    available_layers: list[int] = []
    for t_name in tensor_map:
        if t_name.startswith("blocks.") and t_name.endswith(".attn.qkv_proj.weight"):
            parts = t_name.split(".")
            available_layers.append(int(parts[1]))
    available_layers = sorted(list(set(available_layers)))

    if layers is not None:
        target_layers = [L for L in available_layers if L in layers]
    else:
        target_layers = available_layers

    num_layers = len(target_layers)

    # Struct packing formats
    # Fh3PackTensorDesc: <64sII4qQQqq32s (168 bytes)
    # Fh3PackLayerDesc: <QQII + 9 * Fh3PackTensorDesc (24 + 1512 = 1536 bytes)
    # Fh3PackHeader: <IIIIQQQQ80s (128 bytes)
    header_size = 128
    layer_desc_size = 1536
    index_offset = header_size
    index_bytes = num_layers * layer_desc_size
    data_offset = ((index_offset + index_bytes + FH3_PACK_ALIGNMENT - 1) // FH3_PACK_ALIGNMENT) * FH3_PACK_ALIGNMENT

    if verbose:
        print(f"Packing {num_layers} layers into {trunk_path}...")

    layer_descs_data = bytearray()
    max_run_bytes = 0

    with open(trunk_path, "wb") as f_trunk:
        # Seek to data_offset to reserve space for header & index table
        f_trunk.seek(data_offset)
        cur_file_offset = data_offset

        for idx, L in enumerate(target_layers):
            if verbose and (idx % 10 == 0 or idx == num_layers - 1):
                print(f"  Packing layer {L} ({idx + 1}/{num_layers})...")

            assert cur_file_offset % FH3_PACK_ALIGNMENT == 0, f"Layer {L} offset {cur_file_offset} not 4096-aligned"

            layer_chunk = bytearray()
            tensor_desc_list: list[bytes] = []

            for short_name, suffix in LAYER_TENSOR_SUFFIXES:
                full_name = f"blocks.{L}.{suffix}"
                t_info = tensor_map.get(full_name)
                if t_info is None:
                    raise KeyError(f"Required tensor {full_name} not found in {src}")

                # 32-byte alignment within layer chunk
                cur_len = len(layer_chunk)
                pad_32 = (32 - (cur_len % 32)) % 32
                if pad_32 > 0:
                    layer_chunk.extend(b"\x00" * pad_32)

                offset_in_layer = len(layer_chunk)
                assert offset_in_layer % 32 == 0

                raw_bytes = _read_tensor_bytes(src, hdr, t_info)
                layer_chunk.extend(raw_bytes)
                size_bytes = len(raw_bytes)

                # Determine in_dim, out_dim, shape
                type_id = QUANT_TYPE_MAP.get(t_info.type, 0)
                n_dims = len(t_info.dims)
                shape = [0, 0, 0, 0]
                for d_i, d in enumerate(t_info.dims[:4]):
                    shape[d_i] = d

                if n_dims >= 2:
                    # In GGUF dims: [ne0, ne1] where ne0 is in_dim, ne1 is out_dim
                    in_dim = t_info.dims[0]
                    out_dim = t_info.dims[1]
                elif n_dims == 1:
                    in_dim = t_info.dims[0]
                    out_dim = 1
                else:
                    in_dim = 1
                    out_dim = 1

                t_name_bytes = short_name.encode("utf-8")[:63]
                t_desc_bytes = struct.pack(
                    "<64sII4qQQqq32s",
                    t_name_bytes,
                    type_id,
                    n_dims,
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[3],
                    offset_in_layer,
                    size_bytes,
                    in_dim,
                    out_dim,
                    b"\x00" * 32,
                )
                assert len(t_desc_bytes) == 168
                tensor_desc_list.append(t_desc_bytes)

            run_unpadded = len(layer_chunk)
            # Pad the layer run to 4096 alignment
            pad_4096 = (FH3_PACK_ALIGNMENT - (run_unpadded % FH3_PACK_ALIGNMENT)) % FH3_PACK_ALIGNMENT
            if pad_4096 > 0:
                layer_chunk.extend(b"\x00" * pad_4096)

            run_size = len(layer_chunk)
            assert run_size % FH3_PACK_ALIGNMENT == 0
            if run_size > max_run_bytes:
                max_run_bytes = run_size

            # Pack Fh3PackLayerDesc: <QQII + 9 * 168 bytes
            l_desc_bytes = struct.pack(
                "<QQII",
                cur_file_offset,
                run_size,
                FH3_PACK_TENSORS_PER_LAYER,
                0,
            ) + b"".join(tensor_desc_list)
            assert len(l_desc_bytes) == layer_desc_size
            layer_descs_data.extend(l_desc_bytes)

            f_trunk.write(layer_chunk)
            cur_file_offset += run_size

        total_file_bytes = cur_file_offset

        # Seek to beginning and write header & index table
        f_trunk.seek(0)
        hdr_bytes = struct.pack(
            "<IIIIQQQQ80s",
            FH3_PACK_MAGIC,
            FH3_PACK_VERSION,
            num_layers,
            FH3_PACK_ALIGNMENT,
            index_offset,
            index_bytes,
            data_offset,
            total_file_bytes,
            b"\x00" * 80,
        )
        assert len(hdr_bytes) == header_size
        f_trunk.write(hdr_bytes)
        f_trunk.write(layer_descs_data)

    if verbose:
        print(
            f"wrote dit_trunk.bin: ~{total_file_bytes / 1e9:.2f} GB across {num_layers} layers, "
            f"largest run ~{max_run_bytes / 1e6:.1f} MB"
        )

    # 2. Write dit_globals.bin if requested
    if write_globals and globals_path is not None:
        if verbose:
            print(f"Writing globals to {globals_path}...")
        non_block_tensors = [t for t in hdr.tensors if not t.name.startswith("blocks.")]

        if _HAS_GGUF:
            writer = gguf.GGUFWriter(str(globals_path), arch="wan")
            for k, v in hdr.kv.items():
                if k != "general.architecture" and not k.startswith("blocks.") and not k.startswith("comfy.gguf.orig_shape.blocks."):
                    if isinstance(v, str):
                        writer.add_string(k, v)
                    elif isinstance(v, bool):
                        writer.add_bool(k, v)
                    elif isinstance(v, int):
                        writer.add_uint32(k, v)
                    elif isinstance(v, float):
                        writer.add_float32(k, v)
                    elif isinstance(v, list) and v and isinstance(v[0], int):
                        writer.add_array(k, v)

            for t in non_block_tensors:
                raw_bytes = _read_tensor_bytes(src, hdr, t)
                qtype = getattr(GGMLQuantizationType, t.type, GGMLQuantizationType.F32)
                if t.type == "F32":
                    arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(t.dims[::-1])
                    writer.add_tensor(t.name, arr)
                elif t.type == "F16":
                    arr = np.frombuffer(raw_bytes, dtype=np.float16).reshape(t.dims[::-1])
                    writer.add_tensor(t.name, arr)
                else:
                    if len(t.dims) >= 2:
                        ne0, ne1 = t.dims[0], t.dims[1]
                        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
                        row_bytes = (ne0 // block_size) * type_size
                        raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(ne1, row_bytes)
                    else:
                        raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    writer.add_tensor(t.name, raw_arr, raw_dtype=qtype)

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
        else:
            # Fallback: simple binary archive if gguf package not available
            with open(globals_path, "wb") as f_glob:
                for t in non_block_tensors:
                    raw_bytes = _read_tensor_bytes(src, hdr, t)
                    f_glob.write(raw_bytes)

        if verbose:
            g_sz = os.path.getsize(globals_path)
            print(f"wrote dit_globals.bin: ~{g_sz / 1e6:.1f} MB ({len(non_block_tensors)} tensors)")

    return trunk_path, globals_path


def parse_layers_arg(layers_arg: Optional[str]) -> Optional[list[int]]:
    if not layers_arg or layers_arg.lower() in ("all", "none"):
        return None
    if ".." in layers_arg:
        start_s, end_s = layers_arg.split("..", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(x.strip()) for x in layers_arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack DiT trunk from GGUF into contiguous layer format")
    parser.add_argument("--src", default=gguf_remote.DEFAULT_DIT_URL if hasattr(gguf_remote, "DEFAULT_DIT_URL") else "https://huggingface.co/realrebelai/FastH3-V2_GGUFs/resolve/main/FastH3-8-Step-V2-Q4_K_M.gguf", help="Path or URL to source GGUF")
    parser.add_argument("--out", default="packed", help="Output directory (default: packed/)")
    parser.add_argument("--layers", default=None, help="Layer range, e.g. 0..50, 0..2 (default: all)")
    parser.add_argument("--no-globals", action="store_true", help="Skip writing dit_globals.bin")
    args = parser.parse_args()

    layers = parse_layers_arg(args.layers)
    pack_dit(
        src=args.src,
        out_dir=args.out,
        layers=layers,
        write_globals=not args.no_globals,
        verbose=True,
    )


if __name__ == "__main__":
    main()
