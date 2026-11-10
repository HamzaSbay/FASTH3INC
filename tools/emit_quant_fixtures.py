#!/usr/bin/env python3
"""Emit quantization fixtures and matmul reference for C engine unit tests."""

import json
import os
import struct
import numpy as np
import gguf.quants as gq

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures", "quant")

TYPES_INFO = [
    ("F16",  gq.GGMLQuantizationType.F16,  1,   2),
    ("BF16", gq.GGMLQuantizationType.BF16, 1,   2),
    ("Q8_0", gq.GGMLQuantizationType.Q8_0, 32,  34),
    ("Q2_K", gq.GGMLQuantizationType.Q2_K, 256, 84),
    ("Q3_K", gq.GGMLQuantizationType.Q3_K, 256, 110),
    ("Q4_K", gq.GGMLQuantizationType.Q4_K, 256, 144),
    ("Q5_K", gq.GGMLQuantizationType.Q5_K, 256, 176),
    ("Q6_K", gq.GGMLQuantizationType.Q6_K, 256, 210),
]


def generate_quant_fixtures():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    n_rows = 4
    n_cols = 512
    n_elements = n_rows * n_cols

    for name, qtype, block_size, type_size in TYPES_INFO:
        n_blocks = n_elements // block_size

        if name == "F16":
            x = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
            raw = x.astype(np.float16).view(np.uint8)
            f32 = gq.dequantize(raw, qtype).astype(np.float32)
        elif name == "BF16":
            x = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
            raw = gq.quantize(x, qtype)
            f32 = gq.dequantize(raw, qtype).astype(np.float32)
        elif name == "Q8_0":
            x = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
            raw = gq.quantize(x, qtype)
            f32 = gq.dequantize(raw, qtype).astype(np.float32)
        else:
            # K-quants: synthesize random valid block bytes
            raw_blocks = rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8)
            # Ensure float16 scale / d / dmin fields contain normal, non-zero positive numbers
            for b in range(n_blocks):
                d_val = np.float16(0.5 + 0.1 * (b % 10))
                dmin_val = np.float16(0.05 + 0.01 * (b % 7))
                if name == "Q2_K":
                    # d at 80:82, dmin at 82:84
                    raw_blocks[b, 80:82] = np.frombuffer(d_val.tobytes(), dtype=np.uint8)
                    raw_blocks[b, 82:84] = np.frombuffer(dmin_val.tobytes(), dtype=np.uint8)
                elif name == "Q3_K":
                    # d at 108:110
                    raw_blocks[b, 108:110] = np.frombuffer(d_val.tobytes(), dtype=np.uint8)
                elif name in ("Q4_K", "Q5_K"):
                    # d at 0:2, dmin at 2:4
                    raw_blocks[b, 0:2] = np.frombuffer(d_val.tobytes(), dtype=np.uint8)
                    raw_blocks[b, 2:4] = np.frombuffer(dmin_val.tobytes(), dtype=np.uint8)
                elif name == "Q6_K":
                    # scales at 192:208 (int8), d at 208:210
                    raw_blocks[b, 208:210] = np.frombuffer(d_val.tobytes(), dtype=np.uint8)

            raw = raw_blocks.reshape(-1)
            f32 = gq.dequantize(raw_blocks, qtype).astype(np.float32)

        raw_bytes = raw.tobytes()
        f32_bytes = f32.tobytes()

        # Write binary fixture:
        # Header: uint32 type, uint32 n_rows, uint32 n_cols, uint32 raw_nbytes
        bin_path = os.path.join(FIXTURES_DIR, f"{name}.bin")
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<IIII", int(qtype), n_rows, n_cols, len(raw_bytes)))
            f.write(raw_bytes)
            f.write(f32_bytes)

        # Write JSON fixture
        json_path = os.path.join(FIXTURES_DIR, f"{name}.json")
        doc = {
            "type": int(qtype),
            "type_name": name,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "raw_nbytes": len(raw_bytes),
            "raw_bytes_hex": raw_bytes.hex(),
            "f32_bytes_hex": f32_bytes.hex(),
            "f32_values": f32.flatten().tolist()[:16],  # first 16 values for human inspection
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

        print(f"Generated {name}: {len(raw_bytes)} raw bytes, {f32.size} floats -> {bin_path}")


def generate_matmul_fixture():
    rng = np.random.default_rng(123)
    in_dim = 5376
    out_dim = 192
    n_tok = 37

    # Weight is Q4_K
    qtype = gq.GGMLQuantizationType.Q4_K
    block_size = 256
    type_size = 144
    blocks_per_row = in_dim // block_size
    n_blocks = out_dim * blocks_per_row

    raw_blocks = rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8)
    for b in range(n_blocks):
        d_val = np.float16(0.1 + 0.05 * (b % 8))
        dmin_val = np.float16(0.02 + 0.01 * (b % 5))
        raw_blocks[b, 0:2] = np.frombuffer(d_val.tobytes(), dtype=np.uint8)
        raw_blocks[b, 2:4] = np.frombuffer(dmin_val.tobytes(), dtype=np.uint8)

    w_f32 = gq.dequantize(raw_blocks, qtype).reshape((out_dim, in_dim)).astype(np.float32)
    w_raw_bytes = raw_blocks.tobytes()

    x = rng.standard_normal((n_tok, in_dim)).astype(np.float32)
    bias = rng.standard_normal((out_dim,)).astype(np.float32)

    # Compute high-precision float64 reference
    x_f64 = x.astype(np.float64)
    w_f64 = w_f32.astype(np.float64)
    bias_f64 = bias.astype(np.float64)

    y_ref = (x_f64 @ w_f64.T + bias_f64).astype(np.float32)

    bin_path = os.path.join(FIXTURES_DIR, "matmul_ref.bin")
    with open(bin_path, "wb") as f:
        # Header: uint32 type, uint32 in_dim, uint32 out_dim, uint32 n_tok, uint32 w_bytes
        f.write(struct.pack("<IIIII", int(qtype), in_dim, out_dim, n_tok, len(w_raw_bytes)))
        f.write(w_raw_bytes)
        f.write(x.tobytes())
        f.write(bias.tobytes())
        f.write(y_ref.tobytes())

    json_path = os.path.join(FIXTURES_DIR, "matmul_ref.json")
    doc = {
        "type": int(qtype),
        "type_name": "Q4_K",
        "in_dim": in_dim,
        "out_dim": out_dim,
        "n_tok": n_tok,
        "w_nbytes": len(w_raw_bytes),
        "y_ref_sample": y_ref[0, :8].tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"Generated matmul_ref: in={in_dim}, out={out_dim}, n_tok={n_tok} -> {bin_path}")


if __name__ == "__main__":
    generate_quant_fixtures()
    generate_matmul_fixture()
