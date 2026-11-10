#!/usr/bin/env python3
"""Emit reference fixtures for MiniMax-H3 elementwise ops and dense attention."""

from __future__ import annotations

import json
import os
import struct
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import tools.ref.h3_ref as h3_ref

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "fixtures",
    "ops",
)


def emit_rmsnorm_fixture(rng: np.random.Generator):
    n = 8
    dim = 5376
    eps = 1e-5

    x = rng.standard_normal((n, dim)).astype(np.float32)
    w = (1.0 + 0.2 * rng.standard_normal((dim,))).astype(np.float32)

    ref = h3_ref.rms_norm(x, w, eps=eps).astype(np.float32)

    bin_path = os.path.join(FIXTURES_DIR, "rmsnorm.bin")
    with open(bin_path, "wb") as f:
        # header: uint32 n, uint32 dim, float32 eps
        f.write(struct.pack("<IIf", n, dim, eps))
        f.write(x.tobytes())
        f.write(w.tobytes())
        f.write(ref.tobytes())

    json_path = os.path.join(FIXTURES_DIR, "rmsnorm.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "dim": dim,
                "eps": eps,
                "ref_sample": ref[0, :8].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Emitted RMSNorm: n={n}, dim={dim} -> {bin_path}")


def emit_rope3_fixture(rng: np.random.Generator):
    n = 16
    heads = 4
    head_dim = 128
    rot_dim = 96

    x = rng.standard_normal((n, heads, head_dim)).astype(np.float32)

    # Generate 3-axis positions and build RoPE tables
    pos = rng.uniform(0.0, 100.0, size=(n, 3))
    cos, sin = h3_ref.build_rope(pos, rope_freq_dim=16, rope_theta=10000.0)
    cos = cos.astype(np.float32)
    sin = sin.astype(np.float32)
    assert cos.shape[-1] == rot_dim

    ref = h3_ref.apply_rotary_emb(x, cos, sin).astype(np.float32)

    # Bug verification: rot_dim = 128 must differ significantly
    cos128 = np.concatenate([cos, np.ones((n, head_dim - rot_dim), dtype=np.float32)], axis=-1)
    sin128 = np.concatenate([sin, np.zeros((n, head_dim - rot_dim), dtype=np.float32)], axis=-1)
    bug_128 = h3_ref.apply_rotary_emb(x, cos128, sin128)
    diff = np.max(np.abs(ref - bug_128))
    assert diff > 1.0, f"Plausible bug rot_dim=128 should fail with diff > 1.0, got {diff}"

    bin_path = os.path.join(FIXTURES_DIR, "rope3.bin")
    with open(bin_path, "wb") as f:
        # header: uint32 n, uint32 heads, uint32 head_dim, uint32 rot_dim
        f.write(struct.pack("<IIII", n, heads, head_dim, rot_dim))
        f.write(x.tobytes())
        f.write(cos.tobytes())
        f.write(sin.tobytes())
        f.write(ref.tobytes())

    json_path = os.path.join(FIXTURES_DIR, "rope3.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "heads": heads,
                "head_dim": head_dim,
                "rot_dim": rot_dim,
                "bug_128_max_diff": float(diff),
                "ref_sample": ref[0, 0, :8].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Emitted RoPE3: n={n}, heads={heads}, hd={head_dim}, rot_dim={rot_dim} -> {bin_path}")


def emit_swiglu_fixture(rng: np.random.Generator):
    n = 8
    F = 1536

    # in has shape (n, 2 * F)
    inp = rng.standard_normal((n, 2 * F)).astype(np.float32)

    val, gate = np.split(inp, 2, axis=-1)
    ref = (val * h3_ref.silu(gate)).astype(np.float32)

    # Bug verification: swapping val and gate must fail
    bug_swapped = (gate * h3_ref.silu(val)).astype(np.float32)
    diff = np.max(np.abs(ref - bug_swapped))
    assert diff > 1.0, f"Plausible bug SwiGLU swapped halves should fail with diff > 1.0, got {diff}"

    bin_path = os.path.join(FIXTURES_DIR, "swiglu.bin")
    with open(bin_path, "wb") as f:
        # header: uint32 n, uint32 F
        f.write(struct.pack("<II", n, F))
        f.write(inp.tobytes())
        f.write(ref.tobytes())

    json_path = os.path.join(FIXTURES_DIR, "swiglu.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "F": F,
                "bug_swapped_max_diff": float(diff),
                "ref_sample": ref[0, :8].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Emitted SwiGLU: n={n}, F={F} -> {bin_path}")


def emit_modulate_fixture(rng: np.random.Generator):
    n = 8
    dim = 5376

    x = rng.standard_normal((n, dim)).astype(np.float32)
    scale = rng.standard_normal((n, dim)).astype(np.float32)
    shift = rng.standard_normal((n, dim)).astype(np.float32)

    ref = (x * (1.0 + scale) + shift).astype(np.float32)

    # Bug verification: x * scale + shift must fail
    bug_noplus1 = (x * scale + shift).astype(np.float32)
    diff = np.max(np.abs(ref - bug_noplus1))
    assert diff > 1.0, f"Plausible bug modulate x*scale+shift should fail with diff > 1.0, got {diff}"

    bin_path = os.path.join(FIXTURES_DIR, "modulate.bin")
    with open(bin_path, "wb") as f:
        # header: uint32 n, uint32 dim
        f.write(struct.pack("<II", n, dim))
        f.write(x.tobytes())
        f.write(shift.tobytes())
        f.write(scale.tobytes())
        f.write(ref.tobytes())

    json_path = os.path.join(FIXTURES_DIR, "modulate.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "dim": dim,
                "bug_noplus1_max_diff": float(diff),
                "ref_sample": ref[0, :8].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Emitted Modulate: n={n}, dim={dim} -> {bin_path}")


def emit_gate_residual_fixture(rng: np.random.Generator):
    n = 8
    dim = 5376

    h = rng.standard_normal((n, dim)).astype(np.float32)
    gate = rng.standard_normal((n, dim)).astype(np.float32)
    y = rng.standard_normal((n, dim)).astype(np.float32)

    ref = (h + gate * y).astype(np.float32)

    bin_path = os.path.join(FIXTURES_DIR, "gate_residual.bin")
    with open(bin_path, "wb") as f:
        # header: uint32 n, uint32 dim
        f.write(struct.pack("<II", n, dim))
        f.write(h.tobytes())
        f.write(gate.tobytes())
        f.write(y.tobytes())
        f.write(ref.tobytes())

    json_path = os.path.join(FIXTURES_DIR, "gate_residual.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "dim": dim,
                "ref_sample": ref[0, :8].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Emitted GateResidual: n={n}, dim={dim} -> {bin_path}")


def emit_attention_fixture(rng: np.random.Generator):
    # Non-tile multiple N=777 specifically required
    n = 777
    heads = 4
    hd = 128

    q = (rng.standard_normal((n, heads, hd)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((n, heads, hd)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((n, heads, hd)) * 0.5).astype(np.float32)

    ref = h3_ref.dense_attention(q, k, v).astype(np.float32)

    bin_path = os.path.join(FIXTURES_DIR, "attention.bin")
    with open(bin_path, "wb") as f:
        # header: uint32 n, uint32 heads, uint32 hd
        f.write(struct.pack("<III", n, heads, hd))
        f.write(q.tobytes())
        f.write(k.tobytes())
        f.write(v.tobytes())
        f.write(ref.tobytes())

    json_path = os.path.join(FIXTURES_DIR, "attention.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "heads": heads,
                "hd": hd,
                "ref_sample": ref[0, 0, :8].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Emitted Attention: n={n}, heads={heads}, hd={hd} -> {bin_path}")


def emit_timestep_embed_fixture():
    nt = 5
    freq_dim = 256
    t = np.array([0.0, 0.125, 0.5, 0.874, 1.0], dtype=np.float64)

    ref = h3_ref.get_timestep_embedding(
        t, freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
    ).astype(np.float32)

    bin_path = os.path.join(FIXTURES_DIR, "timestep_embed.bin")
    with open(bin_path, "wb") as f:
        # header: uint32 nt, uint32 freq_dim
        f.write(struct.pack("<II", nt, freq_dim))
        f.write(t.tobytes())
        f.write(ref.tobytes())

    json_path = os.path.join(FIXTURES_DIR, "timestep_embed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "nt": nt,
                "freq_dim": freq_dim,
                "t": t.tolist(),
                "ref_sample": ref[0, :8].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Emitted TimestepEmbed: nt={nt}, freq_dim={freq_dim} -> {bin_path}")


def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(20261110)

    emit_rmsnorm_fixture(rng)
    emit_rope3_fixture(rng)
    emit_swiglu_fixture(rng)
    emit_modulate_fixture(rng)
    emit_gate_residual_fixture(rng)
    emit_attention_fixture(rng)
    emit_timestep_embed_fixture()
    print("All ops fixtures emitted successfully.")


if __name__ == "__main__":
    main()
