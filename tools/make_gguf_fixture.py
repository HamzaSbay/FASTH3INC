#!/usr/bin/env python3
"""Generate tiny GGUF v3 test fixture containing 3 tensors (F32, F16, Q4_K)
and a comfy.gguf.orig_shape metadata entry for unit testing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("Error: 'gguf' Python package not found. Run in project .venv.")


def make_tiny_fixture(out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(str(out_path), "tiny")

    # Metadata
    writer.add_uint32("general.alignment", 32)
    writer.add_array("comfy.gguf.orig_shape.x", [96, 5376])

    # 1. F32 tensor [4, 4] -> 16 elements = 64 bytes
    f32_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    writer.add_tensor("t_f32", f32_data)

    # 2. F16 tensor [4, 4] -> 16 elements = 32 bytes
    f16_data = np.arange(16, dtype=np.float16).reshape(4, 4)
    writer.add_tensor("t_f16", f16_data)

    # 3. Q4_K tensor 'x'
    # Original shape [96, 5376] has 516,096 elements.
    # Q4_K block size = 256, type size = 144 bytes.
    # 516096 // 256 = 2016 blocks -> 2016 * 144 = 290,304 bytes.
    # Stored 1D dimension is 516,096 elements.
    q4k_raw = np.zeros(290304, dtype=np.uint8)
    # Fill with deterministic pseudo-pattern
    for i in range(len(q4k_raw)):
        q4k_raw[i] = (i * 37 + 13) & 0xFF

    writer.add_tensor("x", q4k_raw, raw_dtype=gguf.GGMLQuantizationType.Q4_K)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Generated fixture: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    default_dest = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gguf" / "tiny.gguf"
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dest
    make_tiny_fixture(dest)
