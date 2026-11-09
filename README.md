# FastH3-in-C (`fh3`)

A dependency-free C99 engine that runs **FastH3-8-Step-V2** (35B MiniMax-H3, text → synchronized video + stereo audio) on consumer CPU hardware. It keeps the model weights on the NVMe drive and streams only the parts each phase needs, inspired by `kimi-k3-in-c`.

## Key Features

- **Dependency-Free Runtime:** Pure C99 (`-std=gnu99`), OpenMP, and `libm`. No PyTorch, BLAS, or GPU runtime required.
- **NVMe Trunk Streaming:** Cyclic pinned prefix + 2-slot ring buffer prevents memory bloat while scanning 50 DiT blocks across 8 denoising steps.
- **Offline AdaLN Folding:** 13B modulation parameters are pre-projected into an exact ~0.16 GB table for the fixed 8-step DMD schedule, eliminating 37% of DiT weights from the streaming hot path.
- **Decoupled Three-Phase Pipeline:**
  1. Qwen3-VL text encoder (layers 0–49, un-normed output)
  2. MiniMax-H3 DiT (50 blocks, VSA-H3 sparse attention, 8 steps)
  3. ViT video decoder (clip-chunked) + BigVGAN audio decoder (32 kHz)

## Reductions Ledger (DiT Bytes per Step)

| Stage | DiT Bytes Touched per Step |
|---|---|
| Official BF16 Transformer | 70.10 GB |
| realrebelai Q4_K_M GGUF | 19.84 GB |
| − AdaLN folded into exact table | 12.52 GB |
| − Run-once components (refiner, condition proj) | ≈ 11.92 GB |
| RAM floor for DiT phase (2 ring slots + activations) | ≈ 1.5 GB |

## Building and Testing

Requirements:
- C99 compiler with OpenMP support (e.g. GCC 13+ / MinGW-w64)
- CMake 3.20+
- Python 3.11+ (tools and reference scripts only)

```bash
# Configure and build
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build

# Run unit tests
ctest --test-dir build --output-on-failure
```

## License and Attribution

See [NOTICE](NOTICE) for license information and model weight attribution.
Model weights are governed by the MiniMax H3 Community License.
