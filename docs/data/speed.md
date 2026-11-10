# FastH3 Matrix Multiplication Speed & Performance Reference

Measured baseline on development platform for weight-stationary quantized matrix multiplication.
This benchmark measures the core numerical engine performance (`fh3_matmul`) with on-the-fly dequantization.

## Test Environment

- **CPU:** 13th Gen Intel(R) Core(TM) i5-13420H (8 cores / 12 logical processors: 4 P-cores + 4 E-cores)
- **OS:** Windows 11 (x86_64)
- **Compiler:** GCC 16.2.0 (WinLibs POSIX UCRT x86_64)
- **Compilation Flags:** `-O3 -march=native -fopenmp -ffp-contract=off`
- **Vector Extensions:** AVX2, FMA, F16C

## Measured Benchmark (bench_matmul)

- **Weight Format:** Q4_K (block size 256, 144 bytes per block)
- **Matrix Dimensions:**
  - `in_dim`: 5376 (DiT hidden dimension / projection in)
  - `out_dim`: 14336 (FFN intermediate dimension)
  - `n_tok`: 1024 tokens
- **FLOPs per execution:** $2 \times 5376 \times 14336 \times 1024 = 157.84 \text{ GFLOP}$
- **Memory footprint per matmul:**
  - $W$: 41.3 MB (Q4_K quantized weights)
  - $X$: 21.0 MB (f32 input tokens)
  - $Y$: 56.0 MB (f32 output activations)

### Results

| Metric | Measured Value |
|---|---|
| OpenMP Threads | 12 |
| Average Execution Time | **0.999 s** |
| Best Execution Time | **0.927 s** |
| Average Effective Compute | **158.03 GFLOP/s** |
| Peak Effective Compute | **170.27 GFLOP/s** |

## Numerical Accuracy & Invariants Verified

1. **Quantization Parity:**
   Bit-for-bit identical to `gguf-py` `dequantize()` for all supported formats:
   `F16`, `BF16`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` (`memcmp == 0`).
2. **Deterministic Reduction Tree:**
   Bitwise equality across scalar reference, single-threaded AVX2, and multi-threaded OpenMP tiled execution (`memcmp == 0`).
3. **Reference Parity:**
   Relative L2 error vs float64 NumPy reference: $\text{rel-L2} = 4.21 \times 10^{-7} \le 1.0 \times 10^{-6}$.
