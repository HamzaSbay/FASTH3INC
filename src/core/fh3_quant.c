#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "fh3/fh3.h"
#include "core/fh3_quant.h"

size_t fh3_row_bytes(int type, int64_t n) {
    if (n <= 0) return 0;
    switch (type) {
        case FH3_QUANT_F32:  return (size_t)n * sizeof(float);
        case FH3_QUANT_F16:  return (size_t)n * sizeof(uint16_t);
        case FH3_QUANT_Q4_0: return (size_t)((n + 31) / 32) * 18;
        case FH3_QUANT_Q4_1: return (size_t)((n + 31) / 32) * 20;
        case FH3_QUANT_Q5_0: return (size_t)((n + 31) / 32) * 22;
        case FH3_QUANT_Q5_1: return (size_t)((n + 31) / 32) * 24;
        case FH3_QUANT_Q8_0: return (size_t)((n + 31) / 32) * 34;
        case FH3_QUANT_Q8_1: return (size_t)((n + 31) / 32) * 40;
        case FH3_QUANT_Q2_K: return (size_t)((n + 255) / 256) * 84;
        case FH3_QUANT_Q3_K: return (size_t)((n + 255) / 256) * 110;
        case FH3_QUANT_Q4_K: return (size_t)((n + 255) / 256) * 144;
        case FH3_QUANT_Q5_K: return (size_t)((n + 255) / 256) * 176;
        case FH3_QUANT_Q6_K: return (size_t)((n + 255) / 256) * 210;
        case FH3_QUANT_Q8_K: return (size_t)((n + 255) / 256) * 292;
        case FH3_QUANT_BF16: return (size_t)n * sizeof(uint16_t);
        default:             return 0;
    }
}

static void dequant_f16(const void *src, float *dst, int64_t n) {
    const uint16_t *s = (const uint16_t *)src;
    int64_t i = 0;
#if defined(__F16C__) && defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(s + i));
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
#endif
    for (; i < n; i++) {
        dst[i] = fh3_f16_to_f32(s[i]);
    }
}

static void dequant_bf16(const void *src, float *dst, int64_t n) {
    const uint16_t *s = (const uint16_t *)src;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = fh3_bf16_to_f32(s[i]);
    }
}

static void dequant_q8_0(const void *src, float *dst, int64_t n) {
    int64_t n_blocks = n / 32;
    const uint8_t *b_ptr = (const uint8_t *)src;
    for (int64_t b = 0; b < n_blocks; b++) {
        uint16_t d_raw;
        memcpy(&d_raw, b_ptr, 2);
        float d = fh3_f16_to_f32(d_raw);
        const int8_t *qs = (const int8_t *)(b_ptr + 2);
        for (int i = 0; i < 32; i++) {
            dst[b * 32 + i] = (float)qs[i] * d;
        }
        b_ptr += 34;
    }
}

static void dequant_q2_k(const void *src, float *dst, int64_t n) {
    int64_t n_blocks = n / 256;
    const uint8_t *b_ptr = (const uint8_t *)src;
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t *scales = b_ptr;
        const uint8_t *qs = b_ptr + 16;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, b_ptr + 80, 2);
        memcpy(&dmin_raw, b_ptr + 82, 2);
        float d = fh3_f16_to_f32(d_raw);
        float dmin = fh3_f16_to_f32(dmin_raw);
        float *out = dst + b * 256;

        for (int c = 0; c < 2; c++) {
            const uint8_t *q_chunk = qs + c * 32;
            for (int s = 0; s < 4; s++) {
                int shift = s * 2;
                int g0 = (c * 4 + s) * 2;
                int g1 = g0 + 1;
                float dl0 = d * (float)(scales[g0] & 0x0F);
                float ml0 = dmin * (float)(scales[g0] >> 4);
                float dl1 = d * (float)(scales[g1] & 0x0F);
                float ml1 = dmin * (float)(scales[g1] >> 4);
                for (int l = 0; l < 16; l++) {
                    uint8_t val = (q_chunk[l] >> shift) & 3;
                    *out++ = dl0 * (float)val - ml0;
                }
                for (int l = 0; l < 16; l++) {
                    uint8_t val = (q_chunk[16 + l] >> shift) & 3;
                    *out++ = dl1 * (float)val - ml1;
                }
            }
        }
        b_ptr += 84;
    }
}

static void dequant_q3_k(const void *src, float *dst, int64_t n) {
    int64_t n_blocks = n / 256;
    const uint8_t *b_ptr = (const uint8_t *)src;
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t *hmask = b_ptr;
        const uint8_t *qs = b_ptr + 32;
        const uint8_t *scales_raw = b_ptr + 96;
        uint16_t d_raw;
        memcpy(&d_raw, b_ptr + 108, 2);
        float d = fh3_f16_to_f32(d_raw);
        float *out = dst + b * 256;

        const uint8_t *lscales = scales_raw;      /* 8 bytes */
        const uint8_t *hscales = scales_raw + 8;  /* 4 bytes */
        float dl[16];
        for (int g = 0; g < 16; g++) {
            uint8_t l_val = (g < 8) ? (lscales[g] & 0x0F) : ((lscales[g - 8] >> 4) & 0x0F);
            int shift_h = (g / 4) * 2;
            int byte_h = g % 4;
            uint8_t h_val = (hscales[byte_h] >> shift_h) & 0x03;
            int8_t sc = (int8_t)((l_val & 0x0F) | (h_val << 4)) - 32;
            dl[g] = d * (float)sc;
        }

        for (int g = 0; g < 16; g++) {
            int c = g / 8;
            int rem = g % 8;
            int s = rem / 2;
            int shift_l = s * 2;
            int offset_l = c * 32 + (rem % 2) * 16;
            int shift_h = g / 2;
            int offset_h = (g % 2) * 16;
            float d_val = dl[g];
            for (int l = 0; l < 16; l++) {
                uint8_t ql_val = (qs[offset_l + l] >> shift_l) & 3;
                uint8_t qh_val = ((hmask[offset_h + l] >> shift_h) & 1) ^ 1;
                float q_val = (float)((int8_t)ql_val - (int8_t)(qh_val << 2));
                *out++ = d_val * q_val;
            }
        }
        b_ptr += 110;
    }
}

static void dequant_q4_k(const void *src, float *dst, int64_t n) {
    int64_t n_blocks = n / 256;
    const uint8_t *b_ptr = (const uint8_t *)src;
    for (int64_t b = 0; b < n_blocks; b++) {
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, b_ptr, 2);
        memcpy(&dmin_raw, b_ptr + 2, 2);
        float d = fh3_f16_to_f32(d_raw);
        float dmin = fh3_f16_to_f32(dmin_raw);
        const uint8_t *scales_raw = b_ptr + 4;
        const uint8_t *qs = b_ptr + 16;
        float *out = dst + b * 256;

        float dl[8], ml[8];
        for (int j = 0; j < 4; j++) {
            uint8_t sc0 = scales_raw[j] & 0x3F;
            uint8_t m0  = scales_raw[4 + j] & 0x3F;
            uint8_t sc1 = (scales_raw[8 + j] & 0x0F) | ((scales_raw[j] >> 2) & 0x30);
            uint8_t m1  = (scales_raw[8 + j] >> 4)   | ((scales_raw[4 + j] >> 2) & 0x30);
            dl[j]     = d * (float)sc0;
            ml[j]     = dmin * (float)m0;
            dl[4 + j] = d * (float)sc1;
            ml[4 + j] = dmin * (float)m1;
        }

        for (int c = 0; c < 4; c++) {
            int g0 = c * 2;
            int g1 = c * 2 + 1;
            const uint8_t *q_chunk = qs + c * 32;
            float d0 = dl[g0], m0 = ml[g0];
            float d1 = dl[g1], m1 = ml[g1];
            for (int l = 0; l < 32; l++) {
                float v0 = (float)(q_chunk[l] & 0x0F);
                out[g0 * 32 + l] = d0 * v0 - m0;
            }
            for (int l = 0; l < 32; l++) {
                float v1 = (float)(q_chunk[l] >> 4);
                out[g1 * 32 + l] = d1 * v1 - m1;
            }
        }
        b_ptr += 144;
    }
}

static void dequant_q5_k(const void *src, float *dst, int64_t n) {
    int64_t n_blocks = n / 256;
    const uint8_t *b_ptr = (const uint8_t *)src;
    for (int64_t b = 0; b < n_blocks; b++) {
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, b_ptr, 2);
        memcpy(&dmin_raw, b_ptr + 2, 2);
        float d = fh3_f16_to_f32(d_raw);
        float dmin = fh3_f16_to_f32(dmin_raw);
        const uint8_t *scales_raw = b_ptr + 4;
        const uint8_t *qh = b_ptr + 16;
        const uint8_t *qs = b_ptr + 48;
        float *out = dst + b * 256;

        float dl[8], ml[8];
        for (int j = 0; j < 4; j++) {
            uint8_t sc0 = scales_raw[j] & 0x3F;
            uint8_t m0  = scales_raw[4 + j] & 0x3F;
            uint8_t sc1 = (scales_raw[8 + j] & 0x0F) | ((scales_raw[j] >> 2) & 0x30);
            uint8_t m1  = (scales_raw[8 + j] >> 4)   | ((scales_raw[4 + j] >> 2) & 0x30);
            dl[j]     = d * (float)sc0;
            ml[j]     = dmin * (float)m0;
            dl[4 + j] = d * (float)sc1;
            ml[4 + j] = dmin * (float)m1;
        }

        for (int g = 0; g < 8; g++) {
            int c = g / 2;
            int shift_ql = (g % 2) * 4;
            int shift_qh = g;
            float d_val = dl[g];
            float m_val = ml[g];
            const uint8_t *q_chunk = qs + c * 32;
            for (int l = 0; l < 32; l++) {
                uint8_t ql_val = (q_chunk[l] >> shift_ql) & 0x0F;
                uint8_t qh_val = (qh[l] >> shift_qh) & 0x01;
                float q_val = (float)(ql_val | (qh_val << 4));
                out[g * 32 + l] = d_val * q_val - m_val;
            }
        }
        b_ptr += 176;
    }
}

static void dequant_q6_k(const void *src, float *dst, int64_t n) {
    int64_t n_blocks = n / 256;
    const uint8_t *b_ptr = (const uint8_t *)src;
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t *ql = b_ptr;
        const uint8_t *qh = b_ptr + 128;
        const int8_t *scales = (const int8_t *)(b_ptr + 192);
        uint16_t d_raw;
        memcpy(&d_raw, b_ptr + 208, 2);
        float d = fh3_f16_to_f32(d_raw);
        float *out = dst + b * 256;

        float dl[16];
        for (int i = 0; i < 16; i++) {
            dl[i] = d * (float)scales[i];
        }

        for (int g = 0; g < 16; g++) {
            int c_ql = g / 8;
            int rem_ql = g % 8;
            int shift_ql = (rem_ql / 4) * 4;
            int sub_ql = rem_ql % 4;
            int off_ql = c_ql * 64 + sub_ql * 16;

            int c_qh = g / 8;
            int rem_qh = g % 8;
            int shift_qh = (rem_qh / 2) * 2;
            int sub_qh = rem_qh % 2;
            int off_qh = c_qh * 32 + sub_qh * 16;

            float d_val = dl[g];
            for (int l = 0; l < 16; l++) {
                uint8_t ql_val = (ql[off_ql + l] >> shift_ql) & 0x0F;
                uint8_t qh_val = (qh[off_qh + l] >> shift_qh) & 0x03;
                int8_t q_val = (int8_t)(ql_val | (qh_val << 4)) - 32;
                out[g * 16 + l] = d_val * (float)q_val;
            }
        }
        b_ptr += 210;
    }
}

void fh3_dequant(int type, const void *src, float *dst, int64_t n) {
    if (!src || !dst || n <= 0) return;
    switch (type) {
        case FH3_QUANT_F32:
            memcpy(dst, src, (size_t)n * sizeof(float));
            break;
        case FH3_QUANT_F16:
            dequant_f16(src, dst, n);
            break;
        case FH3_QUANT_BF16:
            dequant_bf16(src, dst, n);
            break;
        case FH3_QUANT_Q8_0:
            dequant_q8_0(src, dst, n);
            break;
        case FH3_QUANT_Q2_K:
            dequant_q2_k(src, dst, n);
            break;
        case FH3_QUANT_Q3_K:
            dequant_q3_k(src, dst, n);
            break;
        case FH3_QUANT_Q4_K:
            dequant_q4_k(src, dst, n);
            break;
        case FH3_QUANT_Q5_K:
            dequant_q5_k(src, dst, n);
            break;
        case FH3_QUANT_Q6_K:
            dequant_q6_k(src, dst, n);
            break;
        default:
            fprintf(stderr, "fh3_dequant: unsupported type %d\n", type);
            break;
    }
}
