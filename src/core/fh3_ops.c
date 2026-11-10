#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <omp.h>

#include "fh3/fh3.h"

#define TILE_OUT 64

/* Deterministic 8-lane reduction tree matching scalar and AVX2 bitwise */
static inline float reduce_tree_8(const float acc[8]) {
    float s01 = acc[0] + acc[1];
    float s23 = acc[2] + acc[3];
    float s45 = acc[4] + acc[5];
    float s67 = acc[6] + acc[7];
    float s03 = s01 + s23;
    float s47 = s45 + s67;
    return s03 + s47;
}

static inline float dot_product_scalar(const float *x, const float *w, int64_t n) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int64_t n8 = n - (n % 8);
    for (int64_t k = 0; k < n8; k += 8) {
        acc[0] = fmaf(x[k + 0], w[k + 0], acc[0]);
        acc[1] = fmaf(x[k + 1], w[k + 1], acc[1]);
        acc[2] = fmaf(x[k + 2], w[k + 2], acc[2]);
        acc[3] = fmaf(x[k + 3], w[k + 3], acc[3]);
        acc[4] = fmaf(x[k + 4], w[k + 4], acc[4]);
        acc[5] = fmaf(x[k + 5], w[k + 5], acc[5]);
        acc[6] = fmaf(x[k + 6], w[k + 6], acc[6]);
        acc[7] = fmaf(x[k + 7], w[k + 7], acc[7]);
    }
    float dot = reduce_tree_8(acc);
    for (int64_t k = n8; k < n; k++) {
        dot = fmaf(x[k], w[k], dot);
    }
    return dot;
}

#if defined(__AVX2__)
static inline float dot_product_avx2(const float *x, const float *w, int64_t n) {
    __m256 vacc = _mm256_setzero_ps();
    int64_t n8 = n - (n % 8);
    for (int64_t k = 0; k < n8; k += 8) {
        __m256 vx = _mm256_loadu_ps(x + k);
        __m256 vw = _mm256_loadu_ps(w + k);
        vacc = _mm256_fmadd_ps(vx, vw, vacc);
    }
    float acc[8];
    _mm256_storeu_ps(acc, vacc);
    float dot = reduce_tree_8(acc);
    for (int64_t k = n8; k < n; k++) {
        dot = fmaf(x[k], w[k], dot);
    }
    return dot;
}

static inline void dot_product_avx2_4x(const float *x, const float *w_base, int64_t in_dim,
                                       const float *bias_4, float *out_4) {
    const float *w0 = w_base;
    const float *w1 = w_base + in_dim;
    const float *w2 = w_base + 2 * in_dim;
    const float *w3 = w_base + 3 * in_dim;

    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    __m256 vacc2 = _mm256_setzero_ps();
    __m256 vacc3 = _mm256_setzero_ps();

    int64_t n8 = in_dim - (in_dim % 8);
    for (int64_t k = 0; k < n8; k += 8) {
        __m256 vx = _mm256_loadu_ps(x + k);
        vacc0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w0 + k), vacc0);
        vacc1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w1 + k), vacc1);
        vacc2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w2 + k), vacc2);
        vacc3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w3 + k), vacc3);
    }

    float acc0[8], acc1[8], acc2[8], acc3[8];
    _mm256_storeu_ps(acc0, vacc0);
    _mm256_storeu_ps(acc1, vacc1);
    _mm256_storeu_ps(acc2, vacc2);
    _mm256_storeu_ps(acc3, vacc3);

    float d0 = reduce_tree_8(acc0);
    float d1 = reduce_tree_8(acc1);
    float d2 = reduce_tree_8(acc2);
    float d3 = reduce_tree_8(acc3);

    for (int64_t k = n8; k < in_dim; k++) {
        d0 = fmaf(x[k], w0[k], d0);
        d1 = fmaf(x[k], w1[k], d1);
        d2 = fmaf(x[k], w2[k], d2);
        d3 = fmaf(x[k], w3[k], d3);
    }

    if (bias_4) {
        d0 += bias_4[0];
        d1 += bias_4[1];
        d2 += bias_4[2];
        d3 += bias_4[3];
    }

    out_4[0] = d0;
    out_4[1] = d1;
    out_4[2] = d2;
    out_4[3] = d3;
}

static inline void dot_product_avx2_2tx4r(const float *xt0, const float *xt1,
                                          const float *w_base, int64_t in_dim,
                                          const float *bias_4,
                                          float *out0_4, float *out1_4) {
    const float *w0 = w_base;
    const float *w1 = w_base + in_dim;
    const float *w2 = w_base + 2 * in_dim;
    const float *w3 = w_base + 3 * in_dim;

    __m256 vacc00 = _mm256_setzero_ps();
    __m256 vacc01 = _mm256_setzero_ps();
    __m256 vacc02 = _mm256_setzero_ps();
    __m256 vacc03 = _mm256_setzero_ps();

    __m256 vacc10 = _mm256_setzero_ps();
    __m256 vacc11 = _mm256_setzero_ps();
    __m256 vacc12 = _mm256_setzero_ps();
    __m256 vacc13 = _mm256_setzero_ps();

    int64_t n8 = in_dim - (in_dim % 8);
    for (int64_t k = 0; k < n8; k += 8) {
        __m256 vx0 = _mm256_loadu_ps(xt0 + k);
        __m256 vx1 = _mm256_loadu_ps(xt1 + k);

        __m256 vw0 = _mm256_loadu_ps(w0 + k);
        vacc00 = _mm256_fmadd_ps(vx0, vw0, vacc00);
        vacc10 = _mm256_fmadd_ps(vx1, vw0, vacc10);

        __m256 vw1 = _mm256_loadu_ps(w1 + k);
        vacc01 = _mm256_fmadd_ps(vx0, vw1, vacc01);
        vacc11 = _mm256_fmadd_ps(vx1, vw1, vacc11);

        __m256 vw2 = _mm256_loadu_ps(w2 + k);
        vacc02 = _mm256_fmadd_ps(vx0, vw2, vacc02);
        vacc12 = _mm256_fmadd_ps(vx1, vw2, vacc12);

        __m256 vw3 = _mm256_loadu_ps(w3 + k);
        vacc03 = _mm256_fmadd_ps(vx0, vw3, vacc03);
        vacc13 = _mm256_fmadd_ps(vx1, vw3, vacc13);
    }

    float a00[8], a01[8], a02[8], a03[8];
    float a10[8], a11[8], a12[8], a13[8];
    _mm256_storeu_ps(a00, vacc00);
    _mm256_storeu_ps(a01, vacc01);
    _mm256_storeu_ps(a02, vacc02);
    _mm256_storeu_ps(a03, vacc03);
    _mm256_storeu_ps(a10, vacc10);
    _mm256_storeu_ps(a11, vacc11);
    _mm256_storeu_ps(a12, vacc12);
    _mm256_storeu_ps(a13, vacc13);

    float d00 = reduce_tree_8(a00);
    float d01 = reduce_tree_8(a01);
    float d02 = reduce_tree_8(a02);
    float d03 = reduce_tree_8(a03);

    float d10 = reduce_tree_8(a10);
    float d11 = reduce_tree_8(a11);
    float d12 = reduce_tree_8(a12);
    float d13 = reduce_tree_8(a13);

    for (int64_t k = n8; k < in_dim; k++) {
        d00 = fmaf(xt0[k], w0[k], d00);
        d01 = fmaf(xt0[k], w1[k], d01);
        d02 = fmaf(xt0[k], w2[k], d02);
        d03 = fmaf(xt0[k], w3[k], d03);

        d10 = fmaf(xt1[k], w0[k], d10);
        d11 = fmaf(xt1[k], w1[k], d11);
        d12 = fmaf(xt1[k], w2[k], d12);
        d13 = fmaf(xt1[k], w3[k], d13);
    }

    if (bias_4) {
        d00 += bias_4[0];
        d01 += bias_4[1];
        d02 += bias_4[2];
        d03 += bias_4[3];

        d10 += bias_4[0];
        d11 += bias_4[1];
        d12 += bias_4[2];
        d13 += bias_4[3];
    }

    out0_4[0] = d00; out0_4[1] = d01; out0_4[2] = d02; out0_4[3] = d03;
    out1_4[0] = d10; out1_4[1] = d11; out1_4[2] = d12; out1_4[3] = d13;
}
#endif

void fh3_matmul_scalar(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null) {
    if (!w || !x || !y || n_tok <= 0 || w->in <= 0 || w->out <= 0) return;

    size_t row_bytes = fh3_row_bytes(w->type, w->in);
    float *tile_w = (float *)malloc(TILE_OUT * (size_t)w->in * sizeof(float));
    if (!tile_w) return;

    for (int64_t o0 = 0; o0 < w->out; o0 += TILE_OUT) {
        int64_t tile_rows = (o0 + TILE_OUT <= w->out) ? TILE_OUT : (w->out - o0);

        for (int64_t r = 0; r < tile_rows; r++) {
            int64_t o = o0 + r;
            const uint8_t *src_row = w->data + (size_t)o * row_bytes;
            float *dst_row = tile_w + (size_t)r * (size_t)w->in;
            fh3_dequant(w->type, src_row, dst_row, w->in);
        }

        for (int t = 0; t < n_tok; t++) {
            const float *xt = x + (size_t)t * (size_t)w->in;
            float *yt = y + (size_t)t * (size_t)w->out;
            for (int64_t r = 0; r < tile_rows; r++) {
                int64_t o = o0 + r;
                const float *wr = tile_w + (size_t)r * (size_t)w->in;
                float dot = dot_product_scalar(xt, wr, w->in);
                if (bias_or_null) dot += bias_or_null[o];
                yt[o] = dot;
            }
        }
    }

    free(tile_w);
}

void fh3_matmul_avx2(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null) {
#if !defined(__AVX2__)
    fh3_matmul_scalar(w, x, y, n_tok, bias_or_null);
#else
    if (!w || !x || !y || n_tok <= 0 || w->in <= 0 || w->out <= 0) return;

    size_t row_bytes = fh3_row_bytes(w->type, w->in);
    float *tile_w = (float *)malloc(TILE_OUT * (size_t)w->in * sizeof(float));
    if (!tile_w) return;

    for (int64_t o0 = 0; o0 < w->out; o0 += TILE_OUT) {
        int64_t tile_rows = (o0 + TILE_OUT <= w->out) ? TILE_OUT : (w->out - o0);

        for (int64_t r = 0; r < tile_rows; r++) {
            int64_t o = o0 + r;
            const uint8_t *src_row = w->data + (size_t)o * row_bytes;
            float *dst_row = tile_w + (size_t)r * (size_t)w->in;
            fh3_dequant(w->type, src_row, dst_row, w->in);
        }

        int t = 0;
        for (; t + 2 <= n_tok; t += 2) {
            const float *xt0 = x + (size_t)t * (size_t)w->in;
            const float *xt1 = x + (size_t)(t + 1) * (size_t)w->in;
            float *yt0 = y + (size_t)t * (size_t)w->out;
            float *yt1 = y + (size_t)(t + 1) * (size_t)w->out;

            int64_t r = 0;
            for (; r + 4 <= tile_rows; r += 4) {
                int64_t o_base = o0 + r;
                dot_product_avx2_2tx4r(xt0, xt1,
                                       tile_w + (size_t)r * (size_t)w->in, w->in,
                                       bias_or_null ? bias_or_null + o_base : NULL,
                                       yt0 + o_base, yt1 + o_base);
            }
            for (; r < tile_rows; r++) {
                int64_t o = o0 + r;
                const float *wr = tile_w + (size_t)r * (size_t)w->in;
                float d0 = dot_product_avx2(xt0, wr, w->in);
                float d1 = dot_product_avx2(xt1, wr, w->in);
                if (bias_or_null) {
                    d0 += bias_or_null[o];
                    d1 += bias_or_null[o];
                }
                yt0[o] = d0;
                yt1[o] = d1;
            }
        }

        for (; t < n_tok; t++) {
            const float *xt = x + (size_t)t * (size_t)w->in;
            float *yt = y + (size_t)t * (size_t)w->out;

            int64_t r = 0;
            for (; r + 4 <= tile_rows; r += 4) {
                int64_t o_base = o0 + r;
                dot_product_avx2_4x(xt, tile_w + (size_t)r * (size_t)w->in, w->in,
                                    bias_or_null ? bias_or_null + o_base : NULL,
                                    yt + o_base);
            }
            for (; r < tile_rows; r++) {
                int64_t o = o0 + r;
                const float *wr = tile_w + (size_t)r * (size_t)w->in;
                float dot = dot_product_avx2(xt, wr, w->in);
                if (bias_or_null) dot += bias_or_null[o];
                yt[o] = dot;
            }
        }
    }

    free(tile_w);
#endif
}

void fh3_matmul(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null) {
    if (!w || !x || !y || n_tok <= 0 || w->in <= 0 || w->out <= 0) return;

    size_t row_bytes = fh3_row_bytes(w->type, w->in);

    #pragma omp parallel
    {
        float *tile_w = (float *)malloc(TILE_OUT * (size_t)w->in * sizeof(float));
        if (tile_w) {
            #pragma omp for schedule(static)
            for (int64_t o0 = 0; o0 < w->out; o0 += TILE_OUT) {
                int64_t tile_rows = (o0 + TILE_OUT <= w->out) ? TILE_OUT : (w->out - o0);

                /* Dequantize tile once for all tokens */
                for (int64_t r = 0; r < tile_rows; r++) {
                    int64_t o = o0 + r;
                    const uint8_t *src_row = w->data + (size_t)o * row_bytes;
                    float *dst_row = tile_w + (size_t)r * (size_t)w->in;
                    fh3_dequant(w->type, src_row, dst_row, w->in);
                }

#if defined(__AVX2__)
                int t = 0;
                for (; t + 2 <= n_tok; t += 2) {
                    const float *xt0 = x + (size_t)t * (size_t)w->in;
                    const float *xt1 = x + (size_t)(t + 1) * (size_t)w->in;
                    float *yt0 = y + (size_t)t * (size_t)w->out;
                    float *yt1 = y + (size_t)(t + 1) * (size_t)w->out;

                    int64_t r = 0;
                    for (; r + 4 <= tile_rows; r += 4) {
                        int64_t o_base = o0 + r;
                        dot_product_avx2_2tx4r(xt0, xt1,
                                               tile_w + (size_t)r * (size_t)w->in, w->in,
                                               bias_or_null ? bias_or_null + o_base : NULL,
                                               yt0 + o_base, yt1 + o_base);
                    }
                    for (; r < tile_rows; r++) {
                        int64_t o = o0 + r;
                        const float *wr = tile_w + (size_t)r * (size_t)w->in;
                        float d0 = dot_product_avx2(xt0, wr, w->in);
                        float d1 = dot_product_avx2(xt1, wr, w->in);
                        if (bias_or_null) {
                            d0 += bias_or_null[o];
                            d1 += bias_or_null[o];
                        }
                        yt0[o] = d0;
                        yt1[o] = d1;
                    }
                }

                for (; t < n_tok; t++) {
                    const float *xt = x + (size_t)t * (size_t)w->in;
                    float *yt = y + (size_t)t * (size_t)w->out;

                    int64_t r = 0;
                    for (; r + 4 <= tile_rows; r += 4) {
                        int64_t o_base = o0 + r;
                        dot_product_avx2_4x(xt, tile_w + (size_t)r * (size_t)w->in, w->in,
                                            bias_or_null ? bias_or_null + o_base : NULL,
                                            yt + o_base);
                    }
                    for (; r < tile_rows; r++) {
                        int64_t o = o0 + r;
                        const float *wr = tile_w + (size_t)r * (size_t)w->in;
                        float dot = dot_product_avx2(xt, wr, w->in);
                        if (bias_or_null) dot += bias_or_null[o];
                        yt[o] = dot;
                    }
                }
#else
                for (int t = 0; t < n_tok; t++) {
                    const float *xt = x + (size_t)t * (size_t)w->in;
                    float *yt = y + (size_t)t * (size_t)w->out;
                    for (int64_t r = 0; r < tile_rows; r++) {
                        int64_t o = o0 + r;
                        const float *wr = tile_w + (size_t)r * (size_t)w->in;
                        float dot = dot_product_scalar(xt, wr, w->in);
                        if (bias_or_null) dot += bias_or_null[o];
                        yt[o] = dot;
                    }
                }
#endif
            }
            free(tile_w);
        }
    }
}
