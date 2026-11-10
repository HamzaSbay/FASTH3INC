#include "fh3_vae_audio.h"
#include "io/fh3_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define FH3_VAE_AUDIO_MAGIC 0x41334846U  /* 'FH3A' in little-endian uint32 */
#define FH3_VAE_AUDIO_VERSION 1U
#define FH3_VAE_AUDIO_HEADER_SIZE 192U

static const float KAISER_SINC_12[12] = {
    0.0020289646927267313f,
    0.009389465674757957f,
   -0.0255434587597847f,
   -0.057657383382320404f,
    0.12857258319854736f,
    0.44320979714393616f,
    0.44320979714393616f,
    0.12857258319854736f,
   -0.057657383382320404f,
   -0.0255434587597847f,
    0.009389465674757957f,
    0.0020289646927267313f
};

typedef struct {
    int dilation;
    int pad1;
    int pad2;
    const float *act1_alpha;
    const float *act1_beta;
    const float *conv1_w;
    const float *conv1_b;
    const float *act2_alpha;
    const float *act2_beta;
    const float *conv2_w;
    const float *conv2_b;
} Fh3VaeBranch;

typedef struct {
    int kernel;
    Fh3VaeBranch branches[8];
} Fh3VaeBlock;

typedef struct {
    int cin;
    int cout;
    int rate;
    int kernel;
    int pad;
    const float *w_up;
    const float *b_up;
    Fh3VaeBlock blocks[8];
} Fh3VaeStage;

struct Fh3VaeA {
    uint32_t magic;
    uint32_t version;
    int latent_channels;
    int latent_dim;
    int decoder_dim;
    int num_upsamples;
    int num_resblocks;
    int num_dilations;
    int rates[8];
    int kernel_sizes[8];
    int resblock_kernels[8];
    int resblock_dilations[8];
    int total_upsample;

    /* Raw allocated weight buffer */
    uint8_t *file_data;
    size_t file_size;

    /* Pointers into file_data */
    const float *latents_mean;
    const float *latents_std;

    const float *w_in;
    const float *b_in;

    const float *w_pre;
    const float *b_pre;

    Fh3VaeStage stages[8];

    int c_final;
    const float *act_post_alpha;
    const float *act_post_beta;
    const float *conv_post_w;
};

int fh3_vae_audio_upsample_ratio(const Fh3VaeA *m) {
    if (!m) return 0;
    return m->total_upsample;
}

static inline uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int fh3_vae_audio_open(const char *path, Fh3VaeA **out) {
    if (!path || !out) return FH3_E_INVALID;
    *out = NULL;

    Fh3File f;
    int rc = fh3_open_file(path, &f);
    if (rc != FH3_OK) return rc;

    if (f.size < FH3_VAE_AUDIO_HEADER_SIZE) {
        fh3_close_file(&f);
        return FH3_E_TRUNCATED;
    }

    uint8_t *data = (uint8_t *)malloc((size_t)f.size);
    if (!data) {
        fh3_close_file(&f);
        return FH3_E_NOMEM;
    }

    int64_t n_read = fh3_pread(&f, data, f.size, 0);
    fh3_close_file(&f);

    if (n_read != (int64_t)f.size) {
        free(data);
        return FH3_E_IO;
    }

    /* Parse 192-byte header */
    uint32_t magic = read_u32_le(data + 0);
    uint32_t version = read_u32_le(data + 4);
    if (magic != FH3_VAE_AUDIO_MAGIC) {
        free(data);
        return FH3_E_BAD_MAGIC;
    }
    if (version != FH3_VAE_AUDIO_VERSION) {
        free(data);
        return FH3_E_BAD_VERSION;
    }

    Fh3VaeA *m = (Fh3VaeA *)calloc(1, sizeof(Fh3VaeA));
    if (!m) {
        free(data);
        return FH3_E_NOMEM;
    }

    m->magic = magic;
    m->version = version;
    m->file_data = data;
    m->file_size = (size_t)f.size;

    m->latent_channels = (int)read_u32_le(data + 8);
    m->latent_dim = (int)read_u32_le(data + 12);
    m->decoder_dim = (int)read_u32_le(data + 16);
    m->num_upsamples = (int)read_u32_le(data + 20);
    m->num_resblocks = (int)read_u32_le(data + 24);
    m->num_dilations = (int)read_u32_le(data + 28);

    if (m->num_upsamples > 8 || m->num_resblocks > 8 || m->num_dilations > 8 ||
        m->latent_channels <= 0 || m->latent_dim <= 0 || m->decoder_dim <= 0) {
        free(data);
        free(m);
        return FH3_E_INVALID;
    }

    for (int i = 0; i < 8; i++) m->rates[i] = (int)read_u32_le(data + 32 + i * 4);
    for (int i = 0; i < 8; i++) m->kernel_sizes[i] = (int)read_u32_le(data + 64 + i * 4);
    for (int i = 0; i < 8; i++) m->resblock_kernels[i] = (int)read_u32_le(data + 96 + i * 4);
    for (int i = 0; i < 8; i++) m->resblock_dilations[i] = (int)read_u32_le(data + 128 + i * 4);
    m->total_upsample = (int)read_u32_le(data + 160);

    /* Assign pointers into mapped payload */
    const float *p = (const float *)(data + FH3_VAE_AUDIO_HEADER_SIZE);
    const float *p_end = (const float *)(data + f.size);

    #define CHECK_PTR(count) do { \
        if (p + (count) > p_end) { \
            free(data); \
            free(m); \
            return FH3_E_TRUNCATED; \
        } \
    } while (0)

    CHECK_PTR(m->latent_channels * 2);
    m->latents_mean = p; p += m->latent_channels;
    m->latents_std = p; p += m->latent_channels;

    /* 1. dec_in_proj */
    CHECK_PTR((size_t)m->latent_dim * m->latent_channels + m->latent_dim);
    m->w_in = p; p += (size_t)m->latent_dim * m->latent_channels;
    m->b_in = p; p += m->latent_dim;

    /* 2. conv_pre */
    CHECK_PTR((size_t)m->decoder_dim * m->latent_dim * 7 + m->decoder_dim);
    m->w_pre = p; p += (size_t)m->decoder_dim * m->latent_dim * 7;
    m->b_pre = p; p += m->decoder_dim;

    /* 3. Upsample stages */
    for (int i = 0; i < m->num_upsamples; i++) {
        Fh3VaeStage *st = &m->stages[i];
        st->cin = m->decoder_dim / (1 << i);
        st->cout = m->decoder_dim / (1 << (i + 1));
        st->rate = m->rates[i];
        st->kernel = m->kernel_sizes[i];
        st->pad = (st->kernel - st->rate) / 2;

        CHECK_PTR((size_t)st->cin * st->cout * st->kernel + st->cout);
        st->w_up = p; p += (size_t)st->cin * st->cout * st->kernel;
        st->b_up = p; p += st->cout;

        for (int j = 0; j < m->num_resblocks; j++) {
            Fh3VaeBlock *blk = &st->blocks[j];
            blk->kernel = m->resblock_kernels[j];

            for (int d = 0; d < m->num_dilations; d++) {
                Fh3VaeBranch *br = &blk->branches[d];
                br->dilation = m->resblock_dilations[d];
                br->pad1 = (blk->kernel * br->dilation - br->dilation) / 2;
                br->pad2 = (blk->kernel - 1) / 2;

                /* act1: alpha, beta */
                CHECK_PTR(st->cout * 2);
                br->act1_alpha = p; p += st->cout;
                br->act1_beta = p; p += st->cout;

                /* conv1: weight, bias */
                CHECK_PTR((size_t)st->cout * st->cout * blk->kernel + st->cout);
                br->conv1_w = p; p += (size_t)st->cout * st->cout * blk->kernel;
                br->conv1_b = p; p += st->cout;

                /* act2: alpha, beta */
                CHECK_PTR(st->cout * 2);
                br->act2_alpha = p; p += st->cout;
                br->act2_beta = p; p += st->cout;

                /* conv2: weight, bias */
                CHECK_PTR((size_t)st->cout * st->cout * blk->kernel + st->cout);
                br->conv2_w = p; p += (size_t)st->cout * st->cout * blk->kernel;
                br->conv2_b = p; p += st->cout;
            }
        }
    }

    /* 4. Post-processing */
    m->c_final = m->decoder_dim / (1 << m->num_upsamples);
    CHECK_PTR(m->c_final * 2 + (size_t)m->c_final * 7);
    m->act_post_alpha = p; p += m->c_final;
    m->act_post_beta = p; p += m->c_final;
    m->conv_post_w = p; p += (size_t)m->c_final * 7;

    #undef CHECK_PTR

    *out = m;
    return FH3_OK;
}

void fh3_vae_audio_close(Fh3VaeA *m) {
    if (!m) return;
    if (m->file_data) free(m->file_data);
    free(m);
}

/* -------------------------------------------------------------------------
 * Low-level neural network operations
 * ------------------------------------------------------------------------- */

static void vae_alias_free_activation(float *x, int C, int T,
                                      const float *alpha, const float *beta,
                                      float *u_pool) {
    #pragma omp parallel for
    for (int c = 0; c < C; c++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *u = u_pool + (size_t)tid * 2 * T;
        const float *x_c = x + (size_t)c * T;

        /* 1. Upsample by 2 with replicate clamping */
        for (int to = 0; to < 2 * T; to++) {
            int m = to + 15;
            float acc = 0.0f;
            for (int k = 0; k < 12; k++) {
                if ((m - k) % 2 == 0) {
                    int n = (m - k) / 2 - 5;
                    int nc = n < 0 ? 0 : (n >= T ? T - 1 : n);
                    acc += x_c[nc] * KAISER_SINC_12[k];
                }
            }
            u[to] = 2.0f * acc;
        }

        /* 2. SnakeBeta: x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2 */
        float a = expf(alpha[c]);
        float b = 1.0f / (expf(beta[c]) + 1e-9f);
        for (int to = 0; to < 2 * T; to++) {
            float v = u[to];
            float s = sinf(a * v);
            u[to] = v + b * (s * s);
        }

        /* 3. Downsample by 2 with replicate clamping (stride 2, pad 5 on left) */
        float *out_c = x + (size_t)c * T;
        for (int to = 0; to < T; to++) {
            float acc = 0.0f;
            for (int k = 0; k < 12; k++) {
                int ti = 2 * to + k - 5;
                int tic = ti < 0 ? 0 : (ti >= 2 * T ? 2 * T - 1 : ti);
                acc += u[tic] * KAISER_SINC_12[k];
            }
            out_c[to] = acc;
        }
    }
}

static void vae_conv1d(const float *x, float *y, int C_in, int C_out, int T,
                       const float *w, const float *bias, int K, int dilation, int padding) {
    #pragma omp parallel for
    for (int co = 0; co < C_out; co++) {
        float b = bias ? bias[co] : 0.0f;
        const float *w_co = w + (size_t)co * C_in * K;
        float *y_co = y + (size_t)co * T;

        for (int to = 0; to < T; to++) {
            float acc = b;
            for (int ci = 0; ci < C_in; ci++) {
                const float *w_ci = w_co + (size_t)ci * K;
                const float *x_ci = x + (size_t)ci * T;
                for (int k = 0; k < K; k++) {
                    int ti = to - padding + k * dilation;
                    if ((unsigned)ti < (unsigned)T) {
                        acc += w_ci[k] * x_ci[ti];
                    }
                }
            }
            y_co[to] = acc;
        }
    }
}

static void vae_conv_transpose1d(const float *x, float *y, int C_in, int C_out, int T_in,
                                const float *w, const float *bias, int K, int stride, int padding) {
    int T_out = (T_in - 1) * stride - 2 * padding + K;

    #pragma omp parallel for
    for (int co = 0; co < C_out; co++) {
        float b = bias ? bias[co] : 0.0f;
        float *y_co = y + (size_t)co * T_out;

        for (int to = 0; to < T_out; to++) {
            float acc = b;
            for (int k = 0; k < K; k++) {
                int diff = to + padding - k;
                if (diff >= 0 && diff % stride == 0) {
                    int ti = diff / stride;
                    if (ti < T_in) {
                        for (int ci = 0; ci < C_in; ci++) {
                            float w_val = w[((size_t)ci * C_out + co) * K + k];
                            acc += x[(size_t)ci * T_in + ti] * w_val;
                        }
                    }
                }
            }
            y_co[to] = acc;
        }
    }
}

/* -------------------------------------------------------------------------
 * Full BigVGAN decode pass
 * ------------------------------------------------------------------------- */

int fh3_vae_audio_decode(const Fh3VaeA *m, const float *lat, int L, float *pcm) {
    if (!m || !lat || !pcm || L <= 0) return FH3_E_INVALID;
    if (m->magic != FH3_VAE_AUDIO_MAGIC) return FH3_E_BAD_MAGIC;

    int total_samples = L * m->total_upsample;

    /* Compute maximum buffer sizes required across all stages */
    size_t max_T = (size_t)total_samples;
    size_t max_buf_elems = (size_t)m->latent_dim * L;
    if ((size_t)m->decoder_dim * L > max_buf_elems) {
        max_buf_elems = (size_t)m->decoder_dim * L;
    }

    size_t cur_T = (size_t)L;
    for (int i = 0; i < m->num_upsamples; i++) {
        cur_T *= (size_t)m->stages[i].rate;
        size_t stage_elems = (size_t)m->stages[i].cout * cur_T;
        if (stage_elems > max_buf_elems) max_buf_elems = stage_elems;
    }

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif

    float *buf_a = (float *)malloc(max_buf_elems * sizeof(float));
    float *buf_b = (float *)malloc(max_buf_elems * sizeof(float));
    float *buf_res_acc = (float *)malloc(max_buf_elems * sizeof(float));
    float *buf_blk = (float *)malloc(max_buf_elems * sizeof(float));
    float *buf_conv = (float *)malloc(max_buf_elems * sizeof(float));
    float *u_pool = (float *)malloc((size_t)max_threads * 2 * max_T * sizeof(float));

    if (!buf_a || !buf_b || !buf_res_acc || !buf_blk || !buf_conv || !u_pool) {
        if (buf_a) free(buf_a);
        if (buf_b) free(buf_b);
        if (buf_res_acc) free(buf_res_acc);
        if (buf_blk) free(buf_blk);
        if (buf_conv) free(buf_conv);
        if (u_pool) free(u_pool);
        return FH3_E_NOMEM;
    }

    /* Process Left (ch=0) and Right (ch=1) channels */
    for (int ch = 0; ch < 2; ch++) {
        const float *lat_ch = lat + (size_t)ch * L * m->latent_channels;
        float *pcm_ch = pcm + (size_t)ch * total_samples;

        /* Step 1: Denormalize & transpose input [L, latent_channels] -> [latent_channels, L] into buf_a */
        for (int t = 0; t < L; t++) {
            for (int d = 0; d < m->latent_channels; d++) {
                float val = lat_ch[(size_t)t * m->latent_channels + d];
                buf_a[(size_t)d * L + t] = val * m->latents_std[d] + m->latents_mean[d];
            }
        }

        /* Step 2: dec_in_proj (Conv1d 1x1: latent_channels -> latent_dim) -> buf_b */
        #pragma omp parallel for
        for (int o = 0; o < m->latent_dim; o++) {
            const float *w_o = m->w_in + (size_t)o * m->latent_channels;
            float b = m->b_in[o];
            float *out_o = buf_b + (size_t)o * L;
            for (int t = 0; t < L; t++) {
                float acc = b;
                for (int i = 0; i < m->latent_channels; i++) {
                    acc += w_o[i] * buf_a[(size_t)i * L + t];
                }
                out_o[t] = acc;
            }
        }

        /* Step 3: conv_pre (Conv1d: latent_dim -> decoder_dim, K=7, P=3) -> buf_a */
        vae_conv1d(buf_b, buf_a, m->latent_dim, m->decoder_dim, L, m->w_pre, m->b_pre, 7, 1, 3);

        int cur_channels = m->decoder_dim;
        int cur_len = L;

        /* Step 4: 7 Upsample Stages */
        for (int i = 0; i < m->num_upsamples; i++) {
            const Fh3VaeStage *st = &m->stages[i];
            int out_len = cur_len * st->rate;

            /* 4a: Transposed conv -> buf_b */
            vae_conv_transpose1d(buf_a, buf_b, cur_channels, st->cout, cur_len,
                                st->w_up, st->b_up, st->kernel, st->rate, st->pad);

            size_t stage_total = (size_t)st->cout * out_len;
            memset(buf_res_acc, 0, stage_total * sizeof(float));

            /* 4b: Parallel AMPBlocks */
            for (int j = 0; j < m->num_resblocks; j++) {
                const Fh3VaeBlock *blk = &st->blocks[j];
                memcpy(buf_blk, buf_b, stage_total * sizeof(float));

                /* 3 branches per block */
                for (int d = 0; d < m->num_dilations; d++) {
                    const Fh3VaeBranch *br = &blk->branches[d];

                    /* Copy buf_blk -> buf_a so buf_blk remains unchanged for residual add */
                    memcpy(buf_a, buf_blk, stage_total * sizeof(float));

                    /* act1 -> buf_a (in-place) */
                    vae_alias_free_activation(buf_a, st->cout, out_len,
                                             br->act1_alpha, br->act1_beta, u_pool);

                    /* conv1 -> buf_conv */
                    vae_conv1d(buf_a, buf_conv, st->cout, st->cout, out_len,
                              br->conv1_w, br->conv1_b, blk->kernel, br->dilation, br->pad1);

                    /* act2 -> buf_conv (in-place) */
                    vae_alias_free_activation(buf_conv, st->cout, out_len,
                                             br->act2_alpha, br->act2_beta, u_pool);

                    /* conv2 -> buf_a */
                    vae_conv1d(buf_conv, buf_a, st->cout, st->cout, out_len,
                              br->conv2_w, br->conv2_b, blk->kernel, 1, br->pad2);

                    /* Branch residual: buf_blk += buf_a */
                    for (size_t idx = 0; idx < stage_total; idx++) {
                        buf_blk[idx] += buf_a[idx];
                    }
                }

                /* Block residual accumulation: buf_res_acc += buf_blk */
                for (size_t idx = 0; idx < stage_total; idx++) {
                    buf_res_acc[idx] += buf_blk[idx];
                }
            }

            /* 4c: Average over blocks: buf_a = buf_res_acc / num_resblocks */
            float inv_blocks = 1.0f / (float)m->num_resblocks;
            for (size_t idx = 0; idx < stage_total; idx++) {
                buf_a[idx] = buf_res_acc[idx] * inv_blocks;
            }

            cur_channels = st->cout;
            cur_len = out_len;
        }

        /* Step 5: Post-processing */
        /* activation_post -> buf_a (in-place) */
        vae_alias_free_activation(buf_a, m->c_final, cur_len,
                                 m->act_post_alpha, m->act_post_beta, u_pool);

        /* conv_post (Conv1d: c_final -> 1, K=7, P=3, no bias) -> buf_b */
        vae_conv1d(buf_a, buf_b, m->c_final, 1, cur_len, m->conv_post_w, NULL, 7, 1, 3);

        /* Step 6: Clamp to [-1.0f, 1.0f] and write to pcm_ch */
        for (int i = 0; i < total_samples; i++) {
            float s = buf_b[i];
            if (s > 1.0f) s = 1.0f;
            else if (s < -1.0f) s = -1.0f;
            pcm_ch[i] = s;
        }
    }

    free(buf_a);
    free(buf_b);
    free(buf_res_acc);
    free(buf_blk);
    free(buf_conv);
    free(u_pool);

    return FH3_OK;
}
