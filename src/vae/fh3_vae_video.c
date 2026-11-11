#include "fh3_vae_video.h"
#include "io/fh3_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FH3_VAE_VIDEO_MAGIC 0x56334846U  /* 'FH3V' in little-endian uint32 */
#define FH3_VAE_VIDEO_VERSION 1U
#define FH3_VAE_VIDEO_HEADER_SIZE 192U

typedef struct {
    const float *norm1_w;
    const float *to_q_w;
    const float *to_q_b;
    const float *to_k_w;
    const float *to_k_b;
    const float *to_v_w;
    const float *to_v_b;
    const float *to_out_w;
    const float *to_out_b;
    const float *scale1;
    const float *norm2_w;
    const float *ff1_w;
    const float *ff1_b;
    const float *ff2_w;
    const float *ff2_b;
    const float *scale2;
} Fh3VaeBlock;

struct Fh3VaeV {
    uint32_t magic;
    uint32_t version;
    int latent_channels;
    int out_channels;
    int dim;
    int heads;
    int dim_head;
    int num_layers;
    int num_register_tokens;
    int patch_size_t;
    int patch_size;
    int ffn_mult;
    int clip_length;
    int token_drop;
    float rope_dim_ratio;
    float rope_theta;
    float norm_eps;

    /* File mapping / buffer */
    Fh3File file;
    void *h_map;
    const uint8_t *mapped_base;
    size_t mapped_size;
    uint8_t *allocated_data;

    /* Pointers to weights inside file */
    const float *latents_mean;
    const float *latents_std;

    const float *pq_w;
    const float *pq_b;

    const float *proj_in_w;
    const float *proj_in_b;

    const float *register_tokens;

    Fh3VaeBlock *blocks;

    const float *norm_out_w;
    const float *norm_out_b;

    const float *proj_out_w;
    const float *proj_out_b;
};

static inline uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline float read_f32(const uint8_t *p) {
    uint32_t u = read_u32(p);
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

int fh3_vae_video_open(const char *path, Fh3VaeV **out) {
    if (!path || !out) return FH3_E_INVALID;
    *out = NULL;

    Fh3File f;
    int rc = fh3_open_file(path, &f);
    if (rc != FH3_OK) return rc;

    if (f.size < FH3_VAE_VIDEO_HEADER_SIZE) {
        fh3_close_file(&f);
        return FH3_E_TRUNCATED;
    }

    const uint8_t *base = NULL;
    void *h_map = NULL;
    uint8_t *allocated = NULL;

#if defined(_WIN32)
    h_map = CreateFileMappingA(f.handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (h_map) {
        base = (const uint8_t *)MapViewOfFile((HANDLE)h_map, FILE_MAP_READ, 0, 0, 0);
    }
#else
    void *map = mmap(NULL, (size_t)f.size, PROT_READ, MAP_SHARED, f.fd, 0);
    if (map != MAP_FAILED) {
        base = (const uint8_t *)map;
    }
#endif

    if (!base) {
        allocated = (uint8_t *)malloc((size_t)f.size);
        if (!allocated) {
            fh3_close_file(&f);
            return FH3_E_NOMEM;
        }
        int64_t n_read = fh3_pread(&f, allocated, f.size, 0);
        if (n_read != (int64_t)f.size) {
            free(allocated);
            fh3_close_file(&f);
            return FH3_E_IO;
        }
        base = allocated;
    }

    uint32_t magic = read_u32(base + 0);
    uint32_t version = read_u32(base + 4);
    if (magic != FH3_VAE_VIDEO_MAGIC) {
        if (allocated) free(allocated);
#if defined(_WIN32)
        if (base && !allocated) UnmapViewOfFile(base);
        if (h_map) CloseHandle((HANDLE)h_map);
#else
        if (base && !allocated) munmap((void *)base, (size_t)f.size);
#endif
        fh3_close_file(&f);
        return FH3_E_BAD_MAGIC;
    }
    if (version != FH3_VAE_VIDEO_VERSION) {
        if (allocated) free(allocated);
#if defined(_WIN32)
        if (base && !allocated) UnmapViewOfFile(base);
        if (h_map) CloseHandle((HANDLE)h_map);
#else
        if (base && !allocated) munmap((void *)base, (size_t)f.size);
#endif
        fh3_close_file(&f);
        return FH3_E_BAD_VERSION;
    }

    Fh3VaeV *m = (Fh3VaeV *)calloc(1, sizeof(Fh3VaeV));
    if (!m) {
        if (allocated) free(allocated);
#if defined(_WIN32)
        if (base && !allocated) UnmapViewOfFile(base);
        if (h_map) CloseHandle((HANDLE)h_map);
#else
        if (base && !allocated) munmap((void *)base, (size_t)f.size);
#endif
        fh3_close_file(&f);
        return FH3_E_NOMEM;
    }

    m->file = f;
    m->h_map = h_map;
    m->mapped_base = base;
    m->mapped_size = (size_t)f.size;
    m->allocated_data = allocated;
    m->magic = magic;
    m->version = version;

    m->latent_channels     = (int)read_u32(base + 8);
    m->out_channels        = (int)read_u32(base + 12);
    m->dim                 = (int)read_u32(base + 16);
    m->heads               = (int)read_u32(base + 20);
    m->dim_head            = (int)read_u32(base + 24);
    m->num_layers          = (int)read_u32(base + 28);
    m->num_register_tokens = (int)read_u32(base + 32);
    m->patch_size_t        = (int)read_u32(base + 36);
    m->patch_size          = (int)read_u32(base + 40);
    m->ffn_mult            = (int)read_u32(base + 44);
    m->clip_length         = (int)read_u32(base + 48);
    m->token_drop          = (int)read_u32(base + 52);
    m->rope_dim_ratio      = read_f32(base + 56);
    m->rope_theta          = read_f32(base + 60);
    m->norm_eps            = read_f32(base + 64);

    if (m->latent_channels != 24 || m->out_channels != 3 ||
        m->dim <= 0 || m->heads <= 0 || m->dim_head <= 0 ||
        m->dim != m->heads * m->dim_head ||
        m->num_layers <= 0 || m->num_register_tokens < 0 ||
        m->patch_size <= 0 || m->patch_size_t <= 0 ||
        m->ffn_mult <= 0 || m->clip_length <= 0 ||
        m->rope_dim_ratio <= 0.0f || m->rope_dim_ratio > 1.0f) {
        fh3_vae_video_close(m);
        return FH3_E_INVALID;
    }
    int rotary_dim = (int)(m->dim_head * m->rope_dim_ratio);
    if (rotary_dim <= 0 || rotary_dim % 6 != 0 || rotary_dim / 2 > 48) {
        fh3_vae_video_close(m);
        return FH3_E_INVALID;
    }

    m->blocks = (Fh3VaeBlock *)calloc((size_t)m->num_layers, sizeof(Fh3VaeBlock));
    if (!m->blocks) {
        fh3_vae_video_close(m);
        return FH3_E_NOMEM;
    }

    /* Assign tensor pointers sequentially */
    const float *p = (const float *)(base + FH3_VAE_VIDEO_HEADER_SIZE);
    m->latents_mean = p; p += m->latent_channels;
    m->latents_std  = p; p += m->latent_channels;

    m->pq_w = p; p += m->latent_channels * m->latent_channels;
    m->pq_b = p; p += m->latent_channels;

    m->proj_in_w = p; p += m->dim * m->latent_channels;
    m->proj_in_b = p; p += m->dim;

    m->register_tokens = p; p += m->num_register_tokens * m->dim;

    int ffn_dim = m->dim * m->ffn_mult;
    for (int l = 0; l < m->num_layers; l++) {
        m->blocks[l].norm1_w  = p; p += m->dim;
        m->blocks[l].to_q_w   = p; p += m->dim * m->dim;
        m->blocks[l].to_q_b   = p; p += m->dim;
        m->blocks[l].to_k_w   = p; p += m->dim * m->dim;
        m->blocks[l].to_k_b   = p; p += m->dim;
        m->blocks[l].to_v_w   = p; p += m->dim * m->dim;
        m->blocks[l].to_v_b   = p; p += m->dim;
        m->blocks[l].to_out_w = p; p += m->dim * m->dim;
        m->blocks[l].to_out_b = p; p += m->dim;
        m->blocks[l].scale1   = p; p += m->dim;
        m->blocks[l].norm2_w  = p; p += m->dim;
        m->blocks[l].ff1_w    = p; p += 2 * ffn_dim * m->dim;
        m->blocks[l].ff1_b    = p; p += 2 * ffn_dim;
        m->blocks[l].ff2_w    = p; p += m->dim * ffn_dim;
        m->blocks[l].ff2_b    = p; p += m->dim;
        m->blocks[l].scale2   = p; p += m->dim;
    }

    m->norm_out_w = p; p += m->dim;
    m->norm_out_b = p; p += m->dim;

    int out_features = m->out_channels * m->patch_size_t * m->patch_size * m->patch_size;
    m->proj_out_w = p; p += out_features * m->dim;
    m->proj_out_b = p; p += out_features;

    size_t bytes_used = (size_t)((const uint8_t *)p - base);
    if (bytes_used > m->mapped_size) {
        fh3_vae_video_close(m);
        return FH3_E_TRUNCATED;
    }

    *out = m;
    return FH3_OK;
}

void fh3_vae_video_close(Fh3VaeV *m) {
    if (!m) return;

    free(m->blocks);

    if (m->allocated_data) {
        free(m->allocated_data);
    } else {
#if defined(_WIN32)
        if (m->mapped_base) UnmapViewOfFile(m->mapped_base);
        if (m->h_map) CloseHandle((HANDLE)m->h_map);
#else
        if (m->mapped_base) munmap((void *)m->mapped_base, m->mapped_size);
#endif
    }

    fh3_close_file(&m->file);
    free(m);
}

int fh3_vae_video_patch_size(const Fh3VaeV *m) {
    if (!m) return 16;
    return m->patch_size;
}

int fh3_vae_video_patch_size_t(const Fh3VaeV *m) {
    if (!m) return 4;
    return m->patch_size_t;
}

/* Vector math utilities */

static inline float reduce_tree_8(const float acc[8]) {
    float s01 = acc[0] + acc[1];
    float s23 = acc[2] + acc[3];
    float s45 = acc[4] + acc[5];
    float s67 = acc[6] + acc[7];
    float s03 = s01 + s23;
    float s47 = s45 + s67;
    return s03 + s47;
}

static inline float dot_f32(const float *a, const float *b, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 vacc = _mm256_setzero_ps();
    int n8 = n - (n % 8);
    for (int k = 0; k < n8; k += 8) {
        __m256 va = _mm256_loadu_ps(a + k);
        __m256 vb = _mm256_loadu_ps(b + k);
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    float acc[8];
    _mm256_storeu_ps(acc, vacc);
    float dot = reduce_tree_8(acc);
    for (int k = n8; k < n; k++) {
        dot = fmaf(a[k], b[k], dot);
    }
    return dot;
#else
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int n8 = n - (n % 8);
    for (int k = 0; k < n8; k += 8) {
        acc[0] = fmaf(a[k + 0], b[k + 0], acc[0]);
        acc[1] = fmaf(a[k + 1], b[k + 1], acc[1]);
        acc[2] = fmaf(a[k + 2], b[k + 2], acc[2]);
        acc[3] = fmaf(a[k + 3], b[k + 3], acc[3]);
        acc[4] = fmaf(a[k + 4], b[k + 4], acc[4]);
        acc[5] = fmaf(a[k + 5], b[k + 5], acc[5]);
        acc[6] = fmaf(a[k + 6], b[k + 6], acc[6]);
        acc[7] = fmaf(a[k + 7], b[k + 7], acc[7]);
    }
    float dot = reduce_tree_8(acc);
    for (int k = n8; k < n; k++) {
        dot = fmaf(a[k], b[k], dot);
    }
    return dot;
#endif
}

static void matmul_f32(const float *x, const float *w, const float *b, float *y, int n, int in_dim, int out_dim) {
    #pragma omp parallel for schedule(static) if (n * out_dim > 64)
    for (int i = 0; i < n; i++) {
        const float *xi = x + (size_t)i * (size_t)in_dim;
        float *yi = y + (size_t)i * (size_t)out_dim;
        for (int j = 0; j < out_dim; j++) {
            const float *wj = w + (size_t)j * (size_t)in_dim;
            float sum = b ? b[j] : 0.0f;
            yi[j] = sum + dot_f32(xi, wj, in_dim);
        }
    }
}

static inline void qk_norm(float *v, int dim_head, float eps) {
    float sum_sq = 0.0f;
    for (int d = 0; d < dim_head; d++) {
        sum_sq = fmaf(v[d], v[d], sum_sq);
    }
    float rms = sqrtf(sum_sq / (float)dim_head + eps);
    float inv_rms = 1.0f / rms;
    for (int d = 0; d < dim_head; d++) {
        v[d] *= inv_rms;
    }
}

static void layernorm_f32(const float *x, const float *w, const float *b, float *y, int n_tok, int dim, float eps) {
    #pragma omp parallel for schedule(static) if (n_tok > 1)
    for (int i = 0; i < n_tok; i++) {
        const float *xi = x + (size_t)i * (size_t)dim;
        float *yi = y + (size_t)i * (size_t)dim;

        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += xi[d];
        }
        float mean = sum / (float)dim;

        float sum_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = xi[d] - mean;
            sum_sq = fmaf(diff, diff, sum_sq);
        }
        float var = sum_sq / (float)dim;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int d = 0; d < dim; d++) {
            float val = (xi[d] - mean) * inv_std;
            if (w) val *= w[d];
            if (b) val += b[d];
            yi[d] = val;
        }
    }
}

static int decode_vit_clip(
    const Fh3VaeV *m,
    const float *lat_clip, /* 24 x F_clip x h x w */
    int F_clip,
    int h,
    int w,
    float *out_rgb /* 3 x (F_clip * patch_t) x (h * patch) x (w * patch) */
) {
    int N_patches = F_clip * h * w;
    int N_reg = m->num_register_tokens;
    int N_cls = 1;
    int N = N_patches + N_reg + N_cls;
    int dim = m->dim;
    int heads = m->heads;
    int dim_head = m->dim_head;
    int ffn_dim = dim * m->ffn_mult;
    int rotary_dim = (int)(dim_head * m->rope_dim_ratio);
    int half_rot_dim = rotary_dim / 2;

    /* Allocate working activations */
    float *x = (float *)malloc((size_t)N * (size_t)dim * sizeof(float));
    float *normed_x = (float *)malloc((size_t)N * (size_t)dim * sizeof(float));
    float *q = (float *)malloc((size_t)N * (size_t)dim * sizeof(float));
    float *k = (float *)malloc((size_t)N * (size_t)dim * sizeof(float));
    float *v = (float *)malloc((size_t)N * (size_t)dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)N * (size_t)dim * sizeof(float));
    float *ff_in = (float *)malloc((size_t)N * (size_t)(2 * ffn_dim) * sizeof(float));
    float *ff_mid = (float *)malloc((size_t)N * (size_t)ffn_dim * sizeof(float));
    float *cos_buf = (float *)malloc((size_t)N * (size_t)rotary_dim * sizeof(float));
    float *sin_buf = (float *)malloc((size_t)N * (size_t)rotary_dim * sizeof(float));

    int out_features = m->out_channels * m->patch_size_t * m->patch_size * m->patch_size;
    float *patch_buf = (float *)malloc((size_t)N_patches * (size_t)out_features * sizeof(float));

    if (!x || !normed_x || !q || !k || !v || !attn_out || !ff_in || !ff_mid || !cos_buf || !sin_buf || !patch_buf) {
        free(x); free(normed_x); free(q); free(k); free(v); free(attn_out);
        free(ff_in); free(ff_mid); free(cos_buf); free(sin_buf); free(patch_buf);
        return FH3_E_NOMEM;
    }

    /* 1. Precompute 3D RoPE angles */
    int n_freq = rotary_dim / 6;
    float *inv_freq = (float *)malloc((size_t)n_freq * sizeof(float));
    if (!inv_freq) {
        free(x); free(normed_x); free(q); free(k); free(v); free(attn_out);
        free(ff_in); free(ff_mid); free(cos_buf); free(sin_buf); free(patch_buf);
        return FH3_E_NOMEM;
    }
    for (int i = 0; i < n_freq; i++) {
        inv_freq[i] = 1.0f / powf(m->rope_theta, (float)i / (float)n_freq);
    }

    #pragma omp parallel for schedule(static)
    for (int tok = 0; tok < N_patches; tok++) {
        int f = tok / (h * w);
        int rem = tok % (h * w);
        int y = rem / w;
        int x_c = rem % w;

        float pos_t = 2.0f * ((float)f + 0.5f) / (float)F_clip - 1.0f;
        float pos_y = 2.0f * ((float)y + 0.5f) / (float)h - 1.0f;
        float pos_x = 2.0f * ((float)x_c + 0.5f) / (float)w - 1.0f;

        float a_half[48];
        for (int i = 0; i < n_freq; i++) {
            a_half[0 * n_freq + i] = (float)(2.0 * M_PI) * pos_t * inv_freq[i];
            a_half[1 * n_freq + i] = (float)(2.0 * M_PI) * pos_y * inv_freq[i];
            a_half[2 * n_freq + i] = (float)(2.0 * M_PI) * pos_x * inv_freq[i];
        }

        float *c_row = cos_buf + (size_t)tok * (size_t)rotary_dim;
        float *s_row = sin_buf + (size_t)tok * (size_t)rotary_dim;
        for (int i = 0; i < half_rot_dim; i++) {
            float c = cosf(a_half[i]);
            float s = sinf(a_half[i]);
            c_row[i]                = c;
            c_row[half_rot_dim + i] = c;
            s_row[i]                = s;
            s_row[half_rot_dim + i] = s;
        }
    }
    free(inv_freq);

    /* Suffix tokens (4 register + 1 cls) have pos = (0,0,0) -> cos=1, sin=0 */
    for (int tok = N_patches; tok < N; tok++) {
        float *c_row = cos_buf + (size_t)tok * (size_t)rotary_dim;
        float *s_row = sin_buf + (size_t)tok * (size_t)rotary_dim;
        for (int i = 0; i < rotary_dim; i++) {
            c_row[i] = 1.0f;
            s_row[i] = 0.0f;
        }
    }

    /* 2. Token initialization via proj_in */
    #pragma omp parallel for schedule(static)
    for (int tok = 0; tok < N_patches; tok++) {
        int f = tok / (h * w);
        int rem = tok % (h * w);
        int y = rem / w;
        int x_c = rem % w;

        float lat_voxel[24];
        for (int c = 0; c < 24; c++) {
            size_t idx = ((size_t)c * (size_t)F_clip + (size_t)f) * (size_t)(h * w) + (size_t)(y * w + x_c);
            lat_voxel[c] = lat_clip[idx];
        }

        float *x_tok = x + (size_t)tok * (size_t)dim;
        for (int d = 0; d < dim; d++) {
            float sum = m->proj_in_b[d];
            const float *w_row = m->proj_in_w + (size_t)d * 24;
            for (int c = 0; c < 24; c++) {
                sum = fmaf(w_row[c], lat_voxel[c], sum);
            }
            x_tok[d] = sum;
        }
    }

    /* Copy register tokens */
    for (int r = 0; r < N_reg; r++) {
        memcpy(x + (size_t)(N_patches + r) * (size_t)dim,
               m->register_tokens + (size_t)r * (size_t)dim,
               (size_t)dim * sizeof(float));
    }
    /* Zero cls token */
    memset(x + (size_t)(N_patches + N_reg) * (size_t)dim, 0, (size_t)dim * sizeof(float));

    /* 3. ViT Block Stack */
    for (int l = 0; l < m->num_layers; l++) {
        const Fh3VaeBlock *b = &m->blocks[l];

        /* norm1 */
        memcpy(normed_x, x, (size_t)N * (size_t)dim * sizeof(float));
        fh3_rmsnorm(normed_x, b->norm1_w, N, dim, m->norm_eps);

        /* Projections to_q, to_k, to_v */
        matmul_f32(normed_x, b->to_q_w, b->to_q_b, q, N, dim, dim);
        matmul_f32(normed_x, b->to_k_w, b->to_k_b, k, N, dim, dim);
        matmul_f32(normed_x, b->to_v_w, b->to_v_b, v, N, dim, dim);

        /* QK-Norm (RMSNorm per head, affine=False) */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int tok = 0; tok < N; tok++) {
            for (int hd = 0; hd < heads; hd++) {
                size_t head_off = ((size_t)tok * (size_t)heads + (size_t)hd) * (size_t)dim_head;
                qk_norm(q + head_off, dim_head, m->norm_eps);
                qk_norm(k + head_off, dim_head, m->norm_eps);
            }
        }

        /* 3D RoPE on q and k */
        fh3_rope3(q, cos_buf, sin_buf, N, heads, dim_head, rotary_dim);
        fh3_rope3(k, cos_buf, sin_buf, N, heads, dim_head, rotary_dim);

        /* Multi-head self attention */
        fh3_attention(q, k, v, attn_out, N, heads, dim_head);

        /* to_out projection */
        matmul_f32(attn_out, b->to_out_w, b->to_out_b, normed_x, N, dim, dim);

        /* scale1 residual */
        #pragma omp parallel for schedule(static)
        for (int tok = 0; tok < N; tok++) {
            float *xt = x + (size_t)tok * (size_t)dim;
            const float *ot = normed_x + (size_t)tok * (size_t)dim;
            for (int d = 0; d < dim; d++) {
                xt[d] = fmaf(ot[d], b->scale1[d], xt[d]);
            }
        }

        /* norm2 */
        memcpy(normed_x, x, (size_t)N * (size_t)dim * sizeof(float));
        fh3_rmsnorm(normed_x, b->norm2_w, N, dim, m->norm_eps);

        /* SwiGLU FFN */
        matmul_f32(normed_x, b->ff1_w, b->ff1_b, ff_in, N, dim, 2 * ffn_dim);
        fh3_swiglu(ff_in, ff_mid, N, ffn_dim);
        matmul_f32(ff_mid, b->ff2_w, b->ff2_b, normed_x, N, ffn_dim, dim);

        /* scale2 residual */
        #pragma omp parallel for schedule(static)
        for (int tok = 0; tok < N; tok++) {
            float *xt = x + (size_t)tok * (size_t)dim;
            const float *ft = normed_x + (size_t)tok * (size_t)dim;
            for (int d = 0; d < dim; d++) {
                xt[d] = fmaf(ft[d], b->scale2[d], xt[d]);
            }
        }
    }

    /* 4. Final LayerNorm (norm_out) */
    layernorm_f32(x, m->norm_out_w, m->norm_out_b, normed_x, N, dim, m->norm_eps);

    /* 5. proj_out linear on the first N_patches tokens */
    matmul_f32(normed_x, m->proj_out_w, m->proj_out_b, patch_buf, N_patches, dim, out_features);

    /* 6. Unpatchify + ImageNet un-normalization + clamp into out_rgb */
    const float pixel_mean[3] = {0.485f, 0.456f, 0.406f};
    const float pixel_std[3]  = {0.229f, 0.224f, 0.225f};

    int patch_t = m->patch_size_t;
    int patch_s = m->patch_size;
    int total_clip_frames = F_clip * patch_t;
    int total_height = h * patch_s;
    int total_width  = w * patch_s;
    size_t hw_pixel = (size_t)total_height * (size_t)total_width;

    #pragma omp parallel for schedule(static)
    for (int tok = 0; tok < N_patches; tok++) {
        int f = tok / (h * w);
        int rem = tok % (h * w);
        int y_lat = rem / w;
        int x_lat = rem % w;

        const float *p_tok = patch_buf + (size_t)tok * (size_t)out_features;

        for (int c = 0; c < 3; c++) {
            float mean_c = pixel_mean[c];
            float std_c  = pixel_std[c];
            size_t c_off = (size_t)c * ((size_t)total_clip_frames * hw_pixel);

            for (int dt = 0; dt < patch_t; dt++) {
                int pf = f * patch_t + dt;
                size_t f_off = c_off + (size_t)pf * hw_pixel;

                for (int dy = 0; dy < patch_s; dy++) {
                    int py = y_lat * patch_s + dy;
                    size_t y_off = f_off + (size_t)py * (size_t)total_width;

                    for (int dx = 0; dx < patch_s; dx++) {
                        int px = x_lat * patch_s + dx;
                        size_t feat_idx = ((size_t)c * (size_t)patch_t + (size_t)dt) * (size_t)(patch_s * patch_s) +
                                          (size_t)dy * (size_t)patch_s + (size_t)dx;

                        float raw = p_tok[feat_idx];
                        float val = raw * std_c + mean_c;
                        if (val < 0.0f) val = 0.0f;
                        else if (val > 1.0f) val = 1.0f;

                        out_rgb[y_off + (size_t)px] = val;
                    }
                }
            }
        }
    }

    free(x); free(normed_x); free(q); free(k); free(v); free(attn_out);
    free(ff_in); free(ff_mid); free(cos_buf); free(sin_buf); free(patch_buf);
    return FH3_OK;
}

int fh3_vae_video_decode(const Fh3VaeV *m, const float *lat, int F, int h, int w, Fh3FrameSink *sink) {
    if (!m || !lat || F <= 0 || h <= 0 || w <= 0 || (sink && !sink->write_frame)) return FH3_E_INVALID;

    int temporal_ratio = m->patch_size_t;
    int clip_length = m->clip_length;
    int token_drop = m->token_drop;

    int frame_pre_padding = ((-clip_length) % temporal_ratio + temporal_ratio) % temporal_ratio;
    int tokens_chunk_size = (int)ceil((double)clip_length / (double)temporal_ratio);
    int token_overlap = ((-token_drop) % tokens_chunk_size + tokens_chunk_size) % tokens_chunk_size;
    int frame_overlap = token_overlap * temporal_ratio - frame_pre_padding;
    if (frame_overlap < 0) frame_overlap = 0;
    int chunk_num_frames = tokens_chunk_size * temporal_ratio;

    int num_tokens = F + token_drop;
    int pad_tokens = ((-num_tokens) % tokens_chunk_size + tokens_chunk_size) % tokens_chunk_size;
    int F_padded = F + pad_tokens;
    int num_chunks = (num_tokens + pad_tokens) / tokens_chunk_size - (token_drop > 0 ? 1 : 0);
    if (num_chunks <= 0) return FH3_E_INVALID;

    /* Compute total target pixel frames after padding truncation */
    int pad_frames = 0;
    if (pad_tokens > 0) {
        int intra_tail = clip_length % temporal_ratio;
        int num_tokens_before_pad = F_padded - pad_tokens;
        for (int k = 0; k < pad_tokens; k++) {
            if (intra_tail && ((num_tokens_before_pad + k) % tokens_chunk_size == 0)) {
                pad_frames += intra_tail;
            } else {
                pad_frames += temporal_ratio;
            }
        }
    }
    int total_frames_to_emit = (num_chunks * (chunk_num_frames - frame_pre_padding) + frame_overlap) - pad_frames;

    /* 1. Latent denormalization & post_quant_conv */
    size_t lat_hw = (size_t)h * (size_t)w;
    size_t total_voxels = (size_t)F_padded * lat_hw;
    float *lat_pq = (float *)malloc((size_t)m->latent_channels * total_voxels * sizeof(float));
    if (!lat_pq) return FH3_E_NOMEM;

    #pragma omp parallel for schedule(static)
    for (size_t v_idx = 0; v_idx < total_voxels; v_idx++) {
        int f = (int)(v_idx / lat_hw);
        int rem = (int)(v_idx % lat_hw);
        int f_src = (f < F) ? f : (F - 1);

        float denorm[24];
        for (int c = 0; c < 24; c++) {
            size_t in_idx = ((size_t)c * (size_t)F + (size_t)f_src) * lat_hw + (size_t)rem;
            denorm[c] = lat[in_idx] * m->latents_std[c] + m->latents_mean[c];
        }

        for (int out_c = 0; out_c < 24; out_c++) {
            float sum = m->pq_b[out_c];
            const float *w_row = m->pq_w + (size_t)out_c * 24;
            for (int c = 0; c < 24; c++) {
                sum = fmaf(w_row[c], denorm[c], sum);
            }
            lat_pq[(size_t)out_c * total_voxels + v_idx] = sum;
        }
    }

    /* 2. Process temporal clips */
    int clip_tok_count = tokens_chunk_size + token_overlap;
    int clip_pixel_frames = clip_tok_count * temporal_ratio;
    int pixel_H = h * m->patch_size;
    int pixel_W = w * m->patch_size;
    size_t pixel_frame_sz = (size_t)3 * (size_t)pixel_H * (size_t)pixel_W;

    float *clip_pixel_buf = (float *)malloc((size_t)clip_pixel_frames * pixel_frame_sz * sizeof(float));
    float *overlap_buf = (float *)malloc((size_t)frame_overlap * pixel_frame_sz * sizeof(float));
    float *frame_temp = (float *)malloc(pixel_frame_sz * sizeof(float));
    bool has_overlap = false;
    int frames_emitted = 0;

    if (!clip_pixel_buf || !overlap_buf || !frame_temp) {
        free(lat_pq);
        free(clip_pixel_buf);
        free(overlap_buf);
        free(frame_temp);
        return FH3_E_NOMEM;
    }

    float *lat_clip_slice = (float *)malloc((size_t)m->latent_channels * (size_t)clip_tok_count * lat_hw * sizeof(float));
    if (!lat_clip_slice) {
        free(lat_pq); free(clip_pixel_buf); free(overlap_buf); free(frame_temp);
        return FH3_E_NOMEM;
    }

    for (int chunk_i = 0; chunk_i < num_chunks; chunk_i++) {
        int start_tok = chunk_i * tokens_chunk_size;

        /* Extract slice [24, clip_tok_count, h, w] */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int c = 0; c < m->latent_channels; c++) {
            for (int t = 0; t < clip_tok_count; t++) {
                int f = start_tok + t;
                const float *src = lat_pq + ((size_t)c * (size_t)F_padded + (size_t)f) * lat_hw;
                float *dst = lat_clip_slice + ((size_t)c * (size_t)clip_tok_count + (size_t)t) * lat_hw;
                memcpy(dst, src, lat_hw * sizeof(float));
            }
        }

        int decode_rc = decode_vit_clip(m, lat_clip_slice, clip_tok_count, h, w, clip_pixel_buf);
        if (decode_rc != FH3_OK) {
            free(lat_pq); free(clip_pixel_buf); free(overlap_buf); free(frame_temp); free(lat_clip_slice);
            return decode_rc;
        }

        int token_drop_active = (token_drop > 0 ? 1 : 0);
        for (int j = 0; j <= token_drop_active; j++) {
            int frame_start = j * chunk_num_frames;
            int avail = clip_pixel_frames - frame_start;
            int chunk_len = (avail < chunk_num_frames) ? avail : chunk_num_frames;
            int useful_len = chunk_len - frame_pre_padding;
            if (useful_len <= 0) continue;

            if (j == 0) {
                for (int t = 0; t < useful_len; t++) {
                    int src_pf = frame_pre_padding + t;

                    /* Copy frame [3, H, W] from clip_pixel_buf into frame_temp */
                    for (int c = 0; c < 3; c++) {
                        size_t c_src = ((size_t)c * (size_t)clip_pixel_frames + (size_t)src_pf) * (size_t)(pixel_H * pixel_W);
                        size_t c_dst = (size_t)c * (size_t)(pixel_H * pixel_W);
                        memcpy(frame_temp + c_dst, clip_pixel_buf + c_src, (size_t)(pixel_H * pixel_W) * sizeof(float));
                    }

                    if (has_overlap && t < frame_overlap) {
                        float wa = 1.0f - (float)t / (float)frame_overlap;
                        float wb = (float)t / (float)frame_overlap;
                        const float *ov_frame = overlap_buf + (size_t)t * pixel_frame_sz;

                        for (size_t p = 0; p < pixel_frame_sz; p++) {
                            frame_temp[p] = wa * ov_frame[p] + wb * frame_temp[p];
                        }
                    }

                    if (sink && frames_emitted < total_frames_to_emit) {
                        int sink_rc = sink->write_frame(sink->user, frames_emitted, frame_temp, pixel_W, pixel_H);
                        if (sink_rc != FH3_OK) {
                            free(lat_pq); free(clip_pixel_buf); free(overlap_buf); free(frame_temp); free(lat_clip_slice);
                            return sink_rc;
                        }
                    }
                    frames_emitted++;
                }
                has_overlap = false;
            } else {
                /* j == 1: save to overlap buffer */
                int copy_overlap = (useful_len < frame_overlap) ? useful_len : frame_overlap;
                for (int t = 0; t < copy_overlap; t++) {
                    int src_pf = frame_start + frame_pre_padding + t;
                    float *ov_dst = overlap_buf + (size_t)t * pixel_frame_sz;

                    for (int c = 0; c < 3; c++) {
                        size_t c_src = ((size_t)c * (size_t)clip_pixel_frames + (size_t)src_pf) * (size_t)(pixel_H * pixel_W);
                        size_t c_dst = (size_t)c * (size_t)(pixel_H * pixel_W);
                        memcpy(ov_dst + c_dst, clip_pixel_buf + c_src, (size_t)(pixel_H * pixel_W) * sizeof(float));
                    }
                }
                has_overlap = true;
            }
        }
    }

    if (has_overlap) {
        for (int t = 0; t < frame_overlap; t++) {
            if (sink && frames_emitted < total_frames_to_emit) {
                const float *ov_frame = overlap_buf + (size_t)t * pixel_frame_sz;
                int sink_rc = sink->write_frame(sink->user, frames_emitted, ov_frame, pixel_W, pixel_H);
                if (sink_rc != FH3_OK) {
                    free(lat_pq); free(clip_pixel_buf); free(overlap_buf); free(frame_temp); free(lat_clip_slice);
                    return sink_rc;
                }
            }
            frames_emitted++;
        }
    }

    free(lat_pq);
    free(clip_pixel_buf);
    free(overlap_buf);
    free(frame_temp);
    free(lat_clip_slice);

    return FH3_OK;
}
