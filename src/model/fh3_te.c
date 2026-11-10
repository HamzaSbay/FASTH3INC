#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#include "fh3/fh3.h"
#include "model/fh3_te.h"
#include "io/fh3_pack.h"
#include "io/fh3_trunk.h"
#include "io/fh3_portable_io.h"
#include "core/fh3_quant.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define FH3_EMBED_MAGIC 0x45334846U  /* 'FH3E' */
#define FH3_EMBED_HEADER_SIZE 64U

static int s_last_te_hidden_size = 5120;

/* -------------------------------------------------------------------------
 * NIST SHA-256 implementation
 * ------------------------------------------------------------------------- */

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} Fh3Sha256;

static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    uint32_t W[64];
    for (int t = 0; t < 16; t++) {
        W[t] = ((uint32_t)data[t * 4] << 24) |
               ((uint32_t)data[t * 4 + 1] << 16) |
               ((uint32_t)data[t * 4 + 2] << 8) |
               ((uint32_t)data[t * 4 + 3]);
    }
    for (int t = 16; t < 64; t++) {
        uint32_t s0 = rotr32(W[t - 15], 7) ^ rotr32(W[t - 15], 18) ^ (W[t - 15] >> 3);
        uint32_t s1 = rotr32(W[t - 2], 17) ^ rotr32(W[t - 2], 19) ^ (W[t - 2] >> 10);
        W[t] = W[t - 16] + s0 + W[t - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int t = 0; t < 64; t++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K[t] + W[t];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void sha256_init(Fh3Sha256 *ctx) {
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

static void sha256_update(Fh3Sha256 *ctx, const uint8_t *data, size_t len) {
    size_t index = (size_t)(ctx->count & 63);
    ctx->count += len;
    size_t part_len = 64 - index;
    size_t i = 0;

    if (len >= part_len) {
        memcpy(&ctx->buffer[index], data, part_len);
        sha256_transform(ctx->state, ctx->buffer);
        for (i = part_len; i + 63 < len; i += 64) {
            sha256_transform(ctx->state, &data[i]);
        }
        index = 0;
    }
    memcpy(&ctx->buffer[index], &data[i], len - i);
}

static void sha256_final(Fh3Sha256 *ctx, uint8_t digest[32]) {
    uint8_t bits[8];
    uint64_t total_bits = ctx->count * 8;
    for (int i = 0; i < 8; i++) {
        bits[7 - i] = (uint8_t)(total_bits >> (i * 8));
    }
    size_t index = (size_t)(ctx->count & 63);
    size_t pad_len = (index < 56) ? (56 - index) : (120 - index);
    static const uint8_t padding[64] = { 0x80 };
    sha256_update(ctx, padding, pad_len);
    sha256_update(ctx, bits, 8);
    for (int i = 0; i < 8; i++) {
        digest[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

/* -------------------------------------------------------------------------
 * Internal Structure
 * ------------------------------------------------------------------------- */

struct Fh3Te {
    Fh3TeCfg cfg;
    Fh3File embed_file;
    uint32_t embed_type;
    uint32_t vocab_size;
    uint32_t embed_hidden;
    uint32_t embed_row_bytes;
    uint64_t embed_data_offset;
    bool has_embed_file;

    Fh3Pack trunk_pack;
    bool has_trunk_pack;
};

/* -------------------------------------------------------------------------
 * Text Encoder Open & Close
 * ------------------------------------------------------------------------- */

int fh3_te_open(const char *trunk_path, const char *embed_path, const Fh3TeCfg *cfg, Fh3Te **out) {
    if (!embed_path || !cfg || !out) return FH3_ERR;
    if (cfg->hidden_size <= 0 || cfg->num_heads <= 0 || cfg->num_kv_heads <= 0 ||
        cfg->head_dim <= 0 || cfg->num_layers <= 0 ||
        (cfg->num_heads % cfg->num_kv_heads != 0)) {
        return FH3_E_INVALID;
    }

    Fh3Te *m = (Fh3Te *)calloc(1, sizeof(*m));
    if (!m) return FH3_E_NOMEM;

    m->cfg = *cfg;
    s_last_te_hidden_size = cfg->hidden_size;

    /* 1. Open embed file */
    int rc = fh3_open_file(embed_path, &m->embed_file);
    if (rc != FH3_OK) {
        free(m);
        return rc;
    }
    m->has_embed_file = true;

    /* Read embedding header if present */
    uint8_t hdr_buf[FH3_EMBED_HEADER_SIZE];
    int64_t rd = fh3_pread(&m->embed_file, hdr_buf, FH3_EMBED_HEADER_SIZE, 0);
    uint32_t magic = 0;
    if (rd >= 4) {
        memcpy(&magic, hdr_buf, 4);
    }

    if (magic == FH3_EMBED_MAGIC && rd >= (int64_t)FH3_EMBED_HEADER_SIZE) {
        memcpy(&m->embed_type, hdr_buf + 8, 4);
        memcpy(&m->vocab_size, hdr_buf + 12, 4);
        memcpy(&m->embed_hidden, hdr_buf + 16, 4);
        memcpy(&m->embed_row_bytes, hdr_buf + 20, 4);
        memcpy(&m->embed_data_offset, hdr_buf + 24, 8);
    } else {
        /* Raw float32 table */
        m->embed_type = FH3_QUANT_F32;
        m->embed_hidden = (uint32_t)cfg->hidden_size;
        m->embed_row_bytes = (uint32_t)cfg->hidden_size * sizeof(float);
        m->embed_data_offset = 0;
        if (m->embed_row_bytes > 0) {
            m->vocab_size = (uint32_t)(m->embed_file.size / m->embed_row_bytes);
        }
    }

    /* 2. Open trunk pack if trunk_path is provided */
    if (trunk_path && strlen(trunk_path) > 0) {
        rc = fh3_pack_open(trunk_path, &m->trunk_pack);
        if (rc != FH3_OK) {
            fh3_te_close(m);
            return rc;
        }
        m->has_trunk_pack = true;
    }

    *out = m;
    return FH3_OK;
}

void fh3_te_close(Fh3Te *m) {
    if (!m) return;
    if (m->has_trunk_pack) {
        fh3_pack_close(&m->trunk_pack);
        m->has_trunk_pack = false;
    }
    if (m->has_embed_file) {
        fh3_close_file(&m->embed_file);
        m->has_embed_file = false;
    }
    free(m);
}

/* -------------------------------------------------------------------------
 * Causal GQA Attention
 * ------------------------------------------------------------------------- */

static void fh3_te_causal_gqa_attention(
    const float *q,
    const float *k,
    const float *v,
    float *o,
    int T,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    if (!q || !k || !v || !o || T <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    int group_size = num_heads / num_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    #pragma omp parallel
    {
        float *scores = (float *)malloc((size_t)T * sizeof(float));
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < T; i++) {
                if (!scores) continue;
                int kv_h = h / group_size;
                const float *qi = q + ((size_t)i * (size_t)num_heads + (size_t)h) * (size_t)head_dim;
                float *oi = o + ((size_t)i * (size_t)num_heads + (size_t)h) * (size_t)head_dim;

                float max_s = -1e30f;
                for (int j = 0; j <= i; j++) {
                    const float *kj = k + ((size_t)j * (size_t)num_kv_heads + (size_t)kv_h) * (size_t)head_dim;
                    float dot = 0.0f;
                    for (int c = 0; c < head_dim; c++) {
                        dot += qi[c] * kj[c];
                    }
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > max_s) max_s = s;
                }

                float sum_exp = 0.0f;
                for (int j = 0; j <= i; j++) {
                    float exp_v = expf(scores[j] - max_s);
                    scores[j] = exp_v;
                    sum_exp += exp_v;
                }

                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                for (int c = 0; c < head_dim; c++) {
                    oi[c] = 0.0f;
                }

                for (int j = 0; j <= i; j++) {
                    float w = scores[j] * inv_sum;
                    const float *vj = v + ((size_t)j * (size_t)num_kv_heads + (size_t)kv_h) * (size_t)head_dim;
                    for (int c = 0; c < head_dim; c++) {
                        oi[c] += w * vj[c];
                    }
                }
            }
        }
        if (scores) free(scores);
    }
}

/* -------------------------------------------------------------------------
 * Text Encoder Forward Pass
 * ------------------------------------------------------------------------- */

static inline Fh3W get_layer_tensor(const Fh3PackLayerDesc *ld, int idx, const uint8_t *base) {
    Fh3W w;
    memset(&w, 0, sizeof(w));
    if (idx < (int)ld->n_tensors) {
        w.type = (int)ld->tensors[idx].type;
        w.in = ld->tensors[idx].in_dim;
        w.out = ld->tensors[idx].out_dim;
        w.data = base + ld->tensors[idx].offset_in_layer;
    }
    return w;
}

int fh3_te_encode(const Fh3Te *m, Fh3Trunk *tr, const int *ids, int T, float *out) {
    if (!m || !ids || T <= 0 || !out) return FH3_ERR;
    if (!tr && !m->has_trunk_pack) return FH3_ERR;

    int hidden = m->cfg.hidden_size;
    int num_heads = m->cfg.num_heads;
    int num_kv_heads = m->cfg.num_kv_heads;
    int head_dim = m->cfg.head_dim;
    int interm = m->cfg.intermediate_size;
    int num_layers = m->cfg.num_layers;

    /* 1. Allocate activation and residual buffers */
    float *h           = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));
    float *residual    = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));
    float *normed      = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));
    float *q           = (float *)malloc((size_t)T * (size_t)num_heads * (size_t)head_dim * sizeof(float));
    float *k           = (float *)malloc((size_t)T * (size_t)num_kv_heads * (size_t)head_dim * sizeof(float));
    float *v           = (float *)malloc((size_t)T * (size_t)num_kv_heads * (size_t)head_dim * sizeof(float));
    float *attn_out    = (float *)malloc((size_t)T * (size_t)num_heads * (size_t)head_dim * sizeof(float));
    float *proj_out    = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));
    float *gate_buf    = (float *)malloc((size_t)T * (size_t)interm * sizeof(float));
    float *up_buf      = (float *)malloc((size_t)T * (size_t)interm * sizeof(float));
    float *swiglu_buf  = (float *)malloc((size_t)T * (size_t)interm * sizeof(float));
    float *down_buf    = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));

    float *cos_table   = (float *)malloc((size_t)T * (size_t)head_dim * sizeof(float));
    float *sin_table   = (float *)malloc((size_t)T * (size_t)head_dim * sizeof(float));
    uint8_t *row_raw   = (uint8_t *)malloc(m->embed_row_bytes > 0 ? (size_t)m->embed_row_bytes : (size_t)hidden * sizeof(float));

    if (!h || !residual || !normed || !q || !k || !v || !attn_out || !proj_out ||
        !gate_buf || !up_buf || !swiglu_buf || !down_buf || !cos_table || !sin_table || !row_raw) {
        free(h); free(residual); free(normed); free(q); free(k); free(v); free(attn_out);
        free(proj_out); free(gate_buf); free(up_buf); free(swiglu_buf); free(down_buf);
        free(cos_table); free(sin_table); free(row_raw);
        return FH3_E_NOMEM;
    }

    /* 2. Token embedding lookup */
    for (int i = 0; i < T; i++) {
        int id = ids[i];
        if (id < 0 || (m->vocab_size > 0 && (uint32_t)id >= m->vocab_size)) {
            free(h); free(residual); free(normed); free(q); free(k); free(v); free(attn_out);
            free(proj_out); free(gate_buf); free(up_buf); free(swiglu_buf); free(down_buf);
            free(cos_table); free(sin_table); free(row_raw);
            return FH3_E_INVALID;
        }

        int64_t offset = (int64_t)m->embed_data_offset + (int64_t)id * (int64_t)m->embed_row_bytes;
        int64_t rd = fh3_pread((Fh3File *)&m->embed_file, row_raw, m->embed_row_bytes, offset);
        if (rd != (int64_t)m->embed_row_bytes) {
            free(h); free(residual); free(normed); free(q); free(k); free(v); free(attn_out);
            free(proj_out); free(gate_buf); free(up_buf); free(swiglu_buf); free(down_buf);
            free(cos_table); free(sin_table); free(row_raw);
            return FH3_E_IO;
        }

        float *row_dst = h + (size_t)i * (size_t)hidden;
        fh3_dequant((int)m->embed_type, row_raw, row_dst, hidden);
    }

    /* 3. Build 1-D RoPE cos and sin tables for positions 0..T-1 */
    int half_dim = head_dim / 2;
    for (int t = 0; t < T; t++) {
        float *c_row = cos_table + (size_t)t * (size_t)head_dim;
        float *s_row = sin_table + (size_t)t * (size_t)head_dim;
        for (int j = 0; j < half_dim; j++) {
            float freq = 1.0f / powf(m->cfg.rope_theta, (float)(2 * j) / (float)head_dim);
            float val = (float)t * freq;
            float c = cosf(val);
            float s = sinf(val);
            c_row[j] = c;
            c_row[half_dim + j] = c;
            s_row[j] = s;
            s_row[half_dim + j] = s;
        }
    }

    /* 4. Stream layers */
    uint8_t *dynamic_layer_buf = NULL;

    for (int L = 0; L < num_layers; L++) {
        const Fh3Pack *pack = NULL;
        const uint8_t *l_base = NULL;

        if (tr) {
            fh3_trunk_prefetch(tr, L + 1);
            l_base = fh3_trunk_acquire(tr, L);
            pack = fh3_trunk_pack(tr);
        } else {
            pack = &m->trunk_pack;
            if (pack->mapped_base) {
                l_base = pack->mapped_base + pack->layers[L].file_offset;
            } else {
                size_t l_sz = (size_t)pack->layers[L].size_bytes;
                if (!dynamic_layer_buf) {
                    dynamic_layer_buf = (uint8_t *)fh3_alloc_aligned(l_sz);
                }
                int64_t rd = fh3_pread((Fh3File *)&pack->file, dynamic_layer_buf, l_sz, pack->layers[L].file_offset);
                if (rd != (int64_t)l_sz) {
                    if (dynamic_layer_buf) fh3_free_aligned(dynamic_layer_buf);
                    free(h); free(residual); free(normed); free(q); free(k); free(v); free(attn_out);
                    free(proj_out); free(gate_buf); free(up_buf); free(swiglu_buf); free(down_buf);
                    free(cos_table); free(sin_table); free(row_raw);
                    return FH3_E_IO;
                }
                l_base = dynamic_layer_buf;
            }
        }

        if (!l_base || !pack || L >= pack->num_layers) {
            if (dynamic_layer_buf) fh3_free_aligned(dynamic_layer_buf);
            free(h); free(residual); free(normed); free(q); free(k); free(v); free(attn_out);
            free(proj_out); free(gate_buf); free(up_buf); free(swiglu_buf); free(down_buf);
            free(cos_table); free(sin_table); free(row_raw);
            return FH3_ERR;
        }

        const Fh3PackLayerDesc *ld = &pack->layers[L];
        Fh3W w_input_norm = get_layer_tensor(ld, 0, l_base);
        Fh3W w_q          = get_layer_tensor(ld, 1, l_base);
        Fh3W w_k          = get_layer_tensor(ld, 2, l_base);
        Fh3W w_v          = get_layer_tensor(ld, 3, l_base);
        Fh3W w_q_norm     = get_layer_tensor(ld, 4, l_base);
        Fh3W w_k_norm     = get_layer_tensor(ld, 5, l_base);
        Fh3W w_o          = get_layer_tensor(ld, 6, l_base);
        Fh3W w_post_norm  = get_layer_tensor(ld, 7, l_base);
        Fh3W w_gate       = get_layer_tensor(ld, 8, l_base);
        Fh3W w_up         = get_layer_tensor(ld, 9, l_base);
        Fh3W w_down       = get_layer_tensor(ld, 10, l_base);

        /* --- Self-Attention branch --- */
        memcpy(residual, h, (size_t)T * (size_t)hidden * sizeof(float));
        memcpy(normed, h, (size_t)T * (size_t)hidden * sizeof(float));
        fh3_rmsnorm(normed, (const float *)w_input_norm.data, T, hidden, m->cfg.norm_eps);

        fh3_matmul(&w_q, normed, q, T, NULL);
        fh3_matmul(&w_k, normed, k, T, NULL);
        fh3_matmul(&w_v, normed, v, T, NULL);

        fh3_rmsnorm(q, (const float *)w_q_norm.data, T * num_heads, head_dim, m->cfg.qk_norm_eps);
        fh3_rmsnorm(k, (const float *)w_k_norm.data, T * num_kv_heads, head_dim, m->cfg.qk_norm_eps);

        fh3_rope3(q, cos_table, sin_table, T, num_heads, head_dim, head_dim);
        fh3_rope3(k, cos_table, sin_table, T, num_kv_heads, head_dim, head_dim);

        fh3_te_causal_gqa_attention(q, k, v, attn_out, T, num_heads, num_kv_heads, head_dim);

        fh3_matmul(&w_o, attn_out, proj_out, T, NULL);

        for (int j = 0; j < T * hidden; j++) {
            h[j] = residual[j] + proj_out[j];
        }

        /* --- Feed-Forward (MLP) branch --- */
        memcpy(residual, h, (size_t)T * (size_t)hidden * sizeof(float));
        memcpy(normed, h, (size_t)T * (size_t)hidden * sizeof(float));
        fh3_rmsnorm(normed, (const float *)w_post_norm.data, T, hidden, m->cfg.norm_eps);

        fh3_matmul(&w_gate, normed, gate_buf, T, NULL);
        fh3_matmul(&w_up, normed, up_buf, T, NULL);

        for (int j = 0; j < T * interm; j++) {
            float g = gate_buf[j];
            float s = g / (1.0f + expf(-g));
            swiglu_buf[j] = up_buf[j] * s;
        }

        fh3_matmul(&w_down, swiglu_buf, down_buf, T, NULL);

        for (int j = 0; j < T * hidden; j++) {
            h[j] = residual[j] + down_buf[j];
        }
    }

    /* 5. Pre-final-norm output written directly to out */
    memcpy(out, h, (size_t)T * (size_t)hidden * sizeof(float));

    if (dynamic_layer_buf) {
        fh3_free_aligned(dynamic_layer_buf);
    }
    free(h); free(residual); free(normed); free(q); free(k); free(v); free(attn_out);
    free(proj_out); free(gate_buf); free(up_buf); free(swiglu_buf); free(down_buf);
    free(cos_table); free(sin_table); free(row_raw);

    return FH3_OK;
}

/* -------------------------------------------------------------------------
 * Prompt Cache Operations
 * ------------------------------------------------------------------------- */

static void compute_cache_path(char *out_path, size_t cap, const char *cache_dir, const int *ids, int T) {
    Fh3Sha256 sha;
    sha256_init(&sha);
    sha256_update(&sha, (const uint8_t *)ids, (size_t)T * sizeof(int));
    uint8_t digest[32];
    sha256_final(&sha, digest);

    char hex[65];
    for (int i = 0; i < 32; i++) {
        snprintf(&hex[i * 2], 3, "%02x", digest[i]);
    }
    hex[64] = '\0';

    snprintf(out_path, cap, "%s/%s.emb", cache_dir, hex);
}

int fh3_te_cache_lookup(const char *cache_dir, const int *ids, int T, float *out) {
    if (!cache_dir || !ids || T <= 0 || !out) return FH3_ERR;

    char path[1024];
    compute_cache_path(path, sizeof(path), cache_dir, ids, T);

    FILE *fp = fopen(path, "rb");
    if (!fp) return FH3_E_NOT_FOUND;

    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (sz <= 0 || (size_t)sz % ((size_t)T * sizeof(float)) != 0) {
        fclose(fp);
        return FH3_ERR;
    }

    size_t rd = fread(out, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) return FH3_E_IO;

    return FH3_OK;
}

int fh3_te_cache_save(const char *cache_dir, const int *ids, int T, const float *emb) {
    if (!cache_dir || !ids || T <= 0 || !emb) return FH3_ERR;

    /* Ensure directory exists */
#if defined(_WIN32)
    CreateDirectoryA(cache_dir, NULL);
#else
    mkdir(cache_dir, 0755);
#endif

    char path[1024];
    compute_cache_path(path, sizeof(path), cache_dir, ids, T);

    FILE *fp = fopen(path, "wb");
    if (!fp) return FH3_E_IO;

    int hidden = s_last_te_hidden_size;
    size_t bytes = (size_t)T * (size_t)hidden * sizeof(float);
    size_t wr = fwrite(emb, 1, bytes, fp);
    fclose(fp);

    if (wr != bytes) return FH3_E_IO;
    return FH3_OK;
}
