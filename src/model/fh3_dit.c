#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fh3/fh3.h"
#include "model/fh3_seq.h"
#include "model/fh3_dit.h"
#include "core/fh3_quant.h"
#include "core/fh3_vsa.h"

static const float *load_tensor_f32(Fh3Dit *m, const char *name) {
    const Fh3Tensor *t = fh3_gguf_find(&m->globals_gguf, name);
    if (!t) return NULL;

    const uint8_t *src = m->globals_data + t->off;
    if (t->type == FH3_QUANT_F32) {
        return (const float *)src;
    }

    /* Allocate and dequantize */
    int64_t count = t->ne[0];
    if (count <= 0) return NULL;

    float *dst = (float *)malloc((size_t)count * sizeof(float));
    if (!dst) return NULL;

    fh3_dequant(t->type, src, dst, count);

    /* Store pointer for cleanup */
    void **new_allocs = (void **)realloc(m->dequant_allocs, (size_t)(m->n_dequant_allocs + 1) * sizeof(void *));
    if (!new_allocs) {
        free(dst);
        return NULL;
    }
    m->dequant_allocs = new_allocs;
    m->dequant_allocs[m->n_dequant_allocs++] = dst;

    return dst;
}

static int load_tensor_w(Fh3Dit *m, const char *name, Fh3W *w) {
    memset(w, 0, sizeof(*w));
    const Fh3Tensor *t = fh3_gguf_find(&m->globals_gguf, name);
    if (!t) return FH3_ERR;

    w->type = t->type;
    w->in = t->ne[0];
    w->out = (t->ndim > 1) ? t->ne[1] : 1;
    w->data = m->globals_data + t->off;
    return FH3_OK;
}

int fh3_dit_open(const char *globals_path, const char *adaln_path, const Fh3DitCfg *cfg, Fh3Dit **out) {
    if (!globals_path || !adaln_path || !cfg || !out) return FH3_ERR;

    Fh3Dit *m = (Fh3Dit *)calloc(1, sizeof(Fh3Dit));
    if (!m) return FH3_E_NOMEM;

    m->cfg = *cfg;

    /* 1. Open AdaLN table */
    int rc = fh3_adaln_open(adaln_path, 0, &m->adaln);
    if (rc != FH3_OK) {
        fh3_dit_close(m);
        return rc;
    }

    /* 2. Open globals GGUF */
    rc = fh3_gguf_open(globals_path, &m->globals_gguf);
    if (rc != FH3_OK) {
        fh3_dit_close(m);
        return rc;
    }

    if (m->globals_gguf.file.size < m->globals_gguf.data_offset) {
        fh3_dit_close(m);
        return FH3_E_TRUNCATED;
    }

    m->globals_data_size = (size_t)(m->globals_gguf.file.size - m->globals_gguf.data_offset);
    m->globals_data = (uint8_t *)malloc(m->globals_data_size);
    if (!m->globals_data) {
        fh3_dit_close(m);
        return FH3_E_NOMEM;
    }

    int64_t rd = fh3_pread(&m->globals_gguf.file, m->globals_data, m->globals_data_size, m->globals_gguf.data_offset);
    if (rd != (int64_t)m->globals_data_size) {
        fh3_dit_close(m);
        return FH3_E_IO;
    }

    /* 3. Patch projections */
    if (load_tensor_w(m, "video_patch_proj.weight", &m->video_patch_proj) != FH3_OK) {
        fh3_dit_close(m); return FH3_ERR;
    }
    m->video_patch_proj_bias = load_tensor_f32(m, "video_patch_proj.bias");

    if (load_tensor_w(m, "audio_patch_proj.weight", &m->audio_patch_proj) != FH3_OK) {
        fh3_dit_close(m); return FH3_ERR;
    }
    m->audio_patch_proj_bias = load_tensor_f32(m, "audio_patch_proj.bias");

    if (load_tensor_w(m, "condition_proj.weight", &m->condition_proj) != FH3_OK) {
        fh3_dit_close(m); return FH3_ERR;
    }
    m->condition_proj_bias = load_tensor_f32(m, "condition_proj.bias");

    /* 4. Token refiner blocks */
    if (m->cfg.num_refiner_layers < 0 || m->cfg.num_refiner_layers > 4) {
        fh3_dit_close(m);
        return FH3_E_INVALID;
    }
    for (int r = 0; r < m->cfg.num_refiner_layers; r++) {
        char name[128];
        snprintf(name, sizeof(name), "token_refiner.blocks.%d.norm1.weight", r);
        m->refiner[r].norm1 = load_tensor_f32(m, name);

        snprintf(name, sizeof(name), "token_refiner.blocks.%d.attn.qkv_proj.weight", r);
        load_tensor_w(m, name, &m->refiner[r].qkv);

        snprintf(name, sizeof(name), "token_refiner.blocks.%d.attn.q_norm.weight", r);
        m->refiner[r].q_norm = load_tensor_f32(m, name);

        snprintf(name, sizeof(name), "token_refiner.blocks.%d.attn.k_norm.weight", r);
        m->refiner[r].k_norm = load_tensor_f32(m, name);

        snprintf(name, sizeof(name), "token_refiner.blocks.%d.attn.out_proj.weight", r);
        load_tensor_w(m, name, &m->refiner[r].out);

        snprintf(name, sizeof(name), "token_refiner.blocks.%d.norm2.weight", r);
        m->refiner[r].norm2 = load_tensor_f32(m, name);

        snprintf(name, sizeof(name), "token_refiner.blocks.%d.mlp.fc1.weight", r);
        load_tensor_w(m, name, &m->refiner[r].fc1);

        snprintf(name, sizeof(name), "token_refiner.blocks.%d.mlp.fc2.weight", r);
        load_tensor_w(m, name, &m->refiner[r].fc2);
    }

    m->refiner_final_norm = load_tensor_f32(m, "token_refiner.final_norm.weight");

    /* 5. Final layer */
    m->final_norm = load_tensor_f32(m, "final_layer.norm.weight");
    if (load_tensor_w(m, "final_layer.video_out.weight", &m->video_out) != FH3_OK) {
        fh3_dit_close(m); return FH3_ERR;
    }
    m->video_out_bias = load_tensor_f32(m, "final_layer.video_out.bias");

    if (load_tensor_w(m, "final_layer.audio_out.weight", &m->audio_out) != FH3_OK) {
        fh3_dit_close(m); return FH3_ERR;
    }
    m->audio_out_bias = load_tensor_f32(m, "final_layer.audio_out.bias");

    *out = m;
    return FH3_OK;
}

void fh3_dit_close(Fh3Dit *m) {
    if (!m) return;

    if (m->dequant_allocs) {
        for (int i = 0; i < m->n_dequant_allocs; i++) {
            if (m->dequant_allocs[i]) free(m->dequant_allocs[i]);
        }
        free(m->dequant_allocs);
        m->dequant_allocs = NULL;
        m->n_dequant_allocs = 0;
    }

    if (m->globals_data) {
        free(m->globals_data);
        m->globals_data = NULL;
    }

    fh3_gguf_close(&m->globals_gguf);
    fh3_adaln_close(&m->adaln);
    free(m);
}

int fh3_dit_refiner_count(const Fh3Dit *m) {
    if (!m) return 0;
    return m->refiner_run_count;
}

int fh3_dit_prepare_text(const Fh3Dit *m, const float *text_emb, int T, float *text_rows) {
    if (!m || !text_emb || !text_rows || T <= 0) return FH3_ERR;

    ((Fh3Dit *)m)->refiner_run_count++;

    int hidden = m->cfg.hidden_size;
    int num_heads = m->cfg.num_heads;
    int head_dim = m->cfg.head_dim;
    int inner_dim = num_heads * head_dim;
    int ffn_dim = m->cfg.ffn_dim;

    float *t_in = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));
    float *residual = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));
    float *normed = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));
    float *qkv = (float *)malloc((size_t)T * 3 * (size_t)inner_dim * sizeof(float));
    float *q = (float *)malloc((size_t)T * (size_t)inner_dim * sizeof(float));
    float *k = (float *)malloc((size_t)T * (size_t)inner_dim * sizeof(float));
    float *v = (float *)malloc((size_t)T * (size_t)inner_dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)T * (size_t)inner_dim * sizeof(float));
    float *proj_out = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));
    float *fc1_buf = (float *)malloc((size_t)T * 2 * (size_t)ffn_dim * sizeof(float));
    float *swiglu_buf = (float *)malloc((size_t)T * (size_t)ffn_dim * sizeof(float));
    float *fc2_buf = (float *)malloc((size_t)T * (size_t)hidden * sizeof(float));

    if (!t_in || !residual || !normed || !qkv || !q || !k || !v ||
        !attn_out || !proj_out || !fc1_buf || !swiglu_buf || !fc2_buf) {
        if (t_in) free(t_in);
        if (residual) free(residual);
        if (normed) free(normed);
        if (qkv) free(qkv);
        if (q) free(q);
        if (k) free(k);
        if (v) free(v);
        if (attn_out) free(attn_out);
        if (proj_out) free(proj_out);
        if (fc1_buf) free(fc1_buf);
        if (swiglu_buf) free(swiglu_buf);
        if (fc2_buf) free(fc2_buf);
        return FH3_E_NOMEM;
    }

    /* 1. Condition Projection */
    fh3_matmul(&m->condition_proj, text_emb, t_in, T, m->condition_proj_bias);

    /* 2. Refiner Blocks */
    for (int r = 0; r < m->cfg.num_refiner_layers; r++) {
        /* Attention branch */
        memcpy(residual, t_in, (size_t)T * (size_t)hidden * sizeof(float));
        memcpy(normed, t_in, (size_t)T * (size_t)hidden * sizeof(float));
        fh3_rmsnorm(normed, m->refiner[r].norm1, T, hidden, m->cfg.norm_eps);

        fh3_matmul(&m->refiner[r].qkv, normed, qkv, T, NULL);

        for (int i = 0; i < T; i++) {
            const float *tok_qkv = qkv + (size_t)i * (3 * (size_t)inner_dim);
            memcpy(q + (size_t)i * (size_t)inner_dim, tok_qkv, (size_t)inner_dim * sizeof(float));
            memcpy(k + (size_t)i * (size_t)inner_dim, tok_qkv + inner_dim, (size_t)inner_dim * sizeof(float));
            memcpy(v + (size_t)i * (size_t)inner_dim, tok_qkv + 2 * inner_dim, (size_t)inner_dim * sizeof(float));
        }

        fh3_rmsnorm(q, m->refiner[r].q_norm, T * num_heads, head_dim, m->cfg.qk_norm_eps);
        fh3_rmsnorm(k, m->refiner[r].k_norm, T * num_heads, head_dim, m->cfg.qk_norm_eps);

        fh3_attention(q, k, v, attn_out, T, num_heads, head_dim);

        fh3_matmul(&m->refiner[r].out, attn_out, proj_out, T, NULL);

        for (int j = 0; j < T * hidden; j++) {
            t_in[j] = residual[j] + proj_out[j];
        }

        /* Feed-Forward branch */
        memcpy(residual, t_in, (size_t)T * (size_t)hidden * sizeof(float));
        memcpy(normed, t_in, (size_t)T * (size_t)hidden * sizeof(float));
        fh3_rmsnorm(normed, m->refiner[r].norm2, T, hidden, m->cfg.norm_eps);

        fh3_matmul(&m->refiner[r].fc1, normed, fc1_buf, T, NULL);
        fh3_swiglu(fc1_buf, swiglu_buf, T, ffn_dim);
        fh3_matmul(&m->refiner[r].fc2, swiglu_buf, fc2_buf, T, NULL);

        for (int j = 0; j < T * hidden; j++) {
            t_in[j] = residual[j] + fc2_buf[j];
        }
    }

    /* 3. Final Norm */
    memcpy(text_rows, t_in, (size_t)T * (size_t)hidden * sizeof(float));
    fh3_rmsnorm(text_rows, m->refiner_final_norm, T, hidden, m->cfg.final_norm_eps);

    free(t_in); free(residual); free(normed); free(qkv); free(q); free(k); free(v);
    free(attn_out); free(proj_out); free(fc1_buf); free(swiglu_buf); free(fc2_buf);
    return FH3_OK;
}

int fh3_dit_forward(
    const Fh3Dit *m,
    Fh3Trunk *tr,
    const Fh3Seq *seq,
    int step,
    const float *x_video,
    const float *x_audio,
    const float *text_rows,
    float *v_video,
    float *v_audio
) {
    if (!m || !tr || !seq || !x_video || !x_audio || !text_rows || !v_video || !v_audio) {
        return FH3_ERR;
    }
    if (step < 0 || step >= m->adaln.n_steps) {
        return FH3_E_INVALID;
    }

    int N = seq->seq_len;
    int T = seq->num_text;
    int A = seq->num_audio;
    int V = seq->num_video;
    int hidden = m->cfg.hidden_size;
    int num_heads = m->cfg.num_heads;
    int head_dim = m->cfg.head_dim;
    int inner_dim = num_heads * head_dim;
    int ffn_dim = m->cfg.ffn_dim;

    float *h           = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *residual    = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *normed      = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *shift_buf   = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *scale_buf   = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *gate_buf    = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *qkv         = (float *)malloc((size_t)N * 3 * (size_t)inner_dim * sizeof(float));
    float *q           = (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float));
    float *k           = (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float));
    float *v           = (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float));
    float *attn_out    = (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float));
    float *proj_out    = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *fc1_buf     = (float *)malloc((size_t)N * 2 * (size_t)ffn_dim * sizeof(float));
    float *swiglu_buf  = (float *)malloc((size_t)N * (size_t)ffn_dim * sizeof(float));
    float *fc2_buf     = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *gate_compress_buf = (m->cfg.attn == FH3_ATTN_VSA)
        ? (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float))
        : NULL;

    if (!h || !residual || !normed || !shift_buf || !scale_buf || !gate_buf ||
        !qkv || !q || !k || !v || !attn_out || !proj_out || !fc1_buf || !swiglu_buf || !fc2_buf ||
        (m->cfg.attn == FH3_ATTN_VSA && !gate_compress_buf)) {
        if (h) free(h);
        if (residual) free(residual);
        if (normed) free(normed);
        if (shift_buf) free(shift_buf);
        if (scale_buf) free(scale_buf);
        if (gate_buf) free(gate_buf);
        if (qkv) free(qkv);
        if (q) free(q);
        if (k) free(k);
        if (v) free(v);
        if (attn_out) free(attn_out);
        if (proj_out) free(proj_out);
        if (fc1_buf) free(fc1_buf);
        if (swiglu_buf) free(swiglu_buf);
        if (fc2_buf) free(fc2_buf);
        if (gate_compress_buf) free(gate_compress_buf);
        return FH3_E_NOMEM;
    }

    /* 1. Modality patch embedding scatter */
    memcpy(h, text_rows, (size_t)T * (size_t)hidden * sizeof(float));
    fh3_matmul(&m->audio_patch_proj, x_audio, h + (size_t)T * (size_t)hidden, A, m->audio_patch_proj_bias);
    fh3_matmul(&m->video_patch_proj, x_video, h + (size_t)(T + A) * (size_t)hidden, V, m->video_patch_proj_bias);

    /* 2. Transformer Blocks */
    for (int L = 0; L < m->cfg.num_layers; L++) {
        if (L + 1 < m->cfg.num_layers) {
            fh3_trunk_prefetch(tr, L + 1);
        }

        const uint8_t *layer_buf = fh3_trunk_acquire(tr, L);
        if (!layer_buf) {
            free(h); free(residual); free(normed); free(shift_buf); free(scale_buf); free(gate_buf);
            free(qkv); free(q); free(k); free(v); free(attn_out); free(proj_out);
            free(fc1_buf); free(swiglu_buf); free(fc2_buf);
            if (gate_compress_buf) free(gate_compress_buf);
            return FH3_ERR;
        }

        Fh3LayerView view = fh3_trunk_layer(tr, L, layer_buf);

        /* --- Self-Attention branch --- */
        for (int i = 0; i < N; i++) {
            int pair = seq->pair_index[i];
            const float *s = fh3_adaln_row(&m->adaln, step, L, pair, 0);
            const float *c = fh3_adaln_row(&m->adaln, step, L, pair, 1);
            const float *g = fh3_adaln_row(&m->adaln, step, L, pair, 2);
            memcpy(shift_buf + (size_t)i * (size_t)hidden, s, (size_t)hidden * sizeof(float));
            memcpy(scale_buf + (size_t)i * (size_t)hidden, c, (size_t)hidden * sizeof(float));
            memcpy(gate_buf  + (size_t)i * (size_t)hidden, g, (size_t)hidden * sizeof(float));
        }

        memcpy(residual, h, (size_t)N * (size_t)hidden * sizeof(float));
        memcpy(normed, h, (size_t)N * (size_t)hidden * sizeof(float));
        fh3_rmsnorm(normed, (const float *)view.norm1.data, N, hidden, m->cfg.norm_eps);
        fh3_modulate(normed, shift_buf, scale_buf, N, hidden);

        fh3_matmul(&view.qkv, normed, qkv, N, NULL);

        for (int i = 0; i < N; i++) {
            const float *tok_qkv = qkv + (size_t)i * (3 * (size_t)inner_dim);
            memcpy(q + (size_t)i * (size_t)inner_dim, tok_qkv, (size_t)inner_dim * sizeof(float));
            memcpy(k + (size_t)i * (size_t)inner_dim, tok_qkv + inner_dim, (size_t)inner_dim * sizeof(float));
            memcpy(v + (size_t)i * (size_t)inner_dim, tok_qkv + 2 * inner_dim, (size_t)inner_dim * sizeof(float));
        }

        fh3_rmsnorm(q, (const float *)view.q_norm.data, N * num_heads, head_dim, m->cfg.qk_norm_eps);
        fh3_rmsnorm(k, (const float *)view.k_norm.data, N * num_heads, head_dim, m->cfg.qk_norm_eps);

        fh3_rope3(q, seq->cos, seq->sin, N, num_heads, head_dim, 96);
        fh3_rope3(k, seq->cos, seq->sin, N, num_heads, head_dim, 96);

        if (m->cfg.attn == FH3_ATTN_VSA) {
            const float *gate_ptr = NULL;
            if (view.gate_compress.data != NULL && view.gate_compress.out > 0 && gate_compress_buf) {
                fh3_matmul(&view.gate_compress, normed, gate_compress_buf, N, NULL);
                gate_ptr = gate_compress_buf;
            }
            fh3_vsa_attention(&m->cfg.vsa, q, k, v, gate_ptr, attn_out, seq, num_heads, head_dim);
        } else {
            fh3_attention(q, k, v, attn_out, N, num_heads, head_dim);
        }

        fh3_matmul(&view.out, attn_out, proj_out, N, NULL);

        fh3_gate_residual(residual, gate_buf, proj_out, N, hidden);
        memcpy(h, residual, (size_t)N * (size_t)hidden * sizeof(float));

        /* --- Feed-Forward branch --- */
        for (int i = 0; i < N; i++) {
            int pair = seq->pair_index[i];
            const float *s = fh3_adaln_row(&m->adaln, step, L, pair, 3);
            const float *c = fh3_adaln_row(&m->adaln, step, L, pair, 4);
            const float *g = fh3_adaln_row(&m->adaln, step, L, pair, 5);
            memcpy(shift_buf + (size_t)i * (size_t)hidden, s, (size_t)hidden * sizeof(float));
            memcpy(scale_buf + (size_t)i * (size_t)hidden, c, (size_t)hidden * sizeof(float));
            memcpy(gate_buf  + (size_t)i * (size_t)hidden, g, (size_t)hidden * sizeof(float));
        }

        memcpy(residual, h, (size_t)N * (size_t)hidden * sizeof(float));
        memcpy(normed, h, (size_t)N * (size_t)hidden * sizeof(float));
        fh3_rmsnorm(normed, (const float *)view.norm2.data, N, hidden, m->cfg.norm_eps);
        fh3_modulate(normed, shift_buf, scale_buf, N, hidden);

        fh3_matmul(&view.fc1, normed, fc1_buf, N, NULL);
        fh3_swiglu(fc1_buf, swiglu_buf, N, ffn_dim);
        fh3_matmul(&view.fc2, swiglu_buf, fc2_buf, N, NULL);

        fh3_gate_residual(residual, gate_buf, fc2_buf, N, hidden);
        memcpy(h, residual, (size_t)N * (size_t)hidden * sizeof(float));
    }

    /* 3. Final Layer Norm and AdaLN Modulation */
    memcpy(normed, h, (size_t)N * (size_t)hidden * sizeof(float));
    fh3_rmsnorm(normed, m->final_norm, N, hidden, m->cfg.final_norm_eps);

    for (int i = 0; i < N; i++) {
        int pair = seq->pair_index[i];
        const float *s = fh3_adaln_row(&m->adaln, step, -1, pair, 0);
        const float *c = fh3_adaln_row(&m->adaln, step, -1, pair, 1);
        memcpy(shift_buf + (size_t)i * (size_t)hidden, s, (size_t)hidden * sizeof(float));
        memcpy(scale_buf + (size_t)i * (size_t)hidden, c, (size_t)hidden * sizeof(float));
    }
    fh3_modulate(normed, shift_buf, scale_buf, N, hidden);

    /* 4. Output Heads */
    const float *audio_rows = normed + (size_t)T * (size_t)hidden;
    const float *video_rows = normed + (size_t)(T + A) * (size_t)hidden;

    fh3_matmul(&m->video_out, video_rows, v_video, V, m->video_out_bias);
    fh3_matmul(&m->audio_out, audio_rows, v_audio, A, m->audio_out_bias);

    free(h); free(residual); free(normed); free(shift_buf); free(scale_buf); free(gate_buf);
    free(qkv); free(q); free(k); free(v); free(attn_out); free(proj_out);
    free(fc1_buf); free(swiglu_buf); free(fc2_buf);
    if (gate_compress_buf) free(gate_compress_buf);
    return FH3_OK;
}
