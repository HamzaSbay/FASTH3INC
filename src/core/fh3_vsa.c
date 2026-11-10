#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include <omp.h>

#include "fh3/fh3.h"
#include "model/fh3_seq.h"
#include "core/fh3_vsa.h"

typedef struct {
    int num_tiles;
    int num_prefix_tiles;
    int num_video_tiles;
    int *tile_offsets; /* [num_tiles + 1] cumulative token counts */
    int *tile_tokens;  /* [N] token indices */
    int *vbs;          /* [num_tiles] token count per tile */
} VsaGeometry;

static void free_vsa_geometry(VsaGeometry *g) {
    if (!g) return;
    if (g->tile_offsets) free(g->tile_offsets);
    if (g->tile_tokens) free(g->tile_tokens);
    if (g->vbs) free(g->vbs);
    memset(g, 0, sizeof(*g));
}

static int build_vsa_geometry(const Fh3Seq *seq, int tile_size, VsaGeometry *g) {
    memset(g, 0, sizeof(*g));
    if (!seq || seq->seq_len <= 0) return -1;
    if (tile_size <= 0) tile_size = 64;

    int N = seq->seq_len;
    int T = (seq->num_text >= 0) ? seq->num_text : 0;
    int A = (seq->num_audio >= 0) ? seq->num_audio : 0;
    int V = (seq->num_video >= 0) ? seq->num_video : 0;

    if (T + A + V != N) {
        /* If modalities don't sum to N, treat all tokens as video */
        T = 0;
        A = 0;
        V = N;
    }

    /* 1. Count prefix tiles */
    int n_prefix_tiles = 0;
    if (T > 0) {
        n_prefix_tiles += (T + tile_size - 1) / tile_size;
    }
    if (A > 0) {
        n_prefix_tiles += (A + tile_size - 1) / tile_size;
    }

    /* 2. Count video tiles */
    int gt = seq->grid_t;
    int gh = seq->grid_h;
    int gw = seq->grid_w;
    if (gt <= 0 || gh <= 0 || gw <= 0 || (gt * gh * gw != V)) {
        gt = 1;
        gh = 1;
        gw = V;
    }

    int nt = (gt + 3) / 4;
    int nh = (gh + 3) / 4;
    int nw = (gw + 3) / 4;
    int n_video_tiles = (V > 0) ? (nt * nh * nw) : 0;

    int num_tiles = n_prefix_tiles + n_video_tiles;
    if (num_tiles <= 0) return -1;

    g->num_tiles = num_tiles;
    g->num_prefix_tiles = n_prefix_tiles;
    g->num_video_tiles = n_video_tiles;

    g->tile_offsets = (int *)malloc((size_t)(num_tiles + 1) * sizeof(int));
    g->tile_tokens = (int *)malloc((size_t)N * sizeof(int));
    g->vbs = (int *)malloc((size_t)num_tiles * sizeof(int));

    if (!g->tile_offsets || !g->tile_tokens || !g->vbs) {
        free_vsa_geometry(g);
        return -1;
    }

    int t_idx = 0;
    int tok_cursor = 0;
    g->tile_offsets[0] = 0;

    /* Text segment */
    if (T > 0) {
        int rem = T;
        int cur = 0;
        while (rem > 0) {
            int sz = (rem < tile_size) ? rem : tile_size;
            g->vbs[t_idx] = sz;
            for (int k = 0; k < sz; k++) {
                g->tile_tokens[tok_cursor++] = cur + k;
            }
            cur += sz;
            rem -= sz;
            t_idx++;
            g->tile_offsets[t_idx] = tok_cursor;
        }
    }

    /* Audio segment */
    if (A > 0) {
        int rem = A;
        int cur = T;
        while (rem > 0) {
            int sz = (rem < tile_size) ? rem : tile_size;
            g->vbs[t_idx] = sz;
            for (int k = 0; k < sz; k++) {
                g->tile_tokens[tok_cursor++] = cur + k;
            }
            cur += sz;
            rem -= sz;
            t_idx++;
            g->tile_offsets[t_idx] = tok_cursor;
        }
    }

    /* Video segment: 3D spatial-temporal tiling */
    if (V > 0) {
        int cursor = T + A;
        for (int tt = 0; tt < nt; tt++) {
            int t_start = tt * 4;
            int t_end = (tt + 1) * 4;
            if (t_end > gt) t_end = gt;

            for (int hh = 0; hh < nh; hh++) {
                int h_start = hh * 4;
                int h_end = (hh + 1) * 4;
                if (h_end > gh) h_end = gh;

                for (int ww = 0; ww < nw; ww++) {
                    int w_start = ww * 4;
                    int w_end = (ww + 1) * 4;
                    if (w_end > gw) w_end = gw;

                    int start_tok = tok_cursor;
                    for (int dt = t_start; dt < t_end; dt++) {
                        for (int dh = h_start; dh < h_end; dh++) {
                            for (int dw = w_start; dw < w_end; dw++) {
                                int idx = cursor + (dt * gh * gw + dh * gw + dw);
                                g->tile_tokens[tok_cursor++] = idx;
                            }
                        }
                    }
                    int sz = tok_cursor - start_tok;
                    g->vbs[t_idx] = sz;
                    t_idx++;
                    g->tile_offsets[t_idx] = tok_cursor;
                }
            }
        }
    }

    return 0;
}

static inline int compute_topk(float sparsity, int num_blocks) {
    if (num_blocks <= 0) return 0;
    int k = (int)ceilf((1.0f - sparsity) * (float)num_blocks);
    if (k < 1) k = 1;
    if (k > num_blocks) k = num_blocks;
    return k;
}

typedef struct {
    float score;
    int idx;
} TileScore;

static int cmp_tile_score(const void *a, const void *b) {
    const TileScore *ta = (const TileScore *)a;
    const TileScore *tb = (const TileScore *)b;
    if (ta->score > tb->score) return -1;
    if (ta->score < tb->score) return 1;
    return (ta->idx < tb->idx) ? -1 : ((ta->idx > tb->idx) ? 1 : 0);
}

void fh3_vsa_attention(
    const Fh3VsaCfg *c,
    const float *q,
    const float *k,
    const float *v,
    const float *gate,
    float *o,
    const Fh3Seq *seq,
    int heads,
    int head_dim
) {
    if (!q || !k || !v || !o || !seq || seq->seq_len <= 0 || heads <= 0 || head_dim <= 0) {
        return;
    }

    int N = seq->seq_len;
    int tile_size = (c && c->tile > 0) ? c->tile : 64;
    float sparsity = c ? c->sparsity : 0.8f;

    VsaGeometry geom;
    if (build_vsa_geometry(seq, tile_size, &geom) != 0) {
        return;
    }

    int num_tiles = geom.num_tiles;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* 1. Pool Q and K (and V if gate != NULL) per tile */
    float *q_pool = (float *)calloc((size_t)heads * (size_t)num_tiles * (size_t)head_dim, sizeof(float));
    float *k_pool = (float *)calloc((size_t)heads * (size_t)num_tiles * (size_t)head_dim, sizeof(float));
    float *v_pool = gate ? (float *)calloc((size_t)heads * (size_t)num_tiles * (size_t)head_dim, sizeof(float)) : NULL;

    if (!q_pool || !k_pool || (gate && !v_pool)) {
        if (q_pool) free(q_pool);
        if (k_pool) free(k_pool);
        if (v_pool) free(v_pool);
        free_vsa_geometry(&geom);
        return;
    }

    for (int i = 0; i < num_tiles; i++) {
        int vbs = geom.vbs[i];
        if (vbs <= 0) continue;
        float inv_vbs = 1.0f / (float)vbs;
        const int *toks = &geom.tile_tokens[geom.tile_offsets[i]];

        for (int tok_i = 0; tok_i < vbs; tok_i++) {
            int tok = toks[tok_i];
            for (int h = 0; h < heads; h++) {
                const float *q_ptr = q + ((size_t)tok * (size_t)heads + (size_t)h) * (size_t)head_dim;
                const float *k_ptr = k + ((size_t)tok * (size_t)heads + (size_t)h) * (size_t)head_dim;
                float *qp_ptr = q_pool + ((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)head_dim;
                float *kp_ptr = k_pool + ((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)head_dim;

                for (int d = 0; d < head_dim; d++) {
                    qp_ptr[d] += q_ptr[d] * inv_vbs;
                    kp_ptr[d] += k_ptr[d] * inv_vbs;
                }

                if (gate) {
                    const float *v_ptr = v + ((size_t)tok * (size_t)heads + (size_t)h) * (size_t)head_dim;
                    float *vp_ptr = v_pool + ((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        vp_ptr[d] += v_ptr[d] * inv_vbs;
                    }
                }
            }
        }
    }

    /* 2. Tile scores: S[h, i, j] = (Q_pool @ K_pool.T) / sqrt(dim) */
    size_t scores_size = (size_t)heads * (size_t)num_tiles * (size_t)num_tiles;
    float *scores = (float *)malloc(scores_size * sizeof(float));
    if (!scores) {
        free(q_pool);
        free(k_pool);
        if (v_pool) free(v_pool);
        free_vsa_geometry(&geom);
        return;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < heads; h++) {
        for (int i = 0; i < num_tiles; i++) {
            const float *qi = q_pool + ((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)head_dim;
            for (int j = 0; j < num_tiles; j++) {
                const float *kj = k_pool + ((size_t)h * (size_t)num_tiles + (size_t)j) * (size_t)head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += qi[d] * kj[d];
                }
                scores[((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)num_tiles + (size_t)j] = dot * scale;
            }
        }
    }

    /* 3. Build tile mask: (heads, num_tiles, num_tiles) */
    uint8_t *tile_mask = (uint8_t *)calloc(scores_size, sizeof(uint8_t));
    if (!tile_mask) {
        free(scores);
        free(q_pool);
        free(k_pool);
        if (v_pool) free(v_pool);
        free_vsa_geometry(&geom);
        return;
    }

    int k_video = compute_topk(sparsity, geom.num_video_tiles);

    // Prefix queries are always dense across all key tiles
    for (int h = 0; h < heads; h++) {
        for (int i = 0; i < geom.num_prefix_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {
                tile_mask[((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)num_tiles + (size_t)j] = 1;
            }
        }
    }

    // Video queries
    if (sparsity <= 0.0f || k_video == geom.num_video_tiles) {
        // Full dense selection
        for (int h = 0; h < heads; h++) {
            for (int i = geom.num_prefix_tiles; i < num_tiles; i++) {
                for (int j = 0; j < num_tiles; j++) {
                    tile_mask[((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)num_tiles + (size_t)j] = 1;
                }
            }
        }
    } else {
        // Prefix keys are exempt: always selected
        for (int h = 0; h < heads; h++) {
            for (int i = geom.num_prefix_tiles; i < num_tiles; i++) {
                for (int j = 0; j < geom.num_prefix_tiles; j++) {
                    tile_mask[((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)num_tiles + (size_t)j] = 1;
                }
            }
        }

        // Top-k video keys selected per video query tile
        #pragma omp parallel
        {
            TileScore *candidates = (TileScore *)malloc((size_t)geom.num_video_tiles * sizeof(TileScore));
            #pragma omp for collapse(2) schedule(static)
            for (int h = 0; h < heads; h++) {
                for (int i = geom.num_prefix_tiles; i < num_tiles; i++) {
                    if (!candidates) continue;
                    for (int vj = 0; vj < geom.num_video_tiles; vj++) {
                        int j = geom.num_prefix_tiles + vj;
                        candidates[vj].score = scores[((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)num_tiles + (size_t)j];
                        candidates[vj].idx = j;
                    }

                    qsort(candidates, (size_t)geom.num_video_tiles, sizeof(TileScore), cmp_tile_score);

                    for (int k_idx = 0; k_idx < k_video; k_idx++) {
                        int j = candidates[k_idx].idx;
                        tile_mask[((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)num_tiles + (size_t)j] = 1;
                    }
                }
            }
            if (candidates) free(candidates);
        }
    }

    /* 4. Fine Attention per query tile */
    size_t out_elements = (size_t)N * (size_t)heads * (size_t)head_dim;
    float *out_fine = (float *)calloc(out_elements, sizeof(float));
    if (!out_fine) {
        free(tile_mask);
        free(scores);
        free(q_pool);
        free(k_pool);
        if (v_pool) free(v_pool);
        free_vsa_geometry(&geom);
        return;
    }

    #pragma omp parallel
    {
        int *k_toks = (int *)malloc((size_t)N * sizeof(int));
        float *attn_scores = (float *)malloc((size_t)N * sizeof(float));
        float *p_buf = (float *)malloc((size_t)N * sizeof(float));

        #pragma omp for collapse(2) schedule(dynamic)
        for (int h = 0; h < heads; h++) {
            for (int i = 0; i < num_tiles; i++) {
                if (!k_toks || !attn_scores || !p_buf) continue;

                // Collect tokens from selected key tiles for (h, i)
                int num_k = 0;
                for (int j = 0; j < num_tiles; j++) {
                    if (tile_mask[((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)num_tiles + (size_t)j]) {
                        int t_start = geom.tile_offsets[j];
                        int t_cnt = geom.vbs[j];
                        for (int t = 0; t < t_cnt; t++) {
                            k_toks[num_k++] = geom.tile_tokens[t_start + t];
                        }
                    }
                }

                if (num_k == 0) continue;

                int q_start = geom.tile_offsets[i];
                int q_cnt = geom.vbs[i];

                for (int qk = 0; qk < q_cnt; qk++) {
                    int q_tok = geom.tile_tokens[q_start + qk];
                    const float *qi = q + ((size_t)q_tok * (size_t)heads + (size_t)h) * (size_t)head_dim;

                    float max_s = -1e30f;
                    for (int m = 0; m < num_k; m++) {
                        int k_tok = k_toks[m];
                        const float *kj = k + ((size_t)k_tok * (size_t)heads + (size_t)h) * (size_t)head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            dot += qi[d] * kj[d];
                        }
                        float s = dot * scale;
                        attn_scores[m] = s;
                        if (s > max_s) max_s = s;
                    }

                    float sum_exp = 0.0f;
                    for (int m = 0; m < num_k; m++) {
                        float p = expf(attn_scores[m] - max_s);
                        p_buf[m] = p;
                        sum_exp += p;
                    }

                    float inv_sum = 1.0f / sum_exp;
                    float *out_tok = out_fine + ((size_t)q_tok * (size_t)heads + (size_t)h) * (size_t)head_dim;

                    for (int d = 0; d < head_dim; d++) {
                        out_tok[d] = 0.0f;
                    }

                    for (int m = 0; m < num_k; m++) {
                        int k_tok = k_toks[m];
                        const float *vj = v + ((size_t)k_tok * (size_t)heads + (size_t)h) * (size_t)head_dim;
                        float weight = p_buf[m] * inv_sum;
                        for (int d = 0; d < head_dim; d++) {
                            out_tok[d] += weight * vj[d];
                        }
                    }
                }
            }
        }

        if (k_toks) free(k_toks);
        if (attn_scores) free(attn_scores);
        if (p_buf) free(p_buf);
    }

    /* 5. Coarse Gated Attention (if gate != NULL) */
    if (gate && v_pool) {
        size_t out_c_size = (size_t)num_tiles * (size_t)heads * (size_t)head_dim;
        float *out_c = (float *)malloc(out_c_size * sizeof(float));

        if (out_c) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < num_tiles; i++) {
                for (int h = 0; h < heads; h++) {
                    const float *s_row = scores + ((size_t)h * (size_t)num_tiles + (size_t)i) * (size_t)num_tiles;

                    float max_s = -1e30f;
                    for (int j = 0; j < num_tiles; j++) {
                        if (s_row[j] > max_s) max_s = s_row[j];
                    }

                    float sum_exp = 0.0f;
                    for (int j = 0; j < num_tiles; j++) {
                        sum_exp += expf(s_row[j] - max_s);
                    }
                    float inv_sum = 1.0f / sum_exp;

                    float *out_ci = out_c + ((size_t)i * (size_t)heads + (size_t)h) * (size_t)head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        out_ci[d] = 0.0f;
                    }

                    for (int j = 0; j < num_tiles; j++) {
                        float weight = expf(s_row[j] - max_s) * inv_sum;
                        const float *vp = v_pool + ((size_t)h * (size_t)num_tiles + (size_t)j) * (size_t)head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            out_ci[d] += weight * vp[d];
                        }
                    }
                }
            }

            // Combine: O = O_fine + O_coarse * gate
            for (int i = 0; i < num_tiles; i++) {
                int t_start = geom.tile_offsets[i];
                int t_cnt = geom.vbs[i];
                for (int k_idx = 0; k_idx < t_cnt; k_idx++) {
                    int tok = geom.tile_tokens[t_start + k_idx];
                    for (int h = 0; h < heads; h++) {
                        size_t tok_off = ((size_t)tok * (size_t)heads + (size_t)h) * (size_t)head_dim;
                        size_t c_off = ((size_t)i * (size_t)heads + (size_t)h) * (size_t)head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            o[tok_off + d] = out_fine[tok_off + d] + out_c[c_off + d] * gate[tok_off + d];
                        }
                    }
                }
            }

            free(out_c);
        } else {
            memcpy(o, out_fine, out_elements * sizeof(float));
        }
    } else {
        memcpy(o, out_fine, out_elements * sizeof(float));
    }

    free(out_fine);
    free(tile_mask);
    free(scores);
    free(q_pool);
    free(k_pool);
    if (v_pool) free(v_pool);
    free_vsa_geometry(&geom);
}
