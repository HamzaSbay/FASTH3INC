#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"
#include "io/fh3_pack.h"
#include "io/fh3_portable_io.h"

#define FH3_BLOCK_IO_MAGIC 0x49334846U  /* 'FH3I' */
#define FH3_BLOCK_IO_VERSION 1U

#pragma pack(push, 1)
typedef struct Fh3BlockInputHeader {
    uint32_t magic;           /* FH3_BLOCK_IO_MAGIC */
    uint32_t version;         /* FH3_BLOCK_IO_VERSION */
    int32_t n_tokens;         /* N */
    int32_t hidden_size;      /* hidden */
    int32_t num_heads;        /* heads */
    int32_t head_dim;         /* head_dim */
    int32_t rope_rot_dim;     /* e.g. 96 */
    int32_t step;             /* default 0 */
    int32_t pair;             /* default 0 */
    float norm_eps;           /* default 1e-5f */
    float qk_norm_eps;        /* default 1e-5f */
    int32_t has_pair_indices; /* 0: uniform pair, 1: N int32s follow */
    uint8_t reserved[48];     /* pad struct to 96 bytes */
} Fh3BlockInputHeader;
#pragma pack(pop)

static const char *resolve_path(const char *path, char *buf, size_t cap) {
    FILE *fp = fopen(path, "rb");
    if (fp) {
        fclose(fp);
        return path;
    }
    snprintf(buf, cap, "../%s", path);
    fp = fopen(buf, "rb");
    if (fp) {
        fclose(fp);
        return buf;
    }
    snprintf(buf, cap, "../../%s", path);
    fp = fopen(buf, "rb");
    if (fp) {
        fclose(fp);
        return buf;
    }
    return path;
}

static float compute_rel_l2(const float *a, const float *b, size_t n) {
    double sum_diff = 0.0;
    double sum_ref = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum_diff += d * d;
        sum_ref += (double)b[i] * (double)b[i];
    }
    if (sum_ref == 0.0) return (float)sqrt(sum_diff);
    return (float)(sqrt(sum_diff) / sqrt(sum_ref));
}

static int run_block_forward(
    const Fh3LayerView *view,
    const Fh3Adaln *adaln,
    int step,
    int mod_block,
    const int *pair_indices,
    int uniform_pair,
    int n_tokens,
    int hidden_size,
    int num_heads,
    int head_dim,
    int rope_rot_dim,
    float norm_eps,
    float qk_norm_eps,
    const float *x,
    const float *cos,
    const float *sin,
    float *out_y
) {
    int N = n_tokens;
    int hidden = hidden_size;
    int inner_dim = num_heads * head_dim;
    int ffn_dim = (int)(view->fc1.out / 2);

    /* Check 1D norm weights for dequantization if not F32 */
    float *norm1_alloc = NULL;
    float *norm2_alloc = NULL;
    float *q_norm_alloc = NULL;
    float *k_norm_alloc = NULL;

    const float *w_norm1 = NULL;
    if (view->norm1.type == FH3_QUANT_F32) {
        w_norm1 = (const float *)view->norm1.data;
    } else {
        norm1_alloc = (float *)malloc((size_t)hidden * sizeof(float));
        if (!norm1_alloc) return -1;
        fh3_dequant(view->norm1.type, view->norm1.data, norm1_alloc, hidden);
        w_norm1 = norm1_alloc;
    }

    const float *w_norm2 = NULL;
    if (view->norm2.type == FH3_QUANT_F32) {
        w_norm2 = (const float *)view->norm2.data;
    } else {
        norm2_alloc = (float *)malloc((size_t)hidden * sizeof(float));
        if (!norm2_alloc) { free(norm1_alloc); return -1; }
        fh3_dequant(view->norm2.type, view->norm2.data, norm2_alloc, hidden);
        w_norm2 = norm2_alloc;
    }

    const float *w_q_norm = NULL;
    if (view->q_norm.type == FH3_QUANT_F32) {
        w_q_norm = (const float *)view->q_norm.data;
    } else {
        q_norm_alloc = (float *)malloc((size_t)head_dim * sizeof(float));
        if (!q_norm_alloc) { free(norm1_alloc); free(norm2_alloc); return -1; }
        fh3_dequant(view->q_norm.type, view->q_norm.data, q_norm_alloc, head_dim);
        w_q_norm = q_norm_alloc;
    }

    const float *w_k_norm = NULL;
    if (view->k_norm.type == FH3_QUANT_F32) {
        w_k_norm = (const float *)view->k_norm.data;
    } else {
        k_norm_alloc = (float *)malloc((size_t)head_dim * sizeof(float));
        if (!k_norm_alloc) { free(norm1_alloc); free(norm2_alloc); free(q_norm_alloc); return -1; }
        fh3_dequant(view->k_norm.type, view->k_norm.data, k_norm_alloc, head_dim);
        w_k_norm = k_norm_alloc;
    }

    /* Scratch buffers */
    float *residual = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *normed   = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *shift_buf = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *scale_buf = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *gate_buf  = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));

    float *qkv      = (float *)malloc((size_t)N * (3 * (size_t)inner_dim) * sizeof(float));
    float *q        = (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float));
    float *k        = (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float));
    float *v        = (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)N * (size_t)inner_dim * sizeof(float));
    float *proj_out = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));

    float *fc1_buf    = (float *)malloc((size_t)N * (2 * (size_t)ffn_dim) * sizeof(float));
    float *swiglu_buf = (float *)malloc((size_t)N * (size_t)ffn_dim * sizeof(float));
    float *fc2_buf    = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));

    if (!residual || !normed || !shift_buf || !scale_buf || !gate_buf ||
        !qkv || !q || !k || !v || !attn_out || !proj_out ||
        !fc1_buf || !swiglu_buf || !fc2_buf) {
        free(residual); free(normed); free(shift_buf); free(scale_buf); free(gate_buf);
        free(qkv); free(q); free(k); free(v); free(attn_out); free(proj_out);
        free(fc1_buf); free(swiglu_buf); free(fc2_buf);
        free(norm1_alloc); free(norm2_alloc); free(q_norm_alloc); free(k_norm_alloc);
        return -1;
    }

    /* -------------------------------------------------------------------------
     * 1. Self-Attention Branch
     * ------------------------------------------------------------------------- */
    for (int i = 0; i < N; i++) {
        int pair = pair_indices ? pair_indices[i] : uniform_pair;
        const float *s = fh3_adaln_row(adaln, step, mod_block, pair, 0);
        const float *c = fh3_adaln_row(adaln, step, mod_block, pair, 1);
        const float *g = fh3_adaln_row(adaln, step, mod_block, pair, 2);
        if (!s || !c || !g) {
            fprintf(stderr, "Error: missing adaln trunk row for step=%d, mod_block=%d, pair=%d\n", step, mod_block, pair);
            free(residual); free(normed); free(shift_buf); free(scale_buf); free(gate_buf);
            free(qkv); free(q); free(k); free(v); free(attn_out); free(proj_out);
            free(fc1_buf); free(swiglu_buf); free(fc2_buf);
            free(norm1_alloc); free(norm2_alloc); free(q_norm_alloc); free(k_norm_alloc);
            return -1;
        }
        memcpy(shift_buf + (size_t)i * (size_t)hidden, s, (size_t)hidden * sizeof(float));
        memcpy(scale_buf + (size_t)i * (size_t)hidden, c, (size_t)hidden * sizeof(float));
        memcpy(gate_buf  + (size_t)i * (size_t)hidden, g, (size_t)hidden * sizeof(float));
    }

    memcpy(residual, x, (size_t)N * (size_t)hidden * sizeof(float));
    memcpy(normed, x, (size_t)N * (size_t)hidden * sizeof(float));
    fh3_rmsnorm(normed, w_norm1, N, hidden, norm_eps);
    fh3_modulate(normed, shift_buf, scale_buf, N, hidden);

    fh3_matmul(&view->qkv, normed, qkv, N, NULL);

    for (int i = 0; i < N; i++) {
        const float *tok_qkv = qkv + (size_t)i * (3 * (size_t)inner_dim);
        memcpy(q + (size_t)i * (size_t)inner_dim, tok_qkv, (size_t)inner_dim * sizeof(float));
        memcpy(k + (size_t)i * (size_t)inner_dim, tok_qkv + inner_dim, (size_t)inner_dim * sizeof(float));
        memcpy(v + (size_t)i * (size_t)inner_dim, tok_qkv + 2 * inner_dim, (size_t)inner_dim * sizeof(float));
    }

    fh3_rmsnorm(q, w_q_norm, N * num_heads, head_dim, qk_norm_eps);
    fh3_rmsnorm(k, w_k_norm, N * num_heads, head_dim, qk_norm_eps);

    fh3_rope3(q, cos, sin, N, num_heads, head_dim, rope_rot_dim);
    fh3_rope3(k, cos, sin, N, num_heads, head_dim, rope_rot_dim);

    fh3_attention(q, k, v, attn_out, N, num_heads, head_dim);

    fh3_matmul(&view->out, attn_out, proj_out, N, NULL);

    fh3_gate_residual(residual, gate_buf, proj_out, N, hidden);

    /* -------------------------------------------------------------------------
     * 2. Feed-Forward Branch
     * ------------------------------------------------------------------------- */
    for (int i = 0; i < N; i++) {
        int pair = pair_indices ? pair_indices[i] : uniform_pair;
        const float *s = fh3_adaln_row(adaln, step, mod_block, pair, 3);
        const float *c = fh3_adaln_row(adaln, step, mod_block, pair, 4);
        const float *g = fh3_adaln_row(adaln, step, mod_block, pair, 5);
        if (!s || !c || !g) {
            fprintf(stderr, "Error: missing adaln trunk row for step=%d, mod_block=%d, pair=%d (mlp)\n", step, mod_block, pair);
            free(residual); free(normed); free(shift_buf); free(scale_buf); free(gate_buf);
            free(qkv); free(q); free(k); free(v); free(attn_out); free(proj_out);
            free(fc1_buf); free(swiglu_buf); free(fc2_buf);
            free(norm1_alloc); free(norm2_alloc); free(q_norm_alloc); free(k_norm_alloc);
            return -1;
        }
        memcpy(shift_buf + (size_t)i * (size_t)hidden, s, (size_t)hidden * sizeof(float));
        memcpy(scale_buf + (size_t)i * (size_t)hidden, c, (size_t)hidden * sizeof(float));
        memcpy(gate_buf  + (size_t)i * (size_t)hidden, g, (size_t)hidden * sizeof(float));
    }

    memcpy(normed, residual, (size_t)N * (size_t)hidden * sizeof(float));
    fh3_rmsnorm(normed, w_norm2, N, hidden, norm_eps);
    fh3_modulate(normed, shift_buf, scale_buf, N, hidden);

    fh3_matmul(&view->fc1, normed, fc1_buf, N, NULL);
    fh3_swiglu(fc1_buf, swiglu_buf, N, ffn_dim);
    fh3_matmul(&view->fc2, swiglu_buf, fc2_buf, N, NULL);

    fh3_gate_residual(residual, gate_buf, fc2_buf, N, hidden);

    memcpy(out_y, residual, (size_t)N * (size_t)hidden * sizeof(float));

    /* Clean up */
    free(residual); free(normed); free(shift_buf); free(scale_buf); free(gate_buf);
    free(qkv); free(q); free(k); free(v); free(attn_out); free(proj_out);
    free(fc1_buf); free(swiglu_buf); free(fc2_buf);
    if (norm1_alloc) free(norm1_alloc);
    if (norm2_alloc) free(norm2_alloc);
    if (q_norm_alloc) free(q_norm_alloc);
    if (k_norm_alloc) free(k_norm_alloc);

    return 0;
}

static int run_self_test(void) {
    char trunk_path[512];
    char adaln_path[512];

    const char *tp = resolve_path("tests/fixtures/tiny_h3/dit_trunk.bin", trunk_path, sizeof(trunk_path));
    const char *ap = resolve_path("tests/fixtures/tiny_h3/adaln_table.bin", adaln_path, sizeof(adaln_path));

    FILE *fp1 = fopen(tp, "rb");
    FILE *fp2 = fopen(ap, "rb");
    if (!fp1 || !fp2) {
        if (fp1) fclose(fp1);
        if (fp2) fclose(fp2);
        fprintf(stderr, "[WARN] test_real_block: tiny_h3 fixtures not found, skipping self-test\n");
        return 0;
    }
    fclose(fp1);
    fclose(fp2);

    Fh3Pack pack;
    memset(&pack, 0, sizeof(pack));
    int rc = fh3_pack_open(tp, &pack);
    if (rc != FH3_OK) {
        fprintf(stderr, "[FAIL] test_real_block: cannot open pack '%s' (rc=%d)\n", tp, rc);
        return 1;
    }

    Fh3Adaln adaln;
    memset(&adaln, 0, sizeof(adaln));
    rc = fh3_adaln_open(ap, 0, &adaln);
    if (rc != FH3_OK) {
        fprintf(stderr, "[FAIL] test_real_block: cannot open adaln table '%s' (rc=%d)\n", ap, rc);
        fh3_pack_close(&pack);
        return 1;
    }

    Fh3LayerView view = fh3_pack_layer(&pack, 0, NULL);
    if (!view.qkv.data) {
        fprintf(stderr, "[FAIL] test_real_block: layer 0 view invalid\n");
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    int N = 18;
    int hidden = (int)view.qkv.in;
    int num_heads = 2;
    int head_dim = 128;
    int rope_rot_dim = 96;

    float *x = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *cos = (float *)malloc((size_t)N * (size_t)rope_rot_dim * sizeof(float));
    float *sin = (float *)malloc((size_t)N * (size_t)rope_rot_dim * sizeof(float));
    float *y0 = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));
    float *y1 = (float *)malloc((size_t)N * (size_t)hidden * sizeof(float));

    if (!x || !cos || !sin || !y0 || !y1) {
        fprintf(stderr, "[FAIL] test_real_block: out of memory in self-test\n");
        free(x); free(cos); free(sin); free(y0); free(y1);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    for (int i = 0; i < N * hidden; i++) {
        x[i] = sinf((float)i * 0.017f);
    }
    for (int i = 0; i < N * rope_rot_dim; i++) {
        cos[i] = cosf((float)i * 0.031f);
        sin[i] = sinf((float)i * 0.031f);
    }

    /* Run correct block 0 */
    rc = run_block_forward(&view, &adaln, 0, 0, NULL, 0, N, hidden, num_heads, head_dim,
                           rope_rot_dim, 1e-5f, 1e-5f, x, cos, sin, y0);
    if (rc != 0) {
        fprintf(stderr, "[FAIL] test_real_block: run_block_forward correct failed\n");
        free(x); free(cos); free(sin); free(y0); free(y1);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    /* Run misbound block 1 mod */
    rc = run_block_forward(&view, &adaln, 0, 1, NULL, 0, N, hidden, num_heads, head_dim,
                           rope_rot_dim, 1e-5f, 1e-5f, x, cos, sin, y1);
    if (rc != 0) {
        fprintf(stderr, "[FAIL] test_real_block: run_block_forward misbound failed\n");
        free(x); free(cos); free(sin); free(y0); free(y1);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    float rel = compute_rel_l2(y0, y1, (size_t)N * (size_t)hidden);
    if (rel <= 0.01f) {
        fprintf(stderr, "[FAIL] test_real_block: misbound rel_L2 too low: %f <= 0.01\n", rel);
        free(x); free(cos); free(sin); free(y0); free(y1);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    printf("[PASS] test_real_block self-test (misbound rel_L2 = %.4f > 0.01)\n", rel);

    free(x); free(cos); free(sin); free(y0); free(y1);
    fh3_adaln_close(&adaln);
    fh3_pack_close(&pack);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        return run_self_test();
    }

    const char *pack_dir = NULL;
    int block_l = -1;
    const char *in_path = NULL;
    const char *out_path = NULL;
    int cli_step = -1;
    int cli_pair = -1;
    int cli_mod_block = -1;
    bool dense_flag = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pack") == 0 && i + 1 < argc) {
            pack_dir = argv[++i];
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            block_l = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) {
            in_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            cli_step = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pair") == 0 && i + 1 < argc) {
            cli_pair = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mod-block") == 0 && i + 1 < argc) {
            cli_mod_block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dense") == 0) {
            dense_flag = true;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            return 1;
        }
    }

    (void)dense_flag;

    if (!pack_dir || block_l < 0 || !in_path || !out_path) {
        fprintf(stderr, "Usage: %s --pack <pack_dir> --block <L> --in <in_path> --out <out_path> [--step <step>] [--pair <pair>] [--mod-block <M>] [--dense]\n", argv[0]);
        return 1;
    }

    /* Open pack file and adaln table */
    char trunk_path[4096];
    char adaln_path[4096];

    size_t plen = strlen(pack_dir);
    if (plen > 14 && strcmp(pack_dir + plen - 14, "dit_trunk.bin") == 0) {
        snprintf(trunk_path, sizeof(trunk_path), "%s", pack_dir);
        /* Strip filename to get directory for adaln */
        char tmp[1024];
        snprintf(tmp, sizeof(tmp), "%.*s", (int)(sizeof(tmp) - 1), pack_dir);
        char *slash = strrchr(tmp, '/');
        if (!slash) slash = strrchr(tmp, '\\');
        if (slash) {
            *slash = '\0';
            snprintf(adaln_path, sizeof(adaln_path), "%s/adaln_table.bin", tmp);
        } else {
            snprintf(adaln_path, sizeof(adaln_path), "adaln_table.bin");
        }
    } else {
        if (plen > 0 && (pack_dir[plen - 1] == '/' || pack_dir[plen - 1] == '\\')) {
            snprintf(trunk_path, sizeof(trunk_path), "%sdit_trunk.bin", pack_dir);
            snprintf(adaln_path, sizeof(adaln_path), "%sadaln_table.bin", pack_dir);
        } else {
            snprintf(trunk_path, sizeof(trunk_path), "%s/dit_trunk.bin", pack_dir);
            snprintf(adaln_path, sizeof(adaln_path), "%s/adaln_table.bin", pack_dir);
        }
    }

    Fh3Pack pack;
    memset(&pack, 0, sizeof(pack));
    int rc = fh3_pack_open(trunk_path, &pack);
    if (rc != FH3_OK) {
        fprintf(stderr, "Error: cannot open pack '%s' (rc=%d)\n", trunk_path, rc);
        return 1;
    }

    if (block_l < 0 || block_l >= pack.num_layers) {
        fprintf(stderr, "Error: block index %d out of range [0, %d)\n", block_l, pack.num_layers);
        fh3_pack_close(&pack);
        return 1;
    }

    Fh3Adaln adaln;
    memset(&adaln, 0, sizeof(adaln));
    rc = fh3_adaln_open(adaln_path, 0, &adaln);
    if (rc != FH3_OK) {
        fprintf(stderr, "Error: cannot open adaln table '%s' (rc=%d)\n", adaln_path, rc);
        fh3_pack_close(&pack);
        return 1;
    }

    /* Read input file */
    FILE *in_fp = fopen(in_path, "rb");
    if (!in_fp) {
        fprintf(stderr, "Error: cannot open input file '%s'\n", in_path);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    Fh3BlockInputHeader in_hdr;
    if (fread(&in_hdr, sizeof(in_hdr), 1, in_fp) != 1) {
        fprintf(stderr, "Error: cannot read header from '%s'\n", in_path);
        fclose(in_fp);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    if (in_hdr.magic != FH3_BLOCK_IO_MAGIC || in_hdr.version != FH3_BLOCK_IO_VERSION) {
        fprintf(stderr, "Error: invalid input file header magic (0x%08X) or version (%u)\n",
                in_hdr.magic, in_hdr.version);
        fclose(in_fp);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    int N = in_hdr.n_tokens;
    int hidden = in_hdr.hidden_size;
    int num_heads = in_hdr.num_heads;
    int head_dim = in_hdr.head_dim;
    int rope_rot_dim = in_hdr.rope_rot_dim > 0 ? in_hdr.rope_rot_dim : 96;
    float norm_eps = in_hdr.norm_eps > 0.0f ? in_hdr.norm_eps : 1e-5f;
    float qk_norm_eps = in_hdr.qk_norm_eps > 0.0f ? in_hdr.qk_norm_eps : 1e-5f;
    int step = cli_step >= 0 ? cli_step : in_hdr.step;
    int pair = cli_pair >= 0 ? cli_pair : in_hdr.pair;
    int mod_block = cli_mod_block >= 0 ? cli_mod_block : block_l;

    int *pair_indices = NULL;
    if (in_hdr.has_pair_indices) {
        pair_indices = (int *)malloc((size_t)N * sizeof(int));
        if (!pair_indices || fread(pair_indices, sizeof(int32_t), (size_t)N, in_fp) != (size_t)N) {
            fprintf(stderr, "Error reading pair indices\n");
            free(pair_indices);
            fclose(in_fp);
            fh3_adaln_close(&adaln);
            fh3_pack_close(&pack);
            return 1;
        }
    }

    size_t x_bytes = (size_t)N * (size_t)hidden * sizeof(float);
    size_t cos_bytes = (size_t)N * (size_t)rope_rot_dim * sizeof(float);
    size_t sin_bytes = (size_t)N * (size_t)rope_rot_dim * sizeof(float);

    float *x = (float *)malloc(x_bytes);
    float *cos = (float *)malloc(cos_bytes);
    float *sin = (float *)malloc(sin_bytes);
    float *out_y = (float *)malloc(x_bytes);

    if (!x || !cos || !sin || !out_y) {
        fprintf(stderr, "Error: out of memory for input/output buffers\n");
        free(pair_indices); free(x); free(cos); free(sin); free(out_y);
        fclose(in_fp);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    if (fread(x, 1, x_bytes, in_fp) != x_bytes ||
        fread(cos, 1, cos_bytes, in_fp) != cos_bytes ||
        fread(sin, 1, sin_bytes, in_fp) != sin_bytes) {
        fprintf(stderr, "Error reading tensors from '%s'\n", in_path);
        free(pair_indices); free(x); free(cos); free(sin); free(out_y);
        fclose(in_fp);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }
    fclose(in_fp);

    /* Fetch layer weights */
    uint8_t *layer_buf = NULL;
    const uint8_t *base = NULL;
    if (pack.mapped_base) {
        base = NULL;
    } else {
        uint64_t lsz = pack.layers[block_l].size_bytes;
        layer_buf = (uint8_t *)malloc((size_t)lsz);
        if (!layer_buf) {
            fprintf(stderr, "Error: out of memory allocating layer buffer\n");
            free(pair_indices); free(x); free(cos); free(sin); free(out_y);
            fh3_adaln_close(&adaln);
            fh3_pack_close(&pack);
            return 1;
        }
        int64_t nread = fh3_pread(&pack.file, layer_buf, (size_t)lsz, pack.layers[block_l].file_offset);
        if (nread != (int64_t)lsz) {
            fprintf(stderr, "Error: pread failed reading layer %d\n", block_l);
            free(layer_buf);
            free(pair_indices); free(x); free(cos); free(sin); free(out_y);
            fh3_adaln_close(&adaln);
            fh3_pack_close(&pack);
            return 1;
        }
        base = layer_buf;
    }

    Fh3LayerView view = fh3_pack_layer(&pack, block_l, base);
    if (!view.qkv.data) {
        fprintf(stderr, "Error: failed to fetch layer %d from pack\n", block_l);
        if (layer_buf) free(layer_buf);
        free(pair_indices); free(x); free(cos); free(sin); free(out_y);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    rc = run_block_forward(&view, &adaln, step, mod_block, pair_indices, pair,
                           N, hidden, num_heads, head_dim, rope_rot_dim,
                           norm_eps, qk_norm_eps, x, cos, sin, out_y);
    if (rc != 0) {
        fprintf(stderr, "Error executing block %d forward pass\n", block_l);
        if (layer_buf) free(layer_buf);
        free(pair_indices); free(x); free(cos); free(sin); free(out_y);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    /* Write output file */
    FILE *out_fp = fopen(out_path, "wb");
    if (!out_fp) {
        fprintf(stderr, "Error: cannot open output file '%s' for writing\n", out_path);
        if (layer_buf) free(layer_buf);
        free(pair_indices); free(x); free(cos); free(sin); free(out_y);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    size_t written = fwrite(out_y, sizeof(float), (size_t)N * (size_t)hidden, out_fp);
    fclose(out_fp);

    if (written != (size_t)N * (size_t)hidden) {
        fprintf(stderr, "Error: failed to write all floats to '%s'\n", out_path);
        if (layer_buf) free(layer_buf);
        free(pair_indices); free(x); free(cos); free(sin); free(out_y);
        fh3_adaln_close(&adaln);
        fh3_pack_close(&pack);
        return 1;
    }

    /* Clean up */
    if (layer_buf) free(layer_buf);
    free(pair_indices);
    free(x);
    free(cos);
    free(sin);
    free(out_y);
    fh3_adaln_close(&adaln);
    fh3_pack_close(&pack);

    return 0;
}
