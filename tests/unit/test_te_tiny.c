#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"
#include "model/fh3_te.h"
#include "io/fh3_trunk.h"
#include "io/fh3_portable_io.h"

static int g_failures = 0;

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

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return 1; \
    } \
} while (0)

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

static void *read_file_bytes(const char *path, size_t *out_sz) {
    char pbuf[512];
    const char *resolved = resolve_path(path, pbuf, sizeof(pbuf));
    FILE *fp = fopen(resolved, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        return NULL;
    }
    void *buf = malloc((size_t)sz);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) {
        free(buf);
        return NULL;
    }
    if (out_sz) *out_sz = (size_t)sz;
    return buf;
}

static Fh3TeCfg get_tiny_te_cfg(void) {
    Fh3TeCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.hidden_size = 128;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 32;
    cfg.num_layers = 3;  /* 3 layers out of 4 in fixture */
    cfg.intermediate_size = 256;
    cfg.norm_eps = 1e-6f;
    cfg.qk_norm_eps = 1e-6f;
    cfg.rope_theta = 5000000.0f;
    return cfg;
}

/* -------------------------------------------------------------------------
 * Test 1: test_te_tiny_matches_transformers
 * ------------------------------------------------------------------------- */

static int test_te_tiny_matches_transformers(void) {
    printf("--- Running test_te_tiny_matches_transformers ---\n");

    char pbuf_trunk[512], pbuf_embed[512];
    const char *trunk_path = resolve_path("tests/fixtures/tiny_te/te_trunk.bin", pbuf_trunk, sizeof(pbuf_trunk));
    const char *embed_path = resolve_path("tests/fixtures/tiny_te/te_embed.bin", pbuf_embed, sizeof(pbuf_embed));

    Fh3TeCfg cfg = get_tiny_te_cfg();
    Fh3Te *te = NULL;
    int rc = fh3_te_open(trunk_path, embed_path, &cfg, &te);
    TEST_ASSERT(rc == FH3_OK && te != NULL);

    size_t ids_sz = 0, golden_sz = 0;
    int *ids = (int *)read_file_bytes("tests/fixtures/tiny_te/prompt_ids.bin", &ids_sz);
    float *golden_prenorm = (float *)read_file_bytes("tests/fixtures/tiny_te/golden_prenorm.bin", &golden_sz);
    TEST_ASSERT(ids != NULL && golden_prenorm != NULL);
    int T = (int)(ids_sz / sizeof(int));
    TEST_ASSERT(T == 5);
    TEST_ASSERT(golden_sz == (size_t)T * (size_t)cfg.hidden_size * sizeof(float));

    /* A. Encode using mapped trunk */
    float out_mapped[5 * 128];
    rc = fh3_te_encode(te, NULL, ids, T, out_mapped);
    TEST_ASSERT(rc == FH3_OK);

    float rel_l2_mapped = compute_rel_l2(out_mapped, golden_prenorm, (size_t)T * 128);
    printf("  Mapped trunk rel-L2 vs golden: %e\n", rel_l2_mapped);
    TEST_ASSERT(rel_l2_mapped <= 1e-4f);

    /* B. Encode using streamed Fh3Trunk */
    Fh3Trunk tr;
    rc = fh3_trunk_open(trunk_path, 20 * 1024 * 1024ULL, &tr);
    TEST_ASSERT(rc == FH3_OK);

    float out_streamed[5 * 128];
    rc = fh3_te_encode(te, &tr, ids, T, out_streamed);
    TEST_ASSERT(rc == FH3_OK);

    float rel_l2_streamed = compute_rel_l2(out_streamed, golden_prenorm, (size_t)T * 128);
    printf("  Streamed trunk rel-L2 vs golden: %e\n", rel_l2_streamed);
    TEST_ASSERT(rel_l2_streamed <= 1e-4f);

    float diff_mapped_vs_streamed = compute_rel_l2(out_mapped, out_streamed, (size_t)T * 128);
    TEST_ASSERT(diff_mapped_vs_streamed <= 1e-6f);

    fh3_trunk_close(&tr);
    fh3_te_close(te);
    free(ids);
    free(golden_prenorm);

    printf("[PASS] test_te_tiny_matches_transformers\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 2: test_te_stop_layer_is_prenorm
 * ------------------------------------------------------------------------- */

static int test_te_stop_layer_is_prenorm(void) {
    printf("--- Running test_te_stop_layer_is_prenorm ---\n");

    char pbuf_trunk[512], pbuf_embed[512];
    const char *trunk_path = resolve_path("tests/fixtures/tiny_te/te_trunk.bin", pbuf_trunk, sizeof(pbuf_trunk));
    const char *embed_path = resolve_path("tests/fixtures/tiny_te/te_embed.bin", pbuf_embed, sizeof(pbuf_embed));

    Fh3TeCfg cfg = get_tiny_te_cfg();
    Fh3Te *te = NULL;
    int rc = fh3_te_open(trunk_path, embed_path, &cfg, &te);
    TEST_ASSERT(rc == FH3_OK && te != NULL);

    size_t ids_sz = 0, golden_normed_sz = 0;
    int *ids = (int *)read_file_bytes("tests/fixtures/tiny_te/prompt_ids.bin", &ids_sz);
    float *golden_normed = (float *)read_file_bytes("tests/fixtures/tiny_te/golden_normed.bin", &golden_normed_sz);
    TEST_ASSERT(ids != NULL && golden_normed != NULL);
    int T = (int)(ids_sz / sizeof(int));

    float out[5 * 128];
    rc = fh3_te_encode(te, NULL, ids, T, out);
    TEST_ASSERT(rc == FH3_OK);

    /* Verify that pre-norm output does NOT equal normed output */
    float rel_l2_normed = compute_rel_l2(out, golden_normed, (size_t)T * 128);
    printf("  rel-L2(out, golden_normed): %e (must be > 1e-3 to confirm pre-norm)\n", rel_l2_normed);
    TEST_ASSERT(rel_l2_normed > 1e-3f);

    /* Also verify directly: norm(out) != out */
    float out_renormed[5 * 128];
    memcpy(out_renormed, out, sizeof(out_renormed));
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    fh3_rmsnorm(out_renormed, ones, T, 128, 1e-6f);

    float rel_l2_self_norm = compute_rel_l2(out, out_renormed, (size_t)T * 128);
    printf("  rel-L2(out, rmsnorm(out)): %e (must be > 1e-3)\n", rel_l2_self_norm);
    TEST_ASSERT(rel_l2_self_norm > 1e-3f);

    fh3_te_close(te);
    free(ids);
    free(golden_normed);

    printf("[PASS] test_te_stop_layer_is_prenorm\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: test_te_mrope_text_eq_1d
 * ------------------------------------------------------------------------- */

static int test_te_mrope_text_eq_1d(void) {
    printf("--- Running test_te_mrope_text_eq_1d ---\n");

    int T = 8;
    int head_dim = 32;
    int half_dim = head_dim / 2;
    float theta = 5000000.0f;

    /* 1-D RoPE table */
    float cos_1d[8 * 32];
    float sin_1d[8 * 32];
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < half_dim; j++) {
            float freq = 1.0f / powf(theta, (float)(2 * j) / (float)head_dim);
            float val = (float)t * freq;
            float c = cosf(val);
            float s = sinf(val);
            cos_1d[t * head_dim + j] = c;
            cos_1d[t * head_dim + half_dim + j] = c;
            sin_1d[t * head_dim + j] = s;
            sin_1d[t * head_dim + half_dim + j] = s;
        }
    }

    /* 3-axis text M-RoPE where (pos_t, pos_h, pos_w) = (t, t, t) */
    int mrope_section[3] = { 6, 5, 5 };  /* 6 + 5 + 5 = 16 = half_dim */
    float cos_3d[8 * 32];
    float sin_3d[8 * 32];

    for (int t = 0; t < T; t++) {
        float freqs_3[3][16];
        for (int axis = 0; axis < 3; axis++) {
            for (int j = 0; j < half_dim; j++) {
                float freq = 1.0f / powf(theta, (float)(2 * j) / (float)head_dim);
                freqs_3[axis][j] = (float)t * freq;
            }
        }

        float freqs_thw[16];
        memcpy(freqs_thw, freqs_3[0], sizeof(freqs_thw));
        for (int dim = 1; dim <= 2; dim++) {
            int offset = dim;
            int length = mrope_section[dim] * 3;
            for (int idx = offset; idx < length && idx < half_dim; idx += 3) {
                freqs_thw[idx] = freqs_3[dim][idx];
            }
        }

        for (int j = 0; j < half_dim; j++) {
            float c = cosf(freqs_thw[j]);
            float s = sinf(freqs_thw[j]);
            cos_3d[t * head_dim + j] = c;
            cos_3d[t * head_dim + half_dim + j] = c;
            sin_3d[t * head_dim + j] = s;
            sin_3d[t * head_dim + half_dim + j] = s;
        }
    }

    float diff_cos = 0.0f;
    float diff_sin = 0.0f;
    for (int i = 0; i < T * head_dim; i++) {
        float dc = fabsf(cos_1d[i] - cos_3d[i]);
        float ds = fabsf(sin_1d[i] - sin_3d[i]);
        if (dc > diff_cos) diff_cos = dc;
        if (ds > diff_sin) diff_sin = ds;
    }

    printf("  max abs diff cos: %e, sin: %e\n", diff_cos, diff_sin);
    TEST_ASSERT(diff_cos <= 1e-6f);
    TEST_ASSERT(diff_sin <= 1e-6f);

    printf("[PASS] test_te_mrope_text_eq_1d\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 4: test_te_prompt_cache
 * ------------------------------------------------------------------------- */

static int test_te_prompt_cache(void) {
    printf("--- Running test_te_prompt_cache ---\n");

    const char *cache_dir = "tests/fixtures/tiny_te/cache_test";
    const char *test_cache_file = "tests/fixtures/tiny_te/cache_test/b90477f4f6666b7fd2618b769ea78f0653138e5a1d0c84b666f78557db2b4685.emb";
    remove(test_cache_file);

    int ids[4] = { 101, 202, 303, 404 };
    int T = 4;
    int hidden = 128;

    /* A. Lookup on non-existent cache returns error / not found */
    float lookup_buf[4 * 128];
    int rc = fh3_te_cache_lookup(cache_dir, ids, T, lookup_buf);
    TEST_ASSERT(rc != FH3_OK);

    /* B. Save embeddings to cache */
    float fake_emb[4 * 128];
    for (int i = 0; i < 4 * 128; i++) {
        fake_emb[i] = (float)sin((double)i * 0.1);
    }

    rc = fh3_te_cache_save(cache_dir, ids, T, fake_emb);
    TEST_ASSERT(rc == FH3_OK);

    /* C. Lookup on existing cache succeeds and returns bit-for-bit identical floats */
    memset(lookup_buf, 0, sizeof(lookup_buf));
    rc = fh3_te_cache_lookup(cache_dir, ids, T, lookup_buf);
    TEST_ASSERT(rc == FH3_OK);

    TEST_ASSERT(memcmp(lookup_buf, fake_emb, (size_t)T * (size_t)hidden * sizeof(float)) == 0);

    /* D. Different token IDs miss cache */
    int diff_ids[4] = { 101, 202, 303, 999 };
    rc = fh3_te_cache_lookup(cache_dir, diff_ids, T, lookup_buf);
    TEST_ASSERT(rc != FH3_OK);

    /* Clean up created file */
    remove(test_cache_file);

    printf("[PASS] test_te_prompt_cache\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

int main(void) {
    printf("=========================================\n");
    printf("Running Truncated Text Encoder Unit Tests\n");
    printf("=========================================\n");

    if (test_te_tiny_matches_transformers() != 0) return 1;
    if (test_te_stop_layer_is_prenorm() != 0) return 1;
    if (test_te_mrope_text_eq_1d() != 0) return 1;
    if (test_te_prompt_cache() != 0) return 1;

    printf("=========================================\n");
    printf("All Truncated Text Encoder Tests Passed!\n");
    printf("=========================================\n");
    return g_failures;
}
