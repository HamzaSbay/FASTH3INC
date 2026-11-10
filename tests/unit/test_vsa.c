#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"
#include "model/fh3_seq.h"
#include "core/fh3_vsa.h"

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

typedef struct {
    uint32_t n;
    uint32_t num_text;
    uint32_t num_audio;
    uint32_t num_video;
    uint32_t grid_t;
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t heads;
    uint32_t head_dim;
    float sparsity;
    uint32_t has_gate;
} VsaFixtureHeader;

typedef struct {
    VsaFixtureHeader hdr;
    float *q;
    float *k;
    float *v;
    float *gate;
    float *ref;
} VsaFixture;

static int load_fixture(const char *rel_path, VsaFixture *fix) {
    char pbuf[512];
    const char *path = resolve_path(rel_path, pbuf, sizeof(pbuf));
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open fixture: %s\n", path);
        return -1;
    }

    if (fread(&fix->hdr, sizeof(VsaFixtureHeader), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    size_t total_elements = (size_t)fix->hdr.n * (size_t)fix->hdr.heads * (size_t)fix->hdr.head_dim;
    size_t tensor_bytes = total_elements * sizeof(float);

    fix->q = (float *)malloc(tensor_bytes);
    fix->k = (float *)malloc(tensor_bytes);
    fix->v = (float *)malloc(tensor_bytes);
    fix->gate = fix->hdr.has_gate ? (float *)malloc(tensor_bytes) : NULL;
    fix->ref = (float *)malloc(tensor_bytes);

    if (!fix->q || !fix->k || !fix->v || (fix->hdr.has_gate && !fix->gate) || !fix->ref) {
        if (fix->q) free(fix->q);
        if (fix->k) free(fix->k);
        if (fix->v) free(fix->v);
        if (fix->gate) free(fix->gate);
        if (fix->ref) free(fix->ref);
        fclose(fp);
        return -1;
    }

    if (fread(fix->q, 1, tensor_bytes, fp) != tensor_bytes ||
        fread(fix->k, 1, tensor_bytes, fp) != tensor_bytes ||
        fread(fix->v, 1, tensor_bytes, fp) != tensor_bytes) {
        fclose(fp);
        return -1;
    }

    if (fix->hdr.has_gate) {
        if (fread(fix->gate, 1, tensor_bytes, fp) != tensor_bytes) {
            fclose(fp);
            return -1;
        }
    }

    if (fread(fix->ref, 1, tensor_bytes, fp) != tensor_bytes) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

static void free_fixture(VsaFixture *fix) {
    if (fix->q) free(fix->q);
    if (fix->k) free(fix->k);
    if (fix->v) free(fix->v);
    if (fix->gate) free(fix->gate);
    if (fix->ref) free(fix->ref);
    memset(fix, 0, sizeof(*fix));
}

/* -------------------------------------------------------------------------
 * Test 1: vsa_matches_ref_n784 (tiny-preset grid 7x8x14, N=784)
 * ------------------------------------------------------------------------- */
static int test_vsa_matches_ref_n784(void) {
    printf("--- Running test_vsa_matches_ref_n784 ---\n");
    VsaFixture fix;
    int rc = load_fixture("tests/fixtures/vsa/vsa_n784.bin", &fix);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(fix.hdr.n == 784);
    TEST_ASSERT(fix.hdr.grid_t == 7 && fix.hdr.grid_h == 8 && fix.hdr.grid_w == 14);

    Fh3Seq seq;
    memset(&seq, 0, sizeof(seq));
    seq.seq_len = (int)fix.hdr.n;
    seq.num_text = (int)fix.hdr.num_text;
    seq.num_audio = (int)fix.hdr.num_audio;
    seq.num_video = (int)fix.hdr.num_video;
    seq.grid_t = (int)fix.hdr.grid_t;
    seq.grid_h = (int)fix.hdr.grid_h;
    seq.grid_w = (int)fix.hdr.grid_w;

    Fh3VsaCfg cfg;
    cfg.tile = 64;
    cfg.sparsity = fix.hdr.sparsity;

    size_t total_elements = (size_t)fix.hdr.n * (size_t)fix.hdr.heads * (size_t)fix.hdr.head_dim;
    float *out = (float *)malloc(total_elements * sizeof(float));
    TEST_ASSERT(out != NULL);

    fh3_vsa_attention(&cfg, fix.q, fix.k, fix.v, fix.gate, out, &seq, (int)fix.hdr.heads, (int)fix.hdr.head_dim);

    float rel_l2 = compute_rel_l2(out, fix.ref, total_elements);
    printf("  vsa_n784 rel-L2 vs ref: %e\n", rel_l2);
    TEST_ASSERT(rel_l2 <= 1e-5f);

    free(out);
    free_fixture(&fix);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 2: vsa_matches_ref_padded (grid length not a multiple of 64)
 * ------------------------------------------------------------------------- */
static int test_vsa_matches_ref_padded(void) {
    printf("--- Running test_vsa_matches_ref_padded ---\n");
    VsaFixture fix;
    int rc = load_fixture("tests/fixtures/vsa/vsa_padded.bin", &fix);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(fix.hdr.n == 268);
    TEST_ASSERT(fix.hdr.num_text == 35 && fix.hdr.num_audio == 23 && fix.hdr.num_video == 210);

    Fh3Seq seq;
    memset(&seq, 0, sizeof(seq));
    seq.seq_len = (int)fix.hdr.n;
    seq.num_text = (int)fix.hdr.num_text;
    seq.num_audio = (int)fix.hdr.num_audio;
    seq.num_video = (int)fix.hdr.num_video;
    seq.grid_t = (int)fix.hdr.grid_t;
    seq.grid_h = (int)fix.hdr.grid_h;
    seq.grid_w = (int)fix.hdr.grid_w;

    Fh3VsaCfg cfg;
    cfg.tile = 64;
    cfg.sparsity = fix.hdr.sparsity;

    size_t total_elements = (size_t)fix.hdr.n * (size_t)fix.hdr.heads * (size_t)fix.hdr.head_dim;
    float *out = (float *)malloc(total_elements * sizeof(float));
    TEST_ASSERT(out != NULL);

    fh3_vsa_attention(&cfg, fix.q, fix.k, fix.v, fix.gate, out, &seq, (int)fix.hdr.heads, (int)fix.hdr.head_dim);

    float rel_l2 = compute_rel_l2(out, fix.ref, total_elements);
    printf("  vsa_padded rel-L2 vs ref: %e\n", rel_l2);
    TEST_ASSERT(rel_l2 <= 1e-5f);

    free(out);
    free_fixture(&fix);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: vsa_sparsity0_eq_dense_plus_coarse (structural check)
 * ------------------------------------------------------------------------- */
static int test_vsa_sparsity0_eq_dense_plus_coarse(void) {
    printf("--- Running test_vsa_sparsity0_eq_dense_plus_coarse ---\n");
    VsaFixture fix;
    int rc = load_fixture("tests/fixtures/vsa/vsa_sparsity0.bin", &fix);
    TEST_ASSERT(rc == 0);

    Fh3Seq seq;
    memset(&seq, 0, sizeof(seq));
    seq.seq_len = (int)fix.hdr.n;
    seq.num_text = (int)fix.hdr.num_text;
    seq.num_audio = (int)fix.hdr.num_audio;
    seq.num_video = (int)fix.hdr.num_video;
    seq.grid_t = (int)fix.hdr.grid_t;
    seq.grid_h = (int)fix.hdr.grid_h;
    seq.grid_w = (int)fix.hdr.grid_w;

    Fh3VsaCfg cfg;
    cfg.tile = 64;
    cfg.sparsity = 0.0f;

    size_t total_elements = (size_t)fix.hdr.n * (size_t)fix.hdr.heads * (size_t)fix.hdr.head_dim;
    float *out_vsa_gated = (float *)malloc(total_elements * sizeof(float));
    float *out_vsa_fine  = (float *)malloc(total_elements * sizeof(float));
    float *out_dense     = (float *)malloc(total_elements * sizeof(float));
    TEST_ASSERT(out_vsa_gated && out_vsa_fine && out_dense);

    /* 1. VSA with sparsity 0.0 and gate must match Python reference */
    fh3_vsa_attention(&cfg, fix.q, fix.k, fix.v, fix.gate, out_vsa_gated, &seq, (int)fix.hdr.heads, (int)fix.hdr.head_dim);
    float rel_l2_gated = compute_rel_l2(out_vsa_gated, fix.ref, total_elements);
    printf("  vsa_sparsity0 rel-L2 vs ref: %e\n", rel_l2_gated);
    TEST_ASSERT(rel_l2_gated <= 1e-5f);

    /* 2. VSA with sparsity 0.0 and NO gate must equal dense attention */
    fh3_vsa_attention(&cfg, fix.q, fix.k, fix.v, NULL, out_vsa_fine, &seq, (int)fix.hdr.heads, (int)fix.hdr.head_dim);
    fh3_attention(fix.q, fix.k, fix.v, out_dense, (int)fix.hdr.n, (int)fix.hdr.heads, (int)fix.hdr.head_dim);

    float rel_l2_dense = compute_rel_l2(out_vsa_fine, out_dense, total_elements);
    printf("  vsa_sparsity0 (no gate) vs dense rel-L2: %e\n", rel_l2_dense);
    TEST_ASSERT(rel_l2_dense <= 1e-5f);

    free(out_vsa_gated);
    free(out_vsa_fine);
    free(out_dense);
    free_fixture(&fix);
    return 0;
}

int main(void) {
    printf("=== Starting test_vsa suite ===\n");
    test_vsa_matches_ref_n784();
    test_vsa_matches_ref_padded();
    test_vsa_sparsity0_eq_dense_plus_coarse();

    if (g_failures > 0) {
        fprintf(stderr, "=== test_vsa FAILED with %d failure(s) ===\n", g_failures);
        return 1;
    }
    printf("=== All test_vsa tests PASSED ===\n");
    return 0;
}
