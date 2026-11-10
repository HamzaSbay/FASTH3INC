#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"

static int g_failures = 0;

static const char *resolve_fixture(const char *rel_path, char *buf, size_t sz) {
    FILE *fp = fopen(rel_path, "rb");
    if (fp) {
        fclose(fp);
        return rel_path;
    }
    snprintf(buf, sz, "../%s", rel_path);
    fp = fopen(buf, "rb");
    if (fp) {
        fclose(fp);
        return buf;
    }
    snprintf(buf, sz, "../../%s", rel_path);
    fp = fopen(buf, "rb");
    if (fp) {
        fclose(fp);
        return buf;
    }
    return rel_path;
}

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return 1; \
    } \
} while (0)

static int test_matmul_suite(void) {
    printf("--- Running test_matmul_suite ---\n");

    char path_buf[512];
    const char *resolved = resolve_fixture("tests/fixtures/quant/matmul_ref.bin", path_buf, sizeof(path_buf));
    FILE *fp = fopen(resolved, "rb");
    if (!fp) {
        fprintf(stderr, "Could not open fixture file: %s\n", resolved);
        TEST_ASSERT(fp != NULL);
    }

    uint32_t header[5]; // type, in_dim, out_dim, n_tok, w_bytes
    size_t nr = fread(header, sizeof(uint32_t), 5, fp);
    TEST_ASSERT(nr == 5);

    int qtype = (int)header[0];
    int64_t in_dim = (int64_t)header[1];
    int64_t out_dim = (int64_t)header[2];
    int n_tok = (int)header[3];
    uint32_t w_bytes = header[4];

    TEST_ASSERT(qtype == FH3_QUANT_Q4_K);
    TEST_ASSERT(in_dim == 5376);
    TEST_ASSERT(out_dim == 192);
    TEST_ASSERT(n_tok == 37);

    uint8_t *w_raw = (uint8_t *)malloc(w_bytes);
    TEST_ASSERT(w_raw != NULL);
    nr = fread(w_raw, 1, w_bytes, fp);
    TEST_ASSERT(nr == w_bytes);

    size_t x_count = (size_t)n_tok * in_dim;
    float *x = (float *)malloc(x_count * sizeof(float));
    TEST_ASSERT(x != NULL);
    nr = fread(x, sizeof(float), x_count, fp);
    TEST_ASSERT(nr == x_count);

    float *bias = (float *)malloc(out_dim * sizeof(float));
    TEST_ASSERT(bias != NULL);
    nr = fread(bias, sizeof(float), out_dim, fp);
    TEST_ASSERT(nr == (size_t)out_dim);

    size_t y_count = (size_t)n_tok * out_dim;
    float *y_ref = (float *)malloc(y_count * sizeof(float));
    TEST_ASSERT(y_ref != NULL);
    nr = fread(y_ref, sizeof(float), y_count, fp);
    TEST_ASSERT(nr == y_count);

    fclose(fp);

    Fh3W w;
    w.type = qtype;
    w.in = in_dim;
    w.out = out_dim;
    w.data = w_raw;

    float *y_scalar = (float *)malloc(y_count * sizeof(float));
    float *y_avx2   = (float *)malloc(y_count * sizeof(float));
    float *y_omp    = (float *)malloc(y_count * sizeof(float));
    TEST_ASSERT(y_scalar && y_avx2 && y_omp);

    /* 1. Compute with scalar, avx2, and omp (with bias) */
    fh3_matmul_scalar(&w, x, y_scalar, n_tok, bias);
    fh3_matmul_avx2(&w, x, y_avx2, n_tok, bias);
    fh3_matmul(&w, x, y_omp, n_tok, bias);

    /* 2. Check bitwise equality: matmul_scalar_eq_avx2_eq_omp */
    int diff_avx2 = 0;
    int diff_omp  = 0;
    for (size_t i = 0; i < y_count; i++) {
        uint32_t b_s, b_a, b_o;
        memcpy(&b_s, &y_scalar[i], sizeof(uint32_t));
        memcpy(&b_a, &y_avx2[i], sizeof(uint32_t));
        memcpy(&b_o, &y_omp[i], sizeof(uint32_t));

        if (b_s != b_a) {
            if (diff_avx2 < 5) {
                fprintf(stderr, "Scalar vs AVX2 diff at %zu: s=%f (0x%08x), a=%f (0x%08x)\n",
                        i, y_scalar[i], b_s, y_avx2[i], b_a);
            }
            diff_avx2++;
        }
        if (b_s != b_o) {
            if (diff_omp < 5) {
                fprintf(stderr, "Scalar vs OMP diff at %zu: s=%f (0x%08x), o=%f (0x%08x)\n",
                        i, y_scalar[i], b_s, y_omp[i], b_o);
            }
            diff_omp++;
        }
    }
    TEST_ASSERT(diff_avx2 == 0);
    TEST_ASSERT(diff_omp == 0);
    printf("matmul_scalar_eq_avx2_eq_omp PASS (all %zu outputs bitwise identical)\n", y_count);

    /* 3. Check bitwise equality without bias */
    fh3_matmul_scalar(&w, x, y_scalar, n_tok, NULL);
    fh3_matmul_avx2(&w, x, y_avx2, n_tok, NULL);
    fh3_matmul(&w, x, y_omp, n_tok, NULL);
    TEST_ASSERT(memcmp(y_scalar, y_avx2, y_count * sizeof(float)) == 0);
    TEST_ASSERT(memcmp(y_scalar, y_omp, y_count * sizeof(float)) == 0);
    printf("matmul_scalar_eq_avx2_eq_omp (nobias) PASS\n");

    /* 4. Compare with numpy float64 reference: matmul_matches_numpy */
    // Re-run with bias for reference comparison
    fh3_matmul(&w, x, y_omp, n_tok, bias);
    double sum_diff_sq = 0.0;
    double sum_ref_sq = 0.0;
    for (size_t i = 0; i < y_count; i++) {
        double diff = (double)y_omp[i] - (double)y_ref[i];
        sum_diff_sq += diff * diff;
        sum_ref_sq += (double)y_ref[i] * (double)y_ref[i];
    }
    double rel_l2 = sqrt(sum_diff_sq) / sqrt(sum_ref_sq);
    printf("matmul_matches_numpy: rel_l2 = %e (tolerance <= 1e-6)\n", rel_l2);
    TEST_ASSERT(rel_l2 <= 1e-6);
    printf("matmul_matches_numpy PASS\n");

    free(w_raw);
    free(x);
    free(bias);
    free(y_ref);
    free(y_scalar);
    free(y_avx2);
    free(y_omp);

    return 0;
}

int main(void) {
    if (test_matmul_suite() != 0) {
        fprintf(stderr, "test_matmul failed\n");
        return 1;
    }
    printf("ALL MATMUL TESTS PASSED!\n");
    return 0;
}
