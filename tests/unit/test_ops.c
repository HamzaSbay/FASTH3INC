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

static bool check_close(const float *actual, const float *ref, size_t count, float tol, float *max_err_out) {
    float max_err = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(actual[i] - ref[i]);
        float ref_val = fabsf(ref[i]);
        float denom = (ref_val > 1.0f) ? ref_val : 1.0f;
        float err = diff / denom;
        if (err > max_err) max_err = err;
        if (err > tol) {
            if (max_err_out) *max_err_out = max_err;
            return false;
        }
    }
    if (max_err_out) *max_err_out = max_err;
    return true;
}

static int test_rmsnorm(void) {
    printf("--- test_rmsnorm ---\n");
    char path_buf[512];
    const char *resolved = resolve_fixture("tests/fixtures/ops/rmsnorm.bin", path_buf, sizeof(path_buf));
    FILE *fp = fopen(resolved, "rb");
    TEST_ASSERT(fp != NULL);

    uint32_t header[2];
    float eps = 0.0f;
    TEST_ASSERT(fread(&header[0], sizeof(uint32_t), 2, fp) == 2);
    TEST_ASSERT(fread(&eps, sizeof(float), 1, fp) == 1);

    int n = (int)header[0];
    int dim = (int)header[1];
    TEST_ASSERT(n == 8 && dim == 5376);

    size_t x_count = (size_t)n * dim;
    float *x = (float *)malloc(x_count * sizeof(float));
    float *w = (float *)malloc(dim * sizeof(float));
    float *ref = (float *)malloc(x_count * sizeof(float));
    TEST_ASSERT(x && w && ref);

    TEST_ASSERT(fread(x, sizeof(float), x_count, fp) == x_count);
    TEST_ASSERT(fread(w, sizeof(float), dim, fp) == (size_t)dim);
    TEST_ASSERT(fread(ref, sizeof(float), x_count, fp) == x_count);
    fclose(fp);

    fh3_rmsnorm(x, w, n, dim, eps);

    float max_err = 0.0f;
    bool ok = check_close(x, ref, x_count, 1e-5f, &max_err);
    printf("  RMSNorm max relative error: %e\n", max_err);

    free(x);
    free(w);
    free(ref);
    TEST_ASSERT(ok);
    return 0;
}

static int test_rope3(void) {
    printf("--- test_rope3 ---\n");
    char path_buf[512];
    const char *resolved = resolve_fixture("tests/fixtures/ops/rope3.bin", path_buf, sizeof(path_buf));
    FILE *fp = fopen(resolved, "rb");
    TEST_ASSERT(fp != NULL);

    uint32_t header[4];
    TEST_ASSERT(fread(header, sizeof(uint32_t), 4, fp) == 4);

    int n = (int)header[0];
    int heads = (int)header[1];
    int head_dim = (int)header[2];
    int rot_dim = (int)header[3];
    TEST_ASSERT(n == 16 && heads == 4 && head_dim == 128 && rot_dim == 96);

    size_t x_count = (size_t)n * heads * head_dim;
    size_t rot_count = (size_t)n * rot_dim;

    float *x = (float *)malloc(x_count * sizeof(float));
    float *x_orig = (float *)malloc(x_count * sizeof(float));
    float *cos = (float *)malloc(rot_count * sizeof(float));
    float *sin = (float *)malloc(rot_count * sizeof(float));
    float *ref = (float *)malloc(x_count * sizeof(float));
    TEST_ASSERT(x && x_orig && cos && sin && ref);

    TEST_ASSERT(fread(x, sizeof(float), x_count, fp) == x_count);
    memcpy(x_orig, x, x_count * sizeof(float));
    TEST_ASSERT(fread(cos, sizeof(float), rot_count, fp) == rot_count);
    TEST_ASSERT(fread(sin, sizeof(float), rot_count, fp) == rot_count);
    TEST_ASSERT(fread(ref, sizeof(float), x_count, fp) == x_count);
    fclose(fp);

    fh3_rope3(x, cos, sin, n, heads, head_dim, rot_dim);

    float max_err = 0.0f;
    bool ok = check_close(x, ref, x_count, 1e-5f, &max_err);
    printf("  RoPE3 (rot_dim=96) max relative error: %e\n", max_err);

    /* Invariant check: rot_dim = 128 must FAIL the 96-dim fixture */
    size_t full_rot_count = (size_t)n * head_dim;
    float *cos128 = (float *)calloc(full_rot_count, sizeof(float));
    float *sin128 = (float *)calloc(full_rot_count, sizeof(float));
    TEST_ASSERT(cos128 && sin128);
    for (int i = 0; i < n; i++) {
        memcpy(cos128 + i * head_dim, cos + i * rot_dim, rot_dim * sizeof(float));
        memcpy(sin128 + i * head_dim, sin + i * rot_dim, rot_dim * sizeof(float));
        for (int d = rot_dim; d < head_dim; d++) {
            cos128[i * head_dim + d] = 1.0f;
            sin128[i * head_dim + d] = 0.0f;
        }
    }
    float *x_bug = (float *)malloc(x_count * sizeof(float));
    TEST_ASSERT(x_bug);
    memcpy(x_bug, x_orig, x_count * sizeof(float));
    fh3_rope3(x_bug, cos128, sin128, n, heads, head_dim, head_dim);

    float bug_diff = 0.0f;
    for (size_t i = 0; i < x_count; i++) {
        float d = fabsf(x_bug[i] - ref[i]);
        if (d > bug_diff) bug_diff = d;
    }
    printf("  RoPE3 bug check (rot_dim=128 vs 96-ref) max diff: %f\n", bug_diff);

    free(x);
    free(x_orig);
    free(cos);
    free(sin);
    free(ref);
    free(cos128);
    free(sin128);
    free(x_bug);

    TEST_ASSERT(ok);
    TEST_ASSERT(bug_diff > 1.0f);
    return 0;
}

static int test_swiglu(void) {
    printf("--- test_swiglu ---\n");
    char path_buf[512];
    const char *resolved = resolve_fixture("tests/fixtures/ops/swiglu.bin", path_buf, sizeof(path_buf));
    FILE *fp = fopen(resolved, "rb");
    TEST_ASSERT(fp != NULL);

    uint32_t header[2];
    TEST_ASSERT(fread(header, sizeof(uint32_t), 2, fp) == 2);

    int n = (int)header[0];
    int F = (int)header[1];
    TEST_ASSERT(n == 8 && F == 1536);

    size_t in_count = (size_t)n * 2 * F;
    size_t out_count = (size_t)n * F;

    float *in = (float *)malloc(in_count * sizeof(float));
    float *out = (float *)malloc(out_count * sizeof(float));
    float *ref = (float *)malloc(out_count * sizeof(float));
    TEST_ASSERT(in && out && ref);

    TEST_ASSERT(fread(in, sizeof(float), in_count, fp) == in_count);
    TEST_ASSERT(fread(ref, sizeof(float), out_count, fp) == out_count);
    fclose(fp);

    fh3_swiglu(in, out, n, F);

    float max_err = 0.0f;
    bool ok = check_close(out, ref, out_count, 1e-5f, &max_err);
    printf("  SwiGLU max relative error: %e\n", max_err);

    /* Invariant check: swapped halves must fail */
    float *in_swapped = (float *)malloc(in_count * sizeof(float));
    float *out_swapped = (float *)malloc(out_count * sizeof(float));
    TEST_ASSERT(in_swapped && out_swapped);
    for (int i = 0; i < n; i++) {
        memcpy(in_swapped + i * 2 * F, in + i * 2 * F + F, F * sizeof(float));
        memcpy(in_swapped + i * 2 * F + F, in + i * 2 * F, F * sizeof(float));
    }
    fh3_swiglu(in_swapped, out_swapped, n, F);

    float bug_diff = 0.0f;
    for (size_t i = 0; i < out_count; i++) {
        float d = fabsf(out_swapped[i] - ref[i]);
        if (d > bug_diff) bug_diff = d;
    }
    printf("  SwiGLU bug check (swapped halves) max diff: %f\n", bug_diff);

    free(in);
    free(out);
    free(ref);
    free(in_swapped);
    free(out_swapped);

    TEST_ASSERT(ok);
    TEST_ASSERT(bug_diff > 1.0f);
    return 0;
}

static int test_modulate(void) {
    printf("--- test_modulate ---\n");
    char path_buf[512];
    const char *resolved = resolve_fixture("tests/fixtures/ops/modulate.bin", path_buf, sizeof(path_buf));
    FILE *fp = fopen(resolved, "rb");
    TEST_ASSERT(fp != NULL);

    uint32_t header[2];
    TEST_ASSERT(fread(header, sizeof(uint32_t), 2, fp) == 2);

    int n = (int)header[0];
    int dim = (int)header[1];
    TEST_ASSERT(n == 8 && dim == 5376);

    size_t count = (size_t)n * dim;
    float *x = (float *)malloc(count * sizeof(float));
    float *x_orig = (float *)malloc(count * sizeof(float));
    float *shift = (float *)malloc(count * sizeof(float));
    float *scale = (float *)malloc(count * sizeof(float));
    float *ref = (float *)malloc(count * sizeof(float));
    TEST_ASSERT(x && x_orig && shift && scale && ref);

    TEST_ASSERT(fread(x, sizeof(float), count, fp) == count);
    memcpy(x_orig, x, count * sizeof(float));
    TEST_ASSERT(fread(shift, sizeof(float), count, fp) == count);
    TEST_ASSERT(fread(scale, sizeof(float), count, fp) == count);
    TEST_ASSERT(fread(ref, sizeof(float), count, fp) == count);
    fclose(fp);

    fh3_modulate(x, shift, scale, n, dim);

    float max_err = 0.0f;
    bool ok = check_close(x, ref, count, 1e-5f, &max_err);
    printf("  Modulate max relative error: %e\n", max_err);

    /* Invariant check: x * scale + shift must FAIL */
    float bug_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float bug_val = x_orig[i] * scale[i] + shift[i];
        float d = fabsf(bug_val - ref[i]);
        if (d > bug_diff) bug_diff = d;
    }
    printf("  Modulate bug check (x * scale + shift) max diff: %f\n", bug_diff);

    free(x);
    free(x_orig);
    free(shift);
    free(scale);
    free(ref);

    TEST_ASSERT(ok);
    TEST_ASSERT(bug_diff > 1.0f);
    return 0;
}

static int test_gate_residual(void) {
    printf("--- test_gate_residual ---\n");
    char path_buf[512];
    const char *resolved = resolve_fixture("tests/fixtures/ops/gate_residual.bin", path_buf, sizeof(path_buf));
    FILE *fp = fopen(resolved, "rb");
    TEST_ASSERT(fp != NULL);

    uint32_t header[2];
    TEST_ASSERT(fread(header, sizeof(uint32_t), 2, fp) == 2);

    int n = (int)header[0];
    int dim = (int)header[1];
    TEST_ASSERT(n == 8 && dim == 5376);

    size_t count = (size_t)n * dim;
    float *h = (float *)malloc(count * sizeof(float));
    float *gate = (float *)malloc(count * sizeof(float));
    float *y = (float *)malloc(count * sizeof(float));
    float *ref = (float *)malloc(count * sizeof(float));
    TEST_ASSERT(h && gate && y && ref);

    TEST_ASSERT(fread(h, sizeof(float), count, fp) == count);
    TEST_ASSERT(fread(gate, sizeof(float), count, fp) == count);
    TEST_ASSERT(fread(y, sizeof(float), count, fp) == count);
    TEST_ASSERT(fread(ref, sizeof(float), count, fp) == count);
    fclose(fp);

    fh3_gate_residual(h, gate, y, n, dim);

    float max_err = 0.0f;
    bool ok = check_close(h, ref, count, 1e-5f, &max_err);
    printf("  GateResidual max relative error: %e\n", max_err);

    free(h);
    free(gate);
    free(y);
    free(ref);

    TEST_ASSERT(ok);
    return 0;
}

static int test_attention(void) {
    printf("--- test_attention (N=777) ---\n");
    char path_buf[512];
    const char *resolved = resolve_fixture("tests/fixtures/ops/attention.bin", path_buf, sizeof(path_buf));
    FILE *fp = fopen(resolved, "rb");
    TEST_ASSERT(fp != NULL);

    uint32_t header[3];
    TEST_ASSERT(fread(header, sizeof(uint32_t), 3, fp) == 3);

    int n = (int)header[0];
    int heads = (int)header[1];
    int hd = (int)header[2];
    TEST_ASSERT(n == 777 && heads == 4 && hd == 128);

    size_t count = (size_t)n * heads * hd;
    float *q = (float *)malloc(count * sizeof(float));
    float *k = (float *)malloc(count * sizeof(float));
    float *v = (float *)malloc(count * sizeof(float));
    float *o = (float *)malloc(count * sizeof(float));
    float *ref = (float *)malloc(count * sizeof(float));
    TEST_ASSERT(q && k && v && o && ref);

    TEST_ASSERT(fread(q, sizeof(float), count, fp) == count);
    TEST_ASSERT(fread(k, sizeof(float), count, fp) == count);
    TEST_ASSERT(fread(v, sizeof(float), count, fp) == count);
    TEST_ASSERT(fread(ref, sizeof(float), count, fp) == count);
    fclose(fp);

    fh3_attention(q, k, v, o, n, heads, hd);

    float max_err = 0.0f;
    bool ok = check_close(o, ref, count, 1e-5f, &max_err);
    printf("  Attention max relative error: %e\n", max_err);

    free(q);
    free(k);
    free(v);
    free(o);
    free(ref);

    TEST_ASSERT(ok);
    return 0;
}

static int test_timestep_embed(void) {
    printf("--- test_timestep_embed ---\n");
    char path_buf[512];
    const char *resolved = resolve_fixture("tests/fixtures/ops/timestep_embed.bin", path_buf, sizeof(path_buf));
    FILE *fp = fopen(resolved, "rb");
    TEST_ASSERT(fp != NULL);

    uint32_t header[2];
    TEST_ASSERT(fread(header, sizeof(uint32_t), 2, fp) == 2);

    int nt = (int)header[0];
    int freq_dim = (int)header[1];
    TEST_ASSERT(nt == 5 && freq_dim == 256);

    double *t = (double *)malloc(nt * sizeof(double));
    size_t out_count = (size_t)nt * freq_dim;
    float *out = (float *)malloc(out_count * sizeof(float));
    float *ref = (float *)malloc(out_count * sizeof(float));
    TEST_ASSERT(t && out && ref);

    TEST_ASSERT(fread(t, sizeof(double), nt, fp) == (size_t)nt);
    TEST_ASSERT(fread(ref, sizeof(float), out_count, fp) == out_count);
    fclose(fp);

    fh3_timestep_embed(t, nt, freq_dim, out);

    float max_err = 0.0f;
    bool ok = check_close(out, ref, out_count, 1e-5f, &max_err);
    printf("  TimestepEmbed max relative error: %e\n", max_err);

    free(t);
    free(out);
    free(ref);

    TEST_ASSERT(ok);
    return 0;
}

int main(void) {
    printf("=========================================\n");
    printf("Running FastH3 Ops Unit Test Suite\n");
    printf("=========================================\n");

    int rc = 0;
    rc |= test_rmsnorm();
    rc |= test_rope3();
    rc |= test_swiglu();
    rc |= test_modulate();
    rc |= test_gate_residual();
    rc |= test_attention();
    rc |= test_timestep_embed();

    if (rc != 0 || g_failures != 0) {
        fprintf(stderr, "\nOps tests FAILED with %d failure(s).\n", g_failures);
        return 1;
    }

    printf("\nAll ops tests PASSED successfully!\n");
    return 0;
}
