#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

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

static int test_dequant_type_bitexact(const char *name, int expected_type) {
    printf("--- Running dequant_%s_bitexact ---\n", name);

    char path_buf[512];
    char rel_path[256];
    snprintf(rel_path, sizeof(rel_path), "tests/fixtures/quant/%s.bin", name);
    const char *resolved = resolve_fixture(rel_path, path_buf, sizeof(path_buf));

    FILE *fp = fopen(resolved, "rb");
    if (!fp) {
        fprintf(stderr, "Could not open fixture file: %s (resolved: %s)\n", rel_path, resolved);
        TEST_ASSERT(fp != NULL);
    }

    uint32_t header[4]; // type, n_rows, n_cols, raw_nbytes
    size_t nr = fread(header, sizeof(uint32_t), 4, fp);
    TEST_ASSERT(nr == 4);

    uint32_t type = header[0];
    uint32_t n_rows = header[1];
    uint32_t n_cols = header[2];
    uint32_t raw_nbytes = header[3];

    TEST_ASSERT(type == (uint32_t)expected_type);
    TEST_ASSERT(n_rows == 4);
    TEST_ASSERT(n_cols == 512);

    size_t expected_row_bytes = fh3_row_bytes((int)type, (int64_t)n_cols);
    TEST_ASSERT(expected_row_bytes * n_rows == raw_nbytes);

    uint8_t *raw_data = (uint8_t *)malloc(raw_nbytes);
    TEST_ASSERT(raw_data != NULL);
    nr = fread(raw_data, 1, raw_nbytes, fp);
    TEST_ASSERT(nr == raw_nbytes);

    size_t total_elements = (size_t)n_rows * n_cols;
    float *ref_f32 = (float *)malloc(total_elements * sizeof(float));
    TEST_ASSERT(ref_f32 != NULL);
    nr = fread(ref_f32, sizeof(float), total_elements, fp);
    TEST_ASSERT(nr == total_elements);

    fclose(fp);

    float *out_f32 = (float *)malloc(total_elements * sizeof(float));
    TEST_ASSERT(out_f32 != NULL);

    for (uint32_t r = 0; r < n_rows; r++) {
        const void *src_row = raw_data + r * expected_row_bytes;
        float *dst_row = out_f32 + r * n_cols;
        fh3_dequant((int)type, src_row, dst_row, (int64_t)n_cols);
    }

    int mismatches = 0;
    for (size_t i = 0; i < total_elements; i++) {
        uint32_t b_out, b_ref;
        memcpy(&b_out, &out_f32[i], sizeof(uint32_t));
        memcpy(&b_ref, &ref_f32[i], sizeof(uint32_t));
        if (b_out != b_ref) {
            if (mismatches < 5) {
                fprintf(stderr, "Mismatch [%s] at element %zu: got=%f (0x%08x), ref=%f (0x%08x)\n",
                        name, i, out_f32[i], b_out, ref_f32[i], b_ref);
            }
            mismatches++;
        }
    }

    free(raw_data);
    free(ref_f32);
    free(out_f32);

    TEST_ASSERT(mismatches == 0);
    printf("dequant_%s_bitexact PASS (all %zu floats match bit-for-bit)\n", name, total_elements);
    return 0;
}

int main(void) {
    const struct {
        const char *name;
        int type;
    } tests[] = {
        {"F16",  FH3_QUANT_F16},
        {"BF16", FH3_QUANT_BF16},
        {"Q8_0", FH3_QUANT_Q8_0},
        {"Q2_K", FH3_QUANT_Q2_K},
        {"Q3_K", FH3_QUANT_Q3_K},
        {"Q4_K", FH3_QUANT_Q4_K},
        {"Q5_K", FH3_QUANT_Q5_K},
        {"Q6_K", FH3_QUANT_Q6_K},
    };

    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    for (size_t i = 0; i < n_tests; i++) {
        if (test_dequant_type_bitexact(tests[i].name, tests[i].type) != 0) {
            fprintf(stderr, "FAILED test for %s\n", tests[i].name);
        }
    }

    if (g_failures > 0) {
        fprintf(stderr, "test_quant failed with %d failure(s)\n", g_failures);
        return 1;
    }

    printf("ALL QUANT TESTS PASSED!\n");
    return 0;
}
