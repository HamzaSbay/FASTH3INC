#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "fh3/fh3.h"
#include "io/fh3_portable_io.h"
#include "io/fh3_gguf.h"

static int g_failures = 0;

static const char *resolve_path(const char *path, char *buf, size_t sz) {
    FILE *fp = fopen(path, "rb");
    if (fp) {
        fclose(fp);
        return path;
    }
    snprintf(buf, sz, "../%s", path);
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

static int test_portable_io_primitives(void) {
    printf("--- Running test_portable_io_primitives ---\n");

    /* 1. Aligned alloc & free */
    void *ptr = fh3_alloc_aligned(4096);
    TEST_ASSERT(ptr != NULL);
    TEST_ASSERT(((uintptr_t)ptr % 4096) == 0);
    memset(ptr, 0xAB, 4096);
    fh3_free_aligned(ptr);

    /* 2. Available memory query */
    uint64_t mem = fh3_mem_available();
    printf("Available system physical memory: %llu MB\n", (unsigned long long)(mem / (1024 * 1024)));
    TEST_ASSERT(mem > 0);

    /* 3. File open & positional read */
    char pbuf[256];
    const char *fixture_path = resolve_path("tests/fixtures/gguf/tiny.gguf", pbuf, sizeof(pbuf));
    Fh3File f;
    int rc = fh3_open_file(fixture_path, &f);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(f.size >= 288);

    char magic[4] = {0};
    int64_t nread = fh3_pread(&f, magic, 4, 0);
    TEST_ASSERT(nread == 4);
    TEST_ASSERT(memcmp(magic, "GGUF", 4) == 0);
    fh3_close_file(&f);

    /* 4. Direct file open */
    Fh3File f_direct;
    rc = fh3_open_direct(fixture_path, &f_direct);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(f_direct.size >= 288);

    void *aligned_buf = fh3_alloc_aligned(4096);
    TEST_ASSERT(aligned_buf != NULL);
    nread = fh3_pread(&f_direct, aligned_buf, 4, 0);
    TEST_ASSERT(nread == 4);
    TEST_ASSERT(memcmp(aligned_buf, "GGUF", 4) == 0);
    fh3_free_aligned(aligned_buf);
    fh3_close_file(&f_direct);

    printf("[PASS] test_portable_io_primitives\n");
    return 0;
}

static int test_gguf_reads_tiny(void) {
    printf("--- Running test_gguf_reads_tiny ---\n");

    char pbuf[256];
    const char *fixture_path = resolve_path("tests/fixtures/gguf/tiny.gguf", pbuf, sizeof(pbuf));
    Fh3Gguf g;
    int rc = fh3_gguf_open(fixture_path, &g);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(g.tensor_count == 3);
    TEST_ASSERT(g.data_offset == 288);

    /* Find t_f32 */
    const Fh3Tensor *t_f32 = fh3_gguf_find(&g, "t_f32");
    TEST_ASSERT(t_f32 != NULL);
    TEST_ASSERT(t_f32->type == 0); /* F32 */
    TEST_ASSERT(t_f32->ndim == 2);
    TEST_ASSERT(t_f32->ne[0] == 4 && t_f32->ne[1] == 4);
    TEST_ASSERT(t_f32->off == 0);
    TEST_ASSERT(t_f32->nbytes == 64);

    /* Find t_f16 */
    const Fh3Tensor *t_f16 = fh3_gguf_find(&g, "t_f16");
    TEST_ASSERT(t_f16 != NULL);
    TEST_ASSERT(t_f16->type == 1); /* F16 */
    TEST_ASSERT(t_f16->ndim == 2);
    TEST_ASSERT(t_f16->ne[0] == 4 && t_f16->ne[1] == 4);
    TEST_ASSERT(t_f16->off == 64);
    TEST_ASSERT(t_f16->nbytes == 32);

    /* Find x */
    const Fh3Tensor *t_x = fh3_gguf_find(&g, "x");
    TEST_ASSERT(t_x != NULL);
    TEST_ASSERT(t_x->type == 12); /* Q4_K */
    TEST_ASSERT(t_x->ndim == 1);
    TEST_ASSERT(t_x->ne[0] == 516096);
    TEST_ASSERT(t_x->off == 96);
    TEST_ASSERT(t_x->nbytes == 290304);

    /* Verify non-existent tensor */
    const Fh3Tensor *t_missing = fh3_gguf_find(&g, "does_not_exist");
    TEST_ASSERT(t_missing == NULL);

    fh3_gguf_close(&g);
    printf("[PASS] test_gguf_reads_tiny\n");
    return 0;
}

static int test_gguf_orig_shape_applies(void) {
    printf("--- Running test_gguf_orig_shape_applies ---\n");

    char pbuf[256];
    const char *fixture_path = resolve_path("tests/fixtures/gguf/tiny.gguf", pbuf, sizeof(pbuf));
    Fh3Gguf g;
    int rc = fh3_gguf_open(fixture_path, &g);
    TEST_ASSERT(rc == FH3_OK);

    const Fh3Tensor *t_x = fh3_gguf_find(&g, "x");
    TEST_ASSERT(t_x != NULL);

    /* Stored dims are 1D: [516096] */
    TEST_ASSERT(t_x->ndim == 1);
    TEST_ASSERT(t_x->ne[0] == 516096);

    /* fh3_gguf_shape should apply comfy.gguf.orig_shape.x -> [96, 5376] */
    int64_t shape_x[4] = {0};
    int ndim = fh3_gguf_shape(&g, t_x, shape_x);
    TEST_ASSERT(ndim == 2);
    TEST_ASSERT(shape_x[0] == 96);
    TEST_ASSERT(shape_x[1] == 5376);

    /* For tensor without orig_shape (t_f32), shape matches stored dims */
    const Fh3Tensor *t_f32 = fh3_gguf_find(&g, "t_f32");
    TEST_ASSERT(t_f32 != NULL);
    int64_t shape_f32[4] = {0};
    ndim = fh3_gguf_shape(&g, t_f32, shape_f32);
    TEST_ASSERT(ndim == 2);
    TEST_ASSERT(shape_f32[0] == 4);
    TEST_ASSERT(shape_f32[1] == 4);

    fh3_gguf_close(&g);
    printf("[PASS] test_gguf_orig_shape_applies\n");
    return 0;
}

static int test_gguf_truncated_refuses(void) {
    printf("--- Running test_gguf_truncated_refuses ---\n");

    char pbuf[256];
    const char *fixture_path = resolve_path("tests/fixtures/gguf/tiny.gguf", pbuf, sizeof(pbuf));
    FILE *fp = fopen(fixture_path, "rb");
    TEST_ASSERT(fp != NULL);

    fseek(fp, 0, SEEK_END);
    long full_size = ftell(fp);
    TEST_ASSERT(full_size > 100);

    long trunc_size = (full_size * 60) / 100; /* Cut at 60% */
    uint8_t *buf = (uint8_t *)malloc((size_t)trunc_size);
    TEST_ASSERT(buf != NULL);

    fseek(fp, 0, SEEK_SET);
    size_t rd = fread(buf, 1, (size_t)trunc_size, fp);
    fclose(fp);
    TEST_ASSERT(rd == (size_t)trunc_size);

    const char *trunc_path = "tiny_trunc_tmp.gguf";
    FILE *out = fopen(trunc_path, "wb");
    TEST_ASSERT(out != NULL);
    fwrite(buf, 1, (size_t)trunc_size, out);
    fclose(out);
    free(buf);

    Fh3Gguf g;
    int rc = fh3_gguf_open(trunc_path, &g);
    remove(trunc_path);

    /* Must refuse with FH3_E_TRUNCATED */
    TEST_ASSERT(rc == FH3_E_TRUNCATED);

    printf("[PASS] test_gguf_truncated_refuses\n");
    return 0;
}

static int test_gguf_bad_magic_refuses(void) {
    printf("--- Running test_gguf_bad_magic_refuses ---\n");

    char pbuf[256];
    const char *fixture_path = resolve_path("tests/fixtures/gguf/tiny.gguf", pbuf, sizeof(pbuf));
    FILE *fp = fopen(fixture_path, "rb");
    TEST_ASSERT(fp != NULL);

    fseek(fp, 0, SEEK_END);
    long full_size = ftell(fp);
    uint8_t *buf = (uint8_t *)malloc((size_t)full_size);
    TEST_ASSERT(buf != NULL);

    fseek(fp, 0, SEEK_SET);
    size_t rd = fread(buf, 1, (size_t)full_size, fp);
    fclose(fp);
    TEST_ASSERT(rd == (size_t)full_size);

    /* Corrupt magic */
    buf[0] = 'N'; buf[1] = 'O'; buf[2] = 'P'; buf[3] = 'E';

    const char *bad_path = "tiny_badmagic_tmp.gguf";
    FILE *out = fopen(bad_path, "wb");
    TEST_ASSERT(out != NULL);
    fwrite(buf, 1, (size_t)full_size, out);
    fclose(out);
    free(buf);

    Fh3Gguf g;
    int rc = fh3_gguf_open(bad_path, &g);
    remove(bad_path);

    /* Must refuse with FH3_E_BAD_MAGIC */
    TEST_ASSERT(rc == FH3_E_BAD_MAGIC);

    printf("[PASS] test_gguf_bad_magic_refuses\n");
    return 0;
}

static int test_gguf_real_q4_slice(void) {
    printf("--- Running test_gguf_real_q4_slice ---\n");

    char pbuf[256];
    const char *slice_path = resolve_path("scratch/q4_head.bin", pbuf, sizeof(pbuf));
    FILE *fp = fopen(slice_path, "rb");
    if (!fp) {
        printf("[SKIP] scratch/q4_head.bin not present\n");
        return 0;
    }
    fclose(fp);

    /* scratch/q4_head.bin is a 16MB slice of the full 15GB model.
       The header must parse cleanly, and the size validation check
       must report truncated (FH3_E_TRUNCATED) because 15GB > 16MB. */
    Fh3Gguf g;
    int rc = fh3_gguf_open(slice_path, &g);
    TEST_ASSERT(rc == FH3_E_TRUNCATED);

    printf("[PASS] test_gguf_real_q4_slice\n");
    return 0;
}

int main(void) {
    printf("=== Starting test_gguf suite ===\n");

    if (test_portable_io_primitives() != 0) return 1;
    if (test_gguf_reads_tiny() != 0) return 1;
    if (test_gguf_orig_shape_applies() != 0) return 1;
    if (test_gguf_truncated_refuses() != 0) return 1;
    if (test_gguf_bad_magic_refuses() != 0) return 1;
    if (test_gguf_real_q4_slice() != 0) return 1;

    if (g_failures == 0) {
        printf("=== ALL GGUF TESTS PASSED ===\n");
        return 0;
    } else {
        printf("=== GGUF TESTS FAILED: %d failures ===\n", g_failures);
        return 1;
    }
}
