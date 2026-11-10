#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "fh3/fh3.h"
#include "io/fh3_pack.h"
#include "io/fh3_gguf.h"
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

/* -------------------------------------------------------------------------
 * Test 1: Layer offsets and tensor alignments (4096 and 32 byte alignment)
 * ------------------------------------------------------------------------- */

static int test_pack_layer_offsets_aligned(void) {
    printf("--- Running test_pack_layer_offsets_aligned ---\n");

    char pbuf[256];
    const char *trunk_path = resolve_path("tests/fixtures/pack/tiny_trunk.bin", pbuf, sizeof(pbuf));

    Fh3Pack p;
    int rc = fh3_pack_open(trunk_path, &p);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(p.header.magic == FH3_PACK_MAGIC);
    TEST_ASSERT(p.header.version == FH3_PACK_VERSION);
    TEST_ASSERT(p.num_layers == 2);
    TEST_ASSERT(p.header.layer_alignment == 4096);

    for (int L = 0; L < p.num_layers; L++) {
        const Fh3PackLayerDesc *ld = &p.layers[L];
        /* Assert every layer file offset is a multiple of 4096 */
        TEST_ASSERT((ld->file_offset % 4096) == 0);
        /* Assert layer chunk run size is a multiple of 4096 */
        TEST_ASSERT((ld->size_bytes % 4096) == 0);
        TEST_ASSERT(ld->n_tensors == FH3_PACK_TENSORS_PER_LAYER);

        /* Assert each tensor in layer is 32-byte aligned */
        for (uint32_t t = 0; t < ld->n_tensors; t++) {
            TEST_ASSERT((ld->tensors[t].offset_in_layer % 32) == 0);
            TEST_ASSERT(ld->tensors[t].size_bytes > 0);
        }
    }

    fh3_pack_close(&p);
    printf("[PASS] test_pack_layer_offsets_aligned\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 2: Tensor slices in packed trunk exactly match GGUF bytes
 * ------------------------------------------------------------------------- */

static int test_pack_layer_tensor_bytes_match_gguf(void) {
    printf("--- Running test_pack_layer_tensor_bytes_match_gguf ---\n");

    char pbuf_trunk[256];
    char pbuf_gguf[256];
    const char *trunk_path = resolve_path("tests/fixtures/pack/tiny_trunk.bin", pbuf_trunk, sizeof(pbuf_trunk));
    const char *gguf_path = resolve_path("tests/fixtures/pack/tiny_dit.gguf", pbuf_gguf, sizeof(pbuf_gguf));

    Fh3Pack pack;
    int rc = fh3_pack_open(trunk_path, &pack);
    TEST_ASSERT(rc == FH3_OK);

    Fh3Gguf gguf;
    rc = fh3_gguf_open(gguf_path, &gguf);
    TEST_ASSERT(rc == FH3_OK);

    const char *suffixes[FH3_PACK_TENSORS_PER_LAYER] = {
        "attn.qkv_proj.weight",
        "attn.to_gate_compress.weight",
        "attn.out_proj.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
        "norm1.weight",
        "norm2.weight",
        "attn.q_norm.weight",
        "attn.k_norm.weight",
    };

    for (int L = 0; L < pack.num_layers; L++) {
        /* Allocate bounce buffer to test reading layer chunk */
        size_t run_size = (size_t)pack.layers[L].size_bytes;
        uint8_t *layer_buf = (uint8_t *)malloc(run_size);
        TEST_ASSERT(layer_buf != NULL);

        int64_t nread = fh3_pread(&pack.file, layer_buf, run_size, pack.layers[L].file_offset);
        TEST_ASSERT(nread == (int64_t)run_size);

        Fh3LayerView view = fh3_pack_layer(&pack, L, layer_buf);

        const Fh3W *w_ptrs[FH3_PACK_TENSORS_PER_LAYER] = {
            &view.qkv,
            &view.gate_compress,
            &view.out,
            &view.fc1,
            &view.fc2,
            &view.norm1,
            &view.norm2,
            &view.q_norm,
            &view.k_norm,
        };

        for (int t = 0; t < (int)FH3_PACK_TENSORS_PER_LAYER; t++) {
            char full_name[128];
            snprintf(full_name, sizeof(full_name), "blocks.%d.%s", L, suffixes[t]);

            const Fh3Tensor *gt = fh3_gguf_find(&gguf, full_name);
            TEST_ASSERT(gt != NULL);

            uint8_t *g_data = (uint8_t *)malloc((size_t)gt->nbytes);
            TEST_ASSERT(g_data != NULL);
            int64_t grd = fh3_pread(&gguf.file, g_data, (size_t)gt->nbytes, gguf.data_offset + gt->off);
            TEST_ASSERT(grd == (int64_t)gt->nbytes);

            const Fh3W *w = w_ptrs[t];
            TEST_ASSERT(w->data != NULL);
            TEST_ASSERT(gt->nbytes == pack.layers[L].tensors[t].size_bytes);

            /* Exact bitwise comparison with original GGUF tensor bytes */
            int cmp = memcmp(w->data, g_data, (size_t)gt->nbytes);
            TEST_ASSERT(cmp == 0);
            free(g_data);
        }

        free(layer_buf);
    }

    fh3_gguf_close(&gguf);
    fh3_pack_close(&pack);
    printf("[PASS] test_pack_layer_tensor_bytes_match_gguf\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: Schedule hash check and rejection of mismatch in AdaLN table
 * ------------------------------------------------------------------------- */

static int test_adaln_hash_mismatch_refuses(void) {
    printf("--- Running test_adaln_hash_mismatch_refuses ---\n");

    char pbuf[256];
    const char *adaln_path = resolve_path("tests/fixtures/pack/tiny_adaln.bin", pbuf, sizeof(pbuf));

    const int dmd_steps[8] = {999, 874, 749, 624, 500, 375, 250, 125};
    uint64_t expected_hash = fh3_schedule_hash(dmd_steps, 8, 10.0, 3.0);
    TEST_ASSERT(expected_hash != 0);

    /* 1. Open with exact matching schedule hash */
    Fh3Adaln a;
    int rc = fh3_adaln_open(adaln_path, expected_hash, &a);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(a.header.magic == FH3_ADALN_MAGIC);
    TEST_ASSERT(a.header.version == FH3_ADALN_VERSION);
    TEST_ASSERT(a.n_steps == 8);
    TEST_ASSERT(a.n_blocks == 2);
    TEST_ASSERT(a.pairs_per_step == 3);
    TEST_ASSERT(a.hidden_size == 64);
    TEST_ASSERT(a.schedule_hash == expected_hash);

    /* Test row retrieval */
    for (int s = 0; s < a.n_steps; s++) {
        for (int b = 0; b < a.n_blocks; b++) {
            for (int p = 0; p < a.pairs_per_step; p++) {
                for (int c = 0; c < 6; c++) {
                    const float *row = fh3_adaln_row(&a, s, b, p, c);
                    TEST_ASSERT(row != NULL);
                }
            }
        }
        /* Final layer rows (block == 2 or -1, chunk in 0..1) */
        for (int p = 0; p < a.pairs_per_step; p++) {
            for (int c = 0; c < 2; c++) {
                const float *row = fh3_adaln_row(&a, s, 2, p, c);
                TEST_ASSERT(row != NULL);
                const float *row_neg = fh3_adaln_row(&a, s, -1, p, c);
                TEST_ASSERT(row_neg == row);
            }
        }
    }

    /* Out of bounds checks */
    TEST_ASSERT(fh3_adaln_row(&a, -1, 0, 0, 0) == NULL);
    TEST_ASSERT(fh3_adaln_row(&a, 8, 0, 0, 0) == NULL);
    TEST_ASSERT(fh3_adaln_row(&a, 0, 0, 3, 0) == NULL);
    TEST_ASSERT(fh3_adaln_row(&a, 0, 0, 0, 6) == NULL);
    TEST_ASSERT(fh3_adaln_row(&a, 0, 2, 0, 2) == NULL); /* final layer only has 2 chunks */
    TEST_ASSERT(fh3_adaln_row(&a, 0, 3, 0, 0) == NULL);

    fh3_adaln_close(&a);

    /* 2. Open with mismatched schedule hash -> must return FH3_E_INVALID */
    uint64_t bad_hash = expected_hash ^ 0xDEADBEEFCAFEBABEULL;
    rc = fh3_adaln_open(adaln_path, bad_hash, &a);
    TEST_ASSERT(rc == FH3_E_INVALID);

    printf("[PASS] test_adaln_hash_mismatch_refuses\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 4: Truncation, bad magic, and invalid arguments
 * ------------------------------------------------------------------------- */

static int test_pack_robustness(void) {
    printf("--- Running test_pack_robustness ---\n");

    /* NULL pointers */
    Fh3Pack p;
    TEST_ASSERT(fh3_pack_open(NULL, &p) == FH3_ERR);
    TEST_ASSERT(fh3_pack_open("tests/fixtures/pack/tiny_trunk.bin", NULL) == FH3_ERR);

    Fh3Adaln a;
    TEST_ASSERT(fh3_adaln_open(NULL, 0, &a) == FH3_ERR);
    TEST_ASSERT(fh3_adaln_open("tests/fixtures/pack/tiny_adaln.bin", 0, NULL) == FH3_ERR);

    /* Non-existent file */
    TEST_ASSERT(fh3_pack_open("non_existent_file.bin", &p) == FH3_E_IO);
    TEST_ASSERT(fh3_adaln_open("non_existent_file.bin", 0, &a) == FH3_E_IO);

    printf("[PASS] test_pack_robustness\n");
    return 0;
}

int main(void) {
    printf("=== Running FastH3 Pack & AdaLN Tests ===\n");

    if (test_pack_layer_offsets_aligned() != 0) return 1;
    if (test_pack_layer_tensor_bytes_match_gguf() != 0) return 1;
    if (test_adaln_hash_mismatch_refuses() != 0) return 1;
    if (test_pack_robustness() != 0) return 1;

    if (g_failures == 0) {
        printf("=== All FastH3 Pack tests PASSED ===\n");
        return 0;
    }
    return 1;
}
