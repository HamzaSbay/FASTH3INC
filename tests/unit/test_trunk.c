#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "fh3/fh3.h"
#include "io/fh3_trunk.h"
#include "io/fh3_portable_io.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static void test_sleep_ms(int ms) {
    Sleep((DWORD)ms);
}
#else
#include <unistd.h>
static void test_sleep_ms(int ms) {
    usleep((useconds_t)ms * 1000);
}
#endif

static int g_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return 1; \
    } \
} while (0)

#define TEST_LAYER_SIZE (1024 * 1024)  /* 1 MB per layer */
#define TEST_NUM_LAYERS 10

/* -------------------------------------------------------------------------
 * Helper: Generate synthetic 10-layer FH3P pack file
 * ------------------------------------------------------------------------- */

static int create_synthetic_pack(const char *path, int num_layers, size_t layer_size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    Fh3PackHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = FH3_PACK_MAGIC;
    hdr.version = FH3_PACK_VERSION;
    hdr.num_layers = (uint32_t)num_layers;
    hdr.layer_alignment = 4096;
    hdr.index_offset = 4096;
    hdr.index_bytes = (uint64_t)num_layers * sizeof(Fh3PackLayerDesc);

    uint64_t data_off = (hdr.index_offset + hdr.index_bytes + 4095ULL) & ~4095ULL;
    hdr.data_offset = data_off;
    hdr.total_bytes = data_off + (uint64_t)num_layers * layer_size;

    fwrite(&hdr, 1, sizeof(hdr), fp);

    /* Pad to index_offset (4096) */
    uint8_t pad[4096];
    memset(pad, 0, sizeof(pad));
    size_t cur = sizeof(hdr);
    if (cur < hdr.index_offset) {
        fwrite(pad, 1, (size_t)(hdr.index_offset - cur), fp);
    }

    /* Write layer descriptors */
    Fh3PackLayerDesc *descs = (Fh3PackLayerDesc *)calloc((size_t)num_layers, sizeof(Fh3PackLayerDesc));
    if (!descs) {
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < num_layers; i++) {
        descs[i].file_offset = data_off + (uint64_t)i * layer_size;
        descs[i].size_bytes = (uint64_t)layer_size;
        descs[i].n_tensors = FH3_PACK_TENSORS_PER_LAYER;
        for (uint32_t t = 0; t < descs[i].n_tensors; t++) {
            snprintf(descs[i].tensors[t].name, sizeof(descs[i].tensors[t].name), "tensor_%d_%u", i, t);
            descs[i].tensors[t].type = FH3_QUANT_F32;
            descs[i].tensors[t].n_dims = 2;
            descs[i].tensors[t].shape[0] = 64;
            descs[i].tensors[t].shape[1] = 64;
            descs[i].tensors[t].offset_in_layer = (uint64_t)t * 4096;
            descs[i].tensors[t].size_bytes = 4096;
            descs[i].tensors[t].in_dim = 64;
            descs[i].tensors[t].out_dim = 64;
        }
    }

    fwrite(descs, sizeof(Fh3PackLayerDesc), (size_t)num_layers, fp);
    free(descs);

    /* Pad to data_offset */
    long pos = ftell(fp);
    if ((uint64_t)pos < data_off) {
        size_t to_pad = (size_t)(data_off - (uint64_t)pos);
        while (to_pad > 0) {
            size_t chunk = (to_pad > sizeof(pad)) ? sizeof(pad) : to_pad;
            fwrite(pad, 1, chunk, fp);
            to_pad -= chunk;
        }
    }

    /* Write layer payload data (patterned by layer index) */
    uint8_t *layer_buf = (uint8_t *)malloc(layer_size);
    if (!layer_buf) {
        fclose(fp);
        return -1;
    }
    for (int i = 0; i < num_layers; i++) {
        memset(layer_buf, (uint8_t)(0xA0 + i), layer_size);
        fwrite(layer_buf, 1, layer_size, fp);
    }
    free(layer_buf);

    fclose(fp);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 1: Pinned prefix hit rate
 * 10-layer pack, budget for 6 pinned + 2 ring slots.
 * 3 sequential scans (0..9) must produce exactly 18 hits and 12 misses.
 * ------------------------------------------------------------------------- */

static int test_trunk_pinned_prefix_hit_rate(const char *pack_path) {
    printf("--- Running test_trunk_pinned_prefix_hit_rate ---\n");

    /* Budget: 6 pinned layers + 2 ring slots = 8 * TEST_LAYER_SIZE */
    uint64_t budget = 8ULL * TEST_LAYER_SIZE;

    Fh3Trunk tr;
    int rc = fh3_trunk_open(pack_path, budget, &tr);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(tr.num_layers == TEST_NUM_LAYERS);
    TEST_ASSERT(tr.npin == 6);
    TEST_ASSERT(tr.slot_size == TEST_LAYER_SIZE);

    /* 3 sequential scans (0..9) */
    for (int scan = 0; scan < 3; scan++) {
        for (int L = 0; L < TEST_NUM_LAYERS; L++) {
            const uint8_t *ptr = fh3_trunk_acquire(&tr, L);
            TEST_ASSERT(ptr != NULL);
            /* Verify byte pattern matches layer payload */
            TEST_ASSERT(ptr[0] == (uint8_t)(0xA0 + L));
            TEST_ASSERT(ptr[TEST_LAYER_SIZE - 1] == (uint8_t)(0xA0 + L));
        }
    }

    Fh3TrunkStats stats = fh3_trunk_stats(&tr);
    printf("Stats after 3 scans: hits=%llu, misses=%llu, bytes_read=%llu\n",
           (unsigned long long)stats.hits,
           (unsigned long long)stats.misses,
           (unsigned long long)stats.bytes_read);

    TEST_ASSERT(stats.hits == 18);
    TEST_ASSERT(stats.misses == 12);
    TEST_ASSERT(stats.bytes_read == 12ULL * TEST_LAYER_SIZE);

    fh3_trunk_close(&tr);
    printf("[PASS] test_trunk_pinned_prefix_hit_rate\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 2: Trunk never LRU
 * Cyclic scan through pure LRU of capacity 8 gets 0 hits.
 * Our pinned prefix + ring gets > 0 hits (specifically 18 hits out of 30).
 * ------------------------------------------------------------------------- */

static int test_trunk_never_lru(const char *pack_path) {
    printf("--- Running test_trunk_never_lru ---\n");

    /* Pure LRU simulation with capacity 8 */
    int lru[8];
    int lru_count = 0;
    int lru_hits = 0;

    for (int scan = 0; scan < 3; scan++) {
        for (int L = 0; L < TEST_NUM_LAYERS; L++) {
            int found_idx = -1;
            for (int i = 0; i < lru_count; i++) {
                if (lru[i] == L) {
                    found_idx = i;
                    break;
                }
            }
            if (found_idx >= 0) {
                lru_hits++;
                int hit_val = lru[found_idx];
                for (int i = found_idx; i < lru_count - 1; i++) {
                    lru[i] = lru[i + 1];
                }
                lru[lru_count - 1] = hit_val;
            } else {
                if (lru_count < 8) {
                    lru[lru_count++] = L;
                } else {
                    for (int i = 0; i < 7; i++) {
                        lru[i] = lru[i + 1];
                    }
                    lru[7] = L;
                }
            }
        }
    }

    printf("Simulated LRU(capacity=8) hits: %d (expected 0)\n", lru_hits);
    TEST_ASSERT(lru_hits == 0);

    /* FastH3 pinned prefix on the identical sequence */
    uint64_t budget = 8ULL * TEST_LAYER_SIZE;
    Fh3Trunk tr;
    int rc = fh3_trunk_open(pack_path, budget, &tr);
    TEST_ASSERT(rc == FH3_OK);

    for (int scan = 0; scan < 3; scan++) {
        for (int L = 0; L < TEST_NUM_LAYERS; L++) {
            const uint8_t *ptr = fh3_trunk_acquire(&tr, L);
            TEST_ASSERT(ptr != NULL);
        }
    }

    Fh3TrunkStats stats = fh3_trunk_stats(&tr);
    printf("FastH3 hits: %llu (hit rate: %.1f%%)\n",
           (unsigned long long)stats.hits,
           (double)stats.hits / (double)(stats.hits + stats.misses) * 100.0);

    TEST_ASSERT(stats.hits > 0);
    TEST_ASSERT(stats.hits == 18);

    fh3_trunk_close(&tr);
    printf("[PASS] test_trunk_never_lru\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: Prefetch overlaps compute
 * Verify wait_s < read_s when work is done between prefetch and acquire.
 * ------------------------------------------------------------------------- */

static int test_trunk_prefetch_overlaps(const char *pack_path) {
    printf("--- Running test_trunk_prefetch_overlaps ---\n");

    uint64_t budget = 8ULL * TEST_LAYER_SIZE;
    Fh3Trunk tr;
    int rc = fh3_trunk_open(pack_path, budget, &tr);
    TEST_ASSERT(rc == FH3_OK);

    /* Layer 6 is a streaming layer (npin == 6) */
    fh3_trunk_prefetch(&tr, 6);

    /* Inject compute / work while prefetch is in flight */
    test_sleep_ms(25);

    const uint8_t *ptr = fh3_trunk_acquire(&tr, 6);
    TEST_ASSERT(ptr != NULL);
    TEST_ASSERT(ptr[0] == (uint8_t)(0xA0 + 6));

    Fh3TrunkStats stats = fh3_trunk_stats(&tr);
    printf("Overlap stats: read_s=%.6f s, wait_s=%.6f s\n", stats.read_s, stats.wait_s);

    TEST_ASSERT(stats.read_s > 0.0);
    TEST_ASSERT(stats.wait_s < stats.read_s);

    fh3_trunk_close(&tr);
    printf("[PASS] test_trunk_prefetch_overlaps\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 4: Plan refuses when short
 * Verify FH3_E_NOMEM when requested exceeds available RAM or below minimum.
 * ------------------------------------------------------------------------- */

static int test_plan_refuses_when_short(void) {
    printf("--- Running test_plan_refuses_when_short ---\n");

    /* Sub-test 4A: requested exceeds 95% of available RAM */
    uint64_t avail = fh3_mem_available();
    TEST_ASSERT(avail > 0);

    Fh3Plan p_huge;
    memset(&p_huge, 0, sizeof(p_huge));
    p_huge.total_bytes = avail + 20ULL * 1024 * 1024 * 1024; /* Exceeds physical RAM */

    int rc = fh3_plan_check(&p_huge);
    TEST_ASSERT(rc == FH3_E_NOMEM);

    /* Sub-test 4B: budget below minimum required memory (e.g. --mem-gb 0.5) */
    Fh3Plan p_short;
    memset(&p_short, 0, sizeof(p_short));
    p_short.min_bytes = 2ULL * 1024 * 1024 * 1024; /* Minimum 2 GB needed */
    p_short.budget_bytes = 536870912ULL;           /* 0.5 GB provided */
    rc = fh3_plan_check(&p_short);
    TEST_ASSERT(rc == FH3_E_NOMEM);

    /* Sub-test 4C: valid plan within limits passes */
    Fh3Plan p_valid;
    memset(&p_valid, 0, sizeof(p_valid));
    p_valid.min_bytes = 1024 * 1024;
    p_valid.budget_bytes = 32 * 1024 * 1024; /* 32 MB */
    rc = fh3_plan_check(&p_valid);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(fh3_plan_bytes(&p_valid) == 32 * 1024 * 1024);

    printf("[PASS] test_plan_refuses_when_short\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Main test runner
 * ------------------------------------------------------------------------- */

int main(void) {
    const char *synth_pack = "test_synthetic_10layer.bin";
    printf("Creating synthetic pack: %s\n", synth_pack);
    int rc = create_synthetic_pack(synth_pack, TEST_NUM_LAYERS, TEST_LAYER_SIZE);
    if (rc != 0) {
        fprintf(stderr, "Failed to create synthetic pack fixture\n");
        return 1;
    }

    if (test_trunk_pinned_prefix_hit_rate(synth_pack) != 0) g_failures++;
    if (test_trunk_never_lru(synth_pack) != 0) g_failures++;
    if (test_trunk_prefetch_overlaps(synth_pack) != 0) g_failures++;
    if (test_plan_refuses_when_short() != 0) g_failures++;

    remove(synth_pack);

    if (g_failures > 0) {
        printf("--- %d TEST(S) FAILED ---\n", g_failures);
        return 1;
    }

    printf("--- ALL TRUNK TESTS PASSED ---\n");
    return 0;
}
