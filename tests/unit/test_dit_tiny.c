#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"
#include "model/fh3_seq.h"
#include "model/fh3_dit.h"
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
    if (sz <= 0) { fclose(fp); return NULL; }
    void *buf = malloc((size_t)sz);
    if (!buf) { fclose(fp); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) { free(buf); return NULL; }
    if (out_sz) *out_sz = (size_t)sz;
    return buf;
}

static Fh3DitCfg get_tiny_cfg(void) {
    Fh3DitCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.hidden_size = 256;
    cfg.num_heads = 2;
    cfg.head_dim = 128;
    cfg.num_layers = 3;
    cfg.num_refiner_layers = 2;
    cfg.ffn_dim = 512;
    cfg.text_dim = 64;
    cfg.time_embed_dim = 128;
    cfg.in_channels = 24;
    cfg.audio_in_channels = 32;
    cfg.patch_size[0] = 1;
    cfg.patch_size[1] = 2;
    cfg.patch_size[2] = 2;
    cfg.norm_eps = 1e-5f;
    cfg.qk_norm_eps = 1e-5f;
    cfg.final_norm_eps = 1e-5f;
    cfg.rope_freq_dim = 16;
    cfg.rope_theta = 10000.0f;
    return cfg;
}

/* -------------------------------------------------------------------------
 * Test 1: dit_tiny_step0_matches_torch
 * ------------------------------------------------------------------------- */
static int test_dit_tiny_step0_matches_torch(void) {
    printf("--- Running test_dit_tiny_step0_matches_torch ---\n");
    char pbuf[512];
    const char *globals_path = resolve_path("tests/fixtures/tiny_h3/dit_globals.bin", pbuf, sizeof(pbuf));
    char pbuf2[512];
    const char *adaln_path = resolve_path("tests/fixtures/tiny_h3/adaln_table.bin", pbuf2, sizeof(pbuf2));
    char pbuf3[512];
    const char *trunk_path = resolve_path("tests/fixtures/tiny_h3/dit_trunk.bin", pbuf3, sizeof(pbuf3));

    Fh3DitCfg cfg = get_tiny_cfg();
    Fh3Dit *dit = NULL;
    int rc = fh3_dit_open(globals_path, adaln_path, &cfg, &dit);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(dit != NULL);

    Fh3Trunk trunk;
    rc = fh3_trunk_open(trunk_path, 50 * 1024 * 1024ULL, &trunk);
    TEST_ASSERT(rc == FH3_OK);

    /* Read inputs */
    size_t sz_pos, sz_xv, sz_xa, sz_te, sz_ref_v, sz_ref_a;
    double *pos = (double *)read_file_bytes("tests/fixtures/tiny_h3/pos.bin", &sz_pos);
    float *x_v = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_video.bin", &sz_xv);
    float *x_a = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_audio.bin", &sz_xa);
    float *text_emb = (float *)read_file_bytes("tests/fixtures/tiny_h3/text_emb.bin", &sz_te);
    float *ref_v0 = (float *)read_file_bytes("tests/fixtures/tiny_h3/video_v_step0.bin", &sz_ref_v);
    float *ref_a0 = (float *)read_file_bytes("tests/fixtures/tiny_h3/audio_v_step0.bin", &sz_ref_a);
    TEST_ASSERT(pos && x_v && x_a && text_emb && ref_v0 && ref_a0);

    Fh3SeqSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.num_text = 6;
    spec.num_audio = 4;
    spec.num_video = 8;
    spec.custom_pos = pos;

    Fh3Seq seq;
    rc = fh3_seq_build(&spec, &seq);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(seq.seq_len == 18);

    /* Prepare text once */
    float text_rows[6 * 256];
    rc = fh3_dit_prepare_text(dit, text_emb, 6, text_rows);
    TEST_ASSERT(rc == FH3_OK);

    /* Forward step 0 */
    float v_video[8 * 96];
    float v_audio[4 * 32];
    rc = fh3_dit_forward(dit, &trunk, &seq, 0, x_v, x_a, text_rows, v_video, v_audio);
    TEST_ASSERT(rc == FH3_OK);

    float rel_v = compute_rel_l2(v_video, ref_v0, 8 * 96);
    float rel_a = compute_rel_l2(v_audio, ref_a0, 4 * 32);
    printf("  step 0 rel-L2: video = %e, audio = %e\n", rel_v, rel_a);
    TEST_ASSERT(rel_v <= 1e-4f);
    TEST_ASSERT(rel_a <= 1e-4f);

    free(pos); free(x_v); free(x_a); free(text_emb); free(ref_v0); free(ref_a0);
    fh3_seq_free(&seq);
    fh3_trunk_close(&trunk);
    fh3_dit_close(dit);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 2: dit_tiny_step7_matches_torch
 * ------------------------------------------------------------------------- */
static int test_dit_tiny_step7_matches_torch(void) {
    printf("--- Running test_dit_tiny_step7_matches_torch ---\n");
    char pbuf[512], pbuf2[512], pbuf3[512];
    const char *globals_path = resolve_path("tests/fixtures/tiny_h3/dit_globals.bin", pbuf, sizeof(pbuf));
    const char *adaln_path = resolve_path("tests/fixtures/tiny_h3/adaln_table.bin", pbuf2, sizeof(pbuf2));
    const char *trunk_path = resolve_path("tests/fixtures/tiny_h3/dit_trunk.bin", pbuf3, sizeof(pbuf3));

    Fh3DitCfg cfg = get_tiny_cfg();
    Fh3Dit *dit = NULL;
    int rc = fh3_dit_open(globals_path, adaln_path, &cfg, &dit);
    TEST_ASSERT(rc == FH3_OK);

    Fh3Trunk trunk;
    rc = fh3_trunk_open(trunk_path, 50 * 1024 * 1024ULL, &trunk);
    TEST_ASSERT(rc == FH3_OK);

    double *pos = (double *)read_file_bytes("tests/fixtures/tiny_h3/pos.bin", NULL);
    float *x_v = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_video.bin", NULL);
    float *x_a = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_audio.bin", NULL);
    float *text_emb = (float *)read_file_bytes("tests/fixtures/tiny_h3/text_emb.bin", NULL);
    float *ref_v7 = (float *)read_file_bytes("tests/fixtures/tiny_h3/video_v_step7.bin", NULL);
    float *ref_a7 = (float *)read_file_bytes("tests/fixtures/tiny_h3/audio_v_step7.bin", NULL);
    TEST_ASSERT(pos && x_v && x_a && text_emb && ref_v7 && ref_a7);

    Fh3SeqSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.num_text = 6;
    spec.num_audio = 4;
    spec.num_video = 8;
    spec.custom_pos = pos;

    Fh3Seq seq;
    rc = fh3_seq_build(&spec, &seq);
    TEST_ASSERT(rc == FH3_OK);

    float text_rows[6 * 256];
    rc = fh3_dit_prepare_text(dit, text_emb, 6, text_rows);
    TEST_ASSERT(rc == FH3_OK);

    float v_video[8 * 96];
    float v_audio[4 * 32];
    rc = fh3_dit_forward(dit, &trunk, &seq, 7, x_v, x_a, text_rows, v_video, v_audio);
    TEST_ASSERT(rc == FH3_OK);

    float rel_v = compute_rel_l2(v_video, ref_v7, 8 * 96);
    float rel_a = compute_rel_l2(v_audio, ref_a7, 4 * 32);
    printf("  step 7 rel-L2: video = %e, audio = %e\n", rel_v, rel_a);
    TEST_ASSERT(rel_v <= 1e-4f);
    TEST_ASSERT(rel_a <= 1e-4f);

    free(pos); free(x_v); free(x_a); free(text_emb); free(ref_v7); free(ref_a7);
    fh3_seq_free(&seq);
    fh3_trunk_close(&trunk);
    fh3_dit_close(dit);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: dit_tiny_streamed_eq_pinned
 * ------------------------------------------------------------------------- */
static int test_dit_tiny_streamed_eq_pinned(void) {
    printf("--- Running test_dit_tiny_streamed_eq_pinned ---\n");
    char pbuf[512], pbuf2[512], pbuf3[512];
    const char *globals_path = resolve_path("tests/fixtures/tiny_h3/dit_globals.bin", pbuf, sizeof(pbuf));
    const char *adaln_path = resolve_path("tests/fixtures/tiny_h3/adaln_table.bin", pbuf2, sizeof(pbuf2));
    const char *trunk_path = resolve_path("tests/fixtures/tiny_h3/dit_trunk.bin", pbuf3, sizeof(pbuf3));

    Fh3DitCfg cfg = get_tiny_cfg();
    Fh3Dit *dit = NULL;
    int rc = fh3_dit_open(globals_path, adaln_path, &cfg, &dit);
    TEST_ASSERT(rc == FH3_OK);

    /* Open pack to inspect max layer run size */
    Fh3Pack pack;
    rc = fh3_pack_open(trunk_path, &pack);
    TEST_ASSERT(rc == FH3_OK);
    uint64_t max_run = 0;
    for (int i = 0; i < pack.num_layers; i++) {
        if (pack.layers[i].size_bytes > max_run) max_run = pack.layers[i].size_bytes;
    }
    fh3_pack_close(&pack);

    /* Streamed: budget for 1 pinned layer + 2 ring slots */
    uint64_t budget_streamed = 3 * max_run;
    Fh3Trunk tr_streamed;
    rc = fh3_trunk_open(trunk_path, budget_streamed, &tr_streamed);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(tr_streamed.npin == 1); /* Assert exactly 1 pinned layer */

    /* Pinned: budget for all layers (3 pinned layers) */
    uint64_t budget_pinned = 50 * 1024 * 1024ULL;
    Fh3Trunk tr_pinned;
    rc = fh3_trunk_open(trunk_path, budget_pinned, &tr_pinned);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(tr_pinned.npin == 3); /* Assert all 3 layers pinned */

    double *pos = (double *)read_file_bytes("tests/fixtures/tiny_h3/pos.bin", NULL);
    float *x_v = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_video.bin", NULL);
    float *x_a = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_audio.bin", NULL);
    float *text_emb = (float *)read_file_bytes("tests/fixtures/tiny_h3/text_emb.bin", NULL);
    TEST_ASSERT(pos && x_v && x_a && text_emb);

    Fh3SeqSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.num_text = 6;
    spec.num_audio = 4;
    spec.num_video = 8;
    spec.custom_pos = pos;

    Fh3Seq seq;
    rc = fh3_seq_build(&spec, &seq);
    TEST_ASSERT(rc == FH3_OK);

    float text_rows[6 * 256];
    rc = fh3_dit_prepare_text(dit, text_emb, 6, text_rows);
    TEST_ASSERT(rc == FH3_OK);

    float v_v_streamed[8 * 96], v_a_streamed[4 * 32];
    float v_v_pinned[8 * 96], v_a_pinned[4 * 32];

    rc = fh3_dit_forward(dit, &tr_streamed, &seq, 0, x_v, x_a, text_rows, v_v_streamed, v_a_streamed);
    TEST_ASSERT(rc == FH3_OK);

    rc = fh3_dit_forward(dit, &tr_pinned, &seq, 0, x_v, x_a, text_rows, v_v_pinned, v_a_pinned);
    TEST_ASSERT(rc == FH3_OK);

    /* Bitwise identical check */
    TEST_ASSERT(memcmp(v_v_streamed, v_v_pinned, sizeof(v_v_streamed)) == 0);
    TEST_ASSERT(memcmp(v_a_streamed, v_a_pinned, sizeof(v_a_streamed)) == 0);
    printf("  streamed vs pinned outputs match bit-exact (memcmp == 0)\n");

    free(pos); free(x_v); free(x_a); free(text_emb);
    fh3_seq_free(&seq);
    fh3_trunk_close(&tr_streamed);
    fh3_trunk_close(&tr_pinned);
    fh3_dit_close(dit);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 4: refiner_runs_once
 * ------------------------------------------------------------------------- */
static int test_refiner_runs_once(void) {
    printf("--- Running test_refiner_runs_once ---\n");
    char pbuf[512], pbuf2[512], pbuf3[512];
    const char *globals_path = resolve_path("tests/fixtures/tiny_h3/dit_globals.bin", pbuf, sizeof(pbuf));
    const char *adaln_path = resolve_path("tests/fixtures/tiny_h3/adaln_table.bin", pbuf2, sizeof(pbuf2));
    const char *trunk_path = resolve_path("tests/fixtures/tiny_h3/dit_trunk.bin", pbuf3, sizeof(pbuf3));

    Fh3DitCfg cfg = get_tiny_cfg();
    Fh3Dit *dit = NULL;
    int rc = fh3_dit_open(globals_path, adaln_path, &cfg, &dit);
    TEST_ASSERT(rc == FH3_OK);

    Fh3Trunk trunk;
    rc = fh3_trunk_open(trunk_path, 50 * 1024 * 1024ULL, &trunk);
    TEST_ASSERT(rc == FH3_OK);

    double *pos = (double *)read_file_bytes("tests/fixtures/tiny_h3/pos.bin", NULL);
    float *x_v = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_video.bin", NULL);
    float *x_a = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_audio.bin", NULL);
    float *text_emb = (float *)read_file_bytes("tests/fixtures/tiny_h3/text_emb.bin", NULL);
    TEST_ASSERT(pos && x_v && x_a && text_emb);

    Fh3SeqSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.num_text = 6;
    spec.num_audio = 4;
    spec.num_video = 8;
    spec.custom_pos = pos;

    Fh3Seq seq;
    rc = fh3_seq_build(&spec, &seq);
    TEST_ASSERT(rc == FH3_OK);

    float text_rows[6 * 256];
    TEST_ASSERT(fh3_dit_refiner_count(dit) == 0);

    /* Prepare text once */
    rc = fh3_dit_prepare_text(dit, text_emb, 6, text_rows);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(fh3_dit_refiner_count(dit) == 1);

    /* Run all 8 steps of DiT forward */
    float v_video[8 * 96];
    float v_audio[4 * 32];
    for (int step = 0; step < 8; step++) {
        rc = fh3_dit_forward(dit, &trunk, &seq, step, x_v, x_a, text_rows, v_video, v_audio);
        TEST_ASSERT(rc == FH3_OK);
    }

    /* Refiner count must still be 1! */
    TEST_ASSERT(fh3_dit_refiner_count(dit) == 1);
    printf("  refiner run count == 1 across all 8 forward steps\n");

    free(pos); free(x_v); free(x_a); free(text_emb);
    fh3_seq_free(&seq);
    fh3_trunk_close(&trunk);
    fh3_dit_close(dit);
    return 0;
}

int main(void) {
    printf("=== Starting test_dit_tiny suite ===\n");
    test_dit_tiny_step0_matches_torch();
    test_dit_tiny_step7_matches_torch();
    test_dit_tiny_streamed_eq_pinned();
    test_refiner_runs_once();

    if (g_failures > 0) {
        fprintf(stderr, "=== test_dit_tiny FAILED with %d failure(s) ===\n", g_failures);
        return 1;
    }
    printf("=== All test_dit_tiny tests PASSED ===\n");
    return 0;
}
