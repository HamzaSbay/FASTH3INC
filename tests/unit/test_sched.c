#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"
#include "model/fh3_sched.h"
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
 * Test 1: sched_sigmas_match_ref
 * Verifies 8 video + 8 audio sigmas vs h3_ref.sampler_plan
 * ------------------------------------------------------------------------- */
static int test_sched_sigmas_match_ref(void) {
    printf("--- Running test_sched_sigmas_match_ref ---\n");

    const int dmd[8] = {999, 874, 749, 624, 500, 375, 250, 125};
    Fh3Sched s;
    int rc = fh3_sched_build(&s, dmd, 10.0, 3.0);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(s.n_steps == 8);
    TEST_ASSERT(fabs(s.video_shift - 10.0) < 1e-12);
    TEST_ASSERT(fabs(s.audio_shift - 3.0) < 1e-12);

    /* Reference sigmas computed via float64: (shift * base) / (1 + (shift - 1) * base) */
    const float expected_v[9] = {
        0.9998999238014221f,
        0.9857884049415588f,
        0.9675752520561218f,
        0.9431681036949158f,
        0.9090909361839294f,
        0.8571428656578064f,
        0.7692307829856873f,
        0.5882353186607361f,
        0.0f
    };

    const float expected_a[9] = {
        0.9996664524078369f,
        0.9541484713554382f,
        0.8995196223258972f,
        0.8327401876449585f,
        0.75f,
        0.6428571343421936f,
        0.5f,
        0.30000001192092896f,
        0.0f
    };

    for (int i = 0; i < 9; i++) {
        TEST_ASSERT(s.video_sigmas[i] == expected_v[i]);
        TEST_ASSERT(s.audio_sigmas[i] == expected_a[i]);
    }

    for (int i = 0; i < 8; i++) {
        TEST_ASSERT(s.steps[i].sigma_v == expected_v[i]);
        TEST_ASSERT(s.steps[i].sigma_v_next == expected_v[i + 1]);
        TEST_ASSERT(s.steps[i].sigma_a == expected_a[i]);
        TEST_ASSERT(s.steps[i].sigma_a_next == expected_a[i + 1]);
        TEST_ASSERT(fabsf(s.steps[i].t_v - (1.0f - expected_v[i])) < 1e-6f);
        TEST_ASSERT(fabsf(s.steps[i].t_a - (1.0f - expected_a[i])) < 1e-6f);
    }

    /* Terminal sigmas are identically 0.0f */
    TEST_ASSERT(s.steps[7].sigma_v_next == 0.0f);
    TEST_ASSERT(s.steps[7].sigma_a_next == 0.0f);
    TEST_ASSERT(s.video_sigmas[8] == 0.0f);
    TEST_ASSERT(s.audio_sigmas[8] == 0.0f);

    /* Test default parameters when dmd is NULL and shifts <= 0 */
    Fh3Sched s_default;
    rc = fh3_sched_build(&s_default, NULL, 0.0, 0.0);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(s_default.n_steps == 8);
    for (int i = 0; i < 9; i++) {
        TEST_ASSERT(s_default.video_sigmas[i] == expected_v[i]);
        TEST_ASSERT(s_default.audio_sigmas[i] == expected_a[i]);
    }

    printf("  Sigmas bit-exact match reference across all 8+1 steps.\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 2: sched_last_step_returns_x0
 * Verifies that on last step, x_next = x0 identically
 * ------------------------------------------------------------------------- */
static int test_sched_last_step_returns_x0(void) {
    printf("--- Running test_sched_last_step_returns_x0 ---\n");

    Fh3Sched s;
    int rc = fh3_sched_build(&s, NULL, 10.0, 3.0);
    TEST_ASSERT(rc == FH3_OK);

#define LAST_STEP_N 64
    float x[LAST_STEP_N], v[LAST_STEP_N], noise[LAST_STEP_N];
    for (int j = 0; j < LAST_STEP_N; j++) {
        x[j] = (float)(j + 1) * 0.1f;
        v[j] = (float)(LAST_STEP_N - j) * 0.05f;
        noise[j] = 999.0f; /* Arbitrary dummy noise */
    }

    /* 1. Video modality (0) with noise_next != NULL on last step (i = 7) */
    float x_work[LAST_STEP_N];
    memcpy(x_work, x, sizeof(x));
    fh3_sched_step(&s, 7, x_work, v, LAST_STEP_N, 0, noise);

    float sigma_v7 = s.steps[7].sigma_v;
    for (int j = 0; j < LAST_STEP_N; j++) {
        float expected_x0 = x[j] + sigma_v7 * v[j];
        TEST_ASSERT(x_work[j] == expected_x0);
    }

    /* 2. Video modality (0) with noise_next == NULL (Euler) on last step (i = 7) */
    memcpy(x_work, x, sizeof(x));
    fh3_sched_step(&s, 7, x_work, v, LAST_STEP_N, 0, NULL);
    for (int j = 0; j < LAST_STEP_N; j++) {
        float expected_x0 = x[j] + sigma_v7 * v[j];
        TEST_ASSERT(x_work[j] == expected_x0);
    }

    /* 3. Audio modality (2) on last step (i = 7) */
    memcpy(x_work, x, sizeof(x));
    fh3_sched_step(&s, 7, x_work, v, LAST_STEP_N, 2, noise);
    float sigma_a7 = s.steps[7].sigma_a;
    for (int j = 0; j < LAST_STEP_N; j++) {
        float expected_x0 = x[j] + sigma_a7 * v[j];
        TEST_ASSERT(x_work[j] == expected_x0);
    }

    printf("  Last step returns x0 bit-identically for both Euler and stochastic renoising.\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: philox_rng_normal
 * Verifies Philox4x32-10 normal RNG generator
 * ------------------------------------------------------------------------- */
static int test_philox_rng_normal(void) {
    printf("--- Running test_philox_rng_normal ---\n");

    const size_t N = 20000;
    float *buf1 = (float *)malloc(N * sizeof(float));
    float *buf2 = (float *)malloc(N * sizeof(float));
    float *buf3 = (float *)malloc(N * sizeof(float));
    TEST_ASSERT(buf1 && buf2 && buf3);

    /* 1. Determinism: same seed and stream produce bit-identical output */
    fh3_rng_normal(12345ULL, 0ULL, buf1, N);
    fh3_rng_normal(12345ULL, 0ULL, buf2, N);
    TEST_ASSERT(memcmp(buf1, buf2, N * sizeof(float)) == 0);

    /* 2. Stream separation: different stream produces different values */
    fh3_rng_normal(12345ULL, 1ULL, buf3, N);
    TEST_ASSERT(memcmp(buf1, buf3, N * sizeof(float)) != 0);

    /* 3. Seed separation */
    fh3_rng_normal(54321ULL, 0ULL, buf3, N);
    TEST_ASSERT(memcmp(buf1, buf3, N * sizeof(float)) != 0);

    /* 4. Statistical properties: mean ~ 0, variance ~ 1 */
    double sum = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += buf1[i];
        sum_sq += (double)buf1[i] * (double)buf1[i];
    }
    double mean = sum / (double)N;
    double var = (sum_sq / (double)N) - (mean * mean);

    printf("  Philox RNG N=%zu: mean = %e, variance = %f\n", N, mean, var);
    TEST_ASSERT(fabs(mean) < 0.03);
    TEST_ASSERT(fabs(var - 1.0) < 0.03);

    free(buf1);
    free(buf2);
    free(buf3);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 4: load_noise_helper
 * ------------------------------------------------------------------------- */
static int test_load_noise_helper(void) {
    printf("--- Running test_load_noise_helper ---\n");

    char pbuf[512];
    const char *noise_path = resolve_path("tests/fixtures/tiny_h3/noise_v_step0.bin", pbuf, sizeof(pbuf));

    float noise[8 * 96];
    int rc = fh3_load_noise(noise_path, noise, 8 * 96);
    TEST_ASSERT(rc == FH3_OK);

    /* Non-existent file should return error */
    rc = fh3_load_noise("non_existent_noise_file_12345.bin", noise, 8 * 96);
    TEST_ASSERT(rc != FH3_OK);

    /* NULL args */
    TEST_ASSERT(fh3_load_noise(NULL, noise, 8 * 96) == FH3_E_INVALID);
    TEST_ASSERT(fh3_load_noise(noise_path, NULL, 8 * 96) == FH3_E_INVALID);

    return 0;
}

/* -------------------------------------------------------------------------
 * Test 5: tiny_denoise_8_steps_matches_torch
 * Full 8-step denoising loop parity with tiny model and reference noise
 * ------------------------------------------------------------------------- */
static int test_tiny_denoise_8_steps_matches_torch(void) {
    printf("--- Running test_tiny_denoise_8_steps_matches_torch ---\n");

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

    /* Read initial inputs */
    double *pos = (double *)read_file_bytes("tests/fixtures/tiny_h3/pos.bin", NULL);
    float *x_v = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_video.bin", NULL);
    float *x_a = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_audio.bin", NULL);
    float *text_emb = (float *)read_file_bytes("tests/fixtures/tiny_h3/text_emb.bin", NULL);
    TEST_ASSERT(pos && x_v && x_a && text_emb);

    /* Read reference final outputs */
    float *ref_final_v = (float *)read_file_bytes("tests/fixtures/tiny_h3/final_video_latents.bin", NULL);
    float *ref_final_a = (float *)read_file_bytes("tests/fixtures/tiny_h3/final_audio_latents.bin", NULL);
    TEST_ASSERT(ref_final_v && ref_final_a);

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

    Fh3Sched sched;
    rc = fh3_sched_build(&sched, NULL, 10.0, 3.0);
    TEST_ASSERT(rc == FH3_OK);

    const int num_v_floats = 8 * 96;
    const int num_a_floats = 4 * 32;
    float v_video[8 * 96];
    float v_audio[4 * 32];
    float noise_v[8 * 96];
    float noise_a[4 * 32];

    /* Run full 8-step denoising loop with stochastic renoising */
    for (int step = 0; step < 8; step++) {
        /* 1. DiT model forward */
        rc = fh3_dit_forward(dit, &trunk, &seq, step, x_v, x_a, text_rows, v_video, v_audio);
        TEST_ASSERT(rc == FH3_OK);

        /* 2. Load fresh noise for step < 7 */
        if (step < 7) {
            char nv_path[512], na_path[512];
            snprintf(nv_path, sizeof(nv_path), "tests/fixtures/tiny_h3/noise_v_step%d.bin", step);
            snprintf(na_path, sizeof(na_path), "tests/fixtures/tiny_h3/noise_a_step%d.bin", step);
            char p_nv[512], p_na[512];
            rc = fh3_load_noise(resolve_path(nv_path, p_nv, sizeof(p_nv)), noise_v, num_v_floats);
            TEST_ASSERT(rc == FH3_OK);
            rc = fh3_load_noise(resolve_path(na_path, p_na, sizeof(p_na)), noise_a, num_a_floats);
            TEST_ASSERT(rc == FH3_OK);

            fh3_sched_step(&sched, step, x_v, v_video, num_v_floats, 0, noise_v);
            fh3_sched_step(&sched, step, x_a, v_audio, num_a_floats, 2, noise_a);
        } else {
            /* Terminal step: no noise needed */
            fh3_sched_step(&sched, step, x_v, v_video, num_v_floats, 0, NULL);
            fh3_sched_step(&sched, step, x_a, v_audio, num_a_floats, 2, NULL);
        }
    }

    /* Compare final latents against reference */
    float rel_v = compute_rel_l2(x_v, ref_final_v, num_v_floats);
    float rel_a = compute_rel_l2(x_a, ref_final_a, num_a_floats);
    printf("  8-step stochastic final rel-L2: video = %e, audio = %e\n", rel_v, rel_a);
    TEST_ASSERT(rel_v <= 1e-4f);
    TEST_ASSERT(rel_a <= 1e-4f);

    free(pos); free(x_v); free(x_a); free(text_emb);
    free(ref_final_v); free(ref_final_a);
    fh3_seq_free(&seq);
    fh3_trunk_close(&trunk);
    fh3_dit_close(dit);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 6: tiny_denoise_8_steps_euler_matches_torch
 * Full 8-step deterministic Euler loop parity with tiny model
 * ------------------------------------------------------------------------- */
static int test_tiny_denoise_8_steps_euler_matches_torch(void) {
    printf("--- Running test_tiny_denoise_8_steps_euler_matches_torch ---\n");

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

    float *ref_final_v = (float *)read_file_bytes("tests/fixtures/tiny_h3/euler_final_video_latents.bin", NULL);
    float *ref_final_a = (float *)read_file_bytes("tests/fixtures/tiny_h3/euler_final_audio_latents.bin", NULL);
    TEST_ASSERT(ref_final_v && ref_final_a);

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

    Fh3Sched sched;
    rc = fh3_sched_build(&sched, NULL, 10.0, 3.0);
    TEST_ASSERT(rc == FH3_OK);

    const int num_v_floats = 8 * 96;
    const int num_a_floats = 4 * 32;
    float v_video[8 * 96];
    float v_audio[4 * 32];

    /* Run full 8-step deterministic Euler loop */
    for (int step = 0; step < 8; step++) {
        rc = fh3_dit_forward(dit, &trunk, &seq, step, x_v, x_a, text_rows, v_video, v_audio);
        TEST_ASSERT(rc == FH3_OK);

        fh3_sched_step(&sched, step, x_v, v_video, num_v_floats, 0, NULL);
        fh3_sched_step(&sched, step, x_a, v_audio, num_a_floats, 2, NULL);
    }

    float rel_v = compute_rel_l2(x_v, ref_final_v, num_v_floats);
    float rel_a = compute_rel_l2(x_a, ref_final_a, num_a_floats);
    printf("  8-step Euler final rel-L2: video = %e, audio = %e\n", rel_v, rel_a);
    TEST_ASSERT(rel_v <= 1e-4f);
    TEST_ASSERT(rel_a <= 1e-4f);

    free(pos); free(x_v); free(x_a); free(text_emb);
    free(ref_final_v); free(ref_final_a);
    fh3_seq_free(&seq);
    fh3_trunk_close(&trunk);
    fh3_dit_close(dit);
    return 0;
}

int main(void) {
    printf("=== Starting test_sched suite ===\n");
    test_sched_sigmas_match_ref();
    test_sched_last_step_returns_x0();
    test_philox_rng_normal();
    test_load_noise_helper();
    test_tiny_denoise_8_steps_matches_torch();
    test_tiny_denoise_8_steps_euler_matches_torch();

    if (g_failures > 0) {
        fprintf(stderr, "=== test_sched FAILED with %d failure(s) ===\n", g_failures);
        return 1;
    }
    printf("=== All test_sched tests PASSED ===\n");
    return 0;
}
