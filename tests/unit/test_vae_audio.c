#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"
#include "vae/fh3_vae_audio.h"
#include "out/fh3_wav.h"

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

/* -------------------------------------------------------------------------
 * Test 1: wav_header_valid
 * ------------------------------------------------------------------------- */
static int test_wav_header_valid(void) {
    printf("[RUN] wav_header_valid\n");
    const int ch = 2;
    const int64_t n = 1000;
    const int rate = 32000;
    float *pcm = (float *)malloc((size_t)(ch * n) * sizeof(float));
    TEST_ASSERT(pcm != NULL);

    for (int64_t i = 0; i < n; i++) {
        pcm[i] = 0.5f * (float)sin(2.0 * 3.141592653589793 * 440.0 * (double)i / 32000.0);
        pcm[n + i] = 0.5f * (float)cos(2.0 * 3.141592653589793 * 880.0 * (double)i / 32000.0);
    }

    const char *wav_path = "test_audio_out.wav";
    int rc = fh3_wav_write(wav_path, pcm, ch, n, rate);
    TEST_ASSERT(rc == FH3_OK);

    FILE *fp = fopen(wav_path, "rb");
    TEST_ASSERT(fp != NULL);
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint32_t expected_data_bytes = (uint32_t)(ch * n * 2);
    uint32_t expected_total_bytes = 44 + expected_data_bytes;
    TEST_ASSERT((uint32_t)sz == expected_total_bytes);

    uint8_t hdr[44];
    size_t rd = fread(hdr, 1, 44, fp);
    TEST_ASSERT(rd == 44);

    /* Verify RIFF */
    TEST_ASSERT(memcmp(hdr + 0, "RIFF", 4) == 0);
    uint32_t riff_sz = (uint32_t)hdr[4] | ((uint32_t)hdr[5] << 8) |
                       ((uint32_t)hdr[6] << 16) | ((uint32_t)hdr[7] << 24);
    TEST_ASSERT(riff_sz == expected_total_bytes - 8);
    TEST_ASSERT(memcmp(hdr + 8, "WAVE", 4) == 0);

    /* Verify fmt */
    TEST_ASSERT(memcmp(hdr + 12, "fmt ", 4) == 0);
    uint32_t fmt_sz = (uint32_t)hdr[16] | ((uint32_t)hdr[17] << 8) |
                      ((uint32_t)hdr[18] << 16) | ((uint32_t)hdr[19] << 24);
    TEST_ASSERT(fmt_sz == 16);
    uint16_t audio_fmt = (uint16_t)hdr[20] | ((uint16_t)hdr[21] << 8);
    TEST_ASSERT(audio_fmt == 1);
    uint16_t n_ch = (uint16_t)hdr[22] | ((uint16_t)hdr[23] << 8);
    TEST_ASSERT(n_ch == (uint16_t)ch);
    uint32_t s_rate = (uint32_t)hdr[24] | ((uint32_t)hdr[25] << 8) |
                      ((uint32_t)hdr[26] << 16) | ((uint32_t)hdr[27] << 24);
    TEST_ASSERT(s_rate == (uint32_t)rate);
    uint32_t byte_rate = (uint32_t)hdr[28] | ((uint32_t)hdr[29] << 8) |
                         ((uint32_t)hdr[30] << 16) | ((uint32_t)hdr[31] << 24);
    TEST_ASSERT(byte_rate == (uint32_t)rate * (uint32_t)ch * 2);
    uint16_t block_align = (uint16_t)hdr[32] | ((uint16_t)hdr[33] << 8);
    TEST_ASSERT(block_align == (uint16_t)(ch * 2));
    uint16_t bits = (uint16_t)hdr[34] | ((uint16_t)hdr[35] << 8);
    TEST_ASSERT(bits == 16);

    /* Verify data */
    TEST_ASSERT(memcmp(hdr + 36, "data", 4) == 0);
    uint32_t data_sz = (uint32_t)hdr[40] | ((uint32_t)hdr[41] << 8) |
                       ((uint32_t)hdr[42] << 16) | ((uint32_t)hdr[43] << 24);
    TEST_ASSERT(data_sz == expected_data_bytes);

    /* Verify sample values (interleaved Left, Right) */
    int16_t sample_pair[2];
    rd = fread(sample_pair, sizeof(int16_t), 2, fp);
    TEST_ASSERT(rd == 2);
    int16_t expected_l0 = (int16_t)lrintf(pcm[0] * 32767.0f);
    int16_t expected_r0 = (int16_t)lrintf(pcm[n] * 32767.0f);
    TEST_ASSERT(sample_pair[0] == expected_l0);
    TEST_ASSERT(sample_pair[1] == expected_r0);

    fclose(fp);
    remove(wav_path);
    free(pcm);

    printf("[PASS] wav_header_valid\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 2: vae_audio_tiny_matches_diffusers
 * ------------------------------------------------------------------------- */
static int test_vae_audio_tiny_matches_diffusers(void) {
    printf("[RUN] vae_audio_tiny_matches_diffusers\n");

    char pbuf[512];
    const char *bin_path = resolve_path("tests/fixtures/tiny_vae_audio/vae_audio_dec.bin", pbuf, sizeof(pbuf));
    FILE *chk = fopen(bin_path, "rb");
    if (!chk) {
        /* Generate tiny fixture on demand if missing */
        printf("Generating tiny audio VAE fixture...\n");
        int sys_rc = system(".venv\\Scripts\\python tools\\pack_vae.py --tiny");
        (void)sys_rc;
        bin_path = resolve_path("tests/fixtures/tiny_vae_audio/vae_audio_dec.bin", pbuf, sizeof(pbuf));
    } else {
        fclose(chk);
    }

    Fh3VaeA *m = NULL;
    int rc = fh3_vae_audio_open(bin_path, &m);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(m != NULL);

    int upsample_ratio = fh3_vae_audio_upsample_ratio(m);
    TEST_ASSERT(upsample_ratio == 4);

    size_t lat_bytes = 0;
    float *lat = (float *)read_file_bytes("tests/fixtures/tiny_vae_audio/latents.bin", &lat_bytes);
    TEST_ASSERT(lat != NULL);
    /* 2 channels x 4 frames x 32 latent_channels x 4 bytes = 1024 bytes */
    TEST_ASSERT(lat_bytes == 2 * 4 * 32 * sizeof(float));

    size_t golden_bytes = 0;
    float *golden_pcm = (float *)read_file_bytes("tests/fixtures/tiny_vae_audio/golden_pcm.bin", &golden_bytes);
    TEST_ASSERT(golden_pcm != NULL);
    /* 2 channels x 16 samples x 4 bytes = 128 bytes */
    TEST_ASSERT(golden_bytes == 2 * 16 * sizeof(float));

    const int L = 4;
    float pcm_out[2 * 16];
    memset(pcm_out, 0, sizeof(pcm_out));

    rc = fh3_vae_audio_decode(m, lat, L, pcm_out);
    TEST_ASSERT(rc == FH3_OK);

    printf("pcm_out ch0[:5]:   %f %f %f %f %f\n",
           pcm_out[0], pcm_out[1], pcm_out[2], pcm_out[3], pcm_out[4]);
    printf("golden_pcm ch0[:5]:%f %f %f %f %f\n",
           golden_pcm[0], golden_pcm[1], golden_pcm[2], golden_pcm[3], golden_pcm[4]);

    float rel_l2 = compute_rel_l2(pcm_out, golden_pcm, 2 * 16);
    printf("Tiny VAE audio diffusers parity rel-L2: %e\n", (double)rel_l2);
    TEST_ASSERT(rel_l2 <= 1e-4f);

    fh3_vae_audio_close(m);
    free(lat);
    free(golden_pcm);

    printf("[PASS] vae_audio_tiny_matches_diffusers\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: vae_audio_real_matches_diffusers
 * ------------------------------------------------------------------------- */
static int test_vae_audio_real_matches_diffusers(void) {
    printf("[RUN] vae_audio_real_matches_diffusers\n");

    char pbuf[512];
    const char *real_path = resolve_path("packed/vae_audio_dec.bin", pbuf, sizeof(pbuf));
    FILE *fp = fopen(real_path, "rb");
    if (!fp) {
        printf("[SKIP] packed/vae_audio_dec.bin not present; skipping real audio VAE test\n");
        return 0;
    }
    fclose(fp);

    Fh3VaeA *m = NULL;
    int rc = fh3_vae_audio_open(real_path, &m);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(m != NULL);
    TEST_ASSERT(fh3_vae_audio_upsample_ratio(m) == 800);

    /* Decode L=2 frames to verify pipeline runs cleanly on real weights */
    const int L = 2;
    float *lat = (float *)calloc((size_t)(2 * L * 32), sizeof(float));
    float *pcm = (float *)calloc((size_t)(2 * L * 800), sizeof(float));
    TEST_ASSERT(lat != NULL && pcm != NULL);

    rc = fh3_vae_audio_decode(m, lat, L, pcm);
    TEST_ASSERT(rc == FH3_OK);

    free(lat);
    free(pcm);
    fh3_vae_audio_close(m);

    printf("[PASS] vae_audio_real_matches_diffusers\n");
    return 0;
}

int main(void) {
    printf("=== Audio VAE & WAV Unit Tests ===\n");
    test_wav_header_valid();
    test_vae_audio_tiny_matches_diffusers();
    test_vae_audio_real_matches_diffusers();

    if (g_failures == 0) {
        printf("=== All Audio VAE Tests Passed ===\n");
        return 0;
    } else {
        printf("=== %d Test Failures ===\n", g_failures);
        return 1;
    }
}
