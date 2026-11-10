#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"
#include "vae/fh3_vae_video.h"
#include "out/fh3_y4m.h"

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
 * Test 1: y4m_header_and_frame_valid
 * ------------------------------------------------------------------------- */
static int test_y4m_header_and_frame_valid(void) {
    printf("[RUN] y4m_header_and_frame_valid\n");
    const int w = 4;
    const int h = 4;
    const int fps_num = 24;
    const int fps_den = 1;
    const char *out_path = "test_video_out.y4m";

    Fh3Y4m *y = NULL;
    int rc = fh3_y4m_open(out_path, w, h, fps_num, fps_den, &y);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(y != NULL);

    /* Frame 1: pure white RGB (1, 1, 1) */
    float white[3 * 4 * 4];
    for (int i = 0; i < 3 * 4 * 4; i++) white[i] = 1.0f;
    rc = fh3_y4m_write_rgb(y, white);
    TEST_ASSERT(rc == FH3_OK);

    /* Frame 2: pure black RGB (0, 0, 0) */
    float black[3 * 4 * 4];
    for (int i = 0; i < 3 * 4 * 4; i++) black[i] = 0.0f;
    rc = fh3_y4m_write_rgb(y, black);
    TEST_ASSERT(rc == FH3_OK);

    /* Frame 3: 50% gray RGB (0.5, 0.5, 0.5) */
    float gray[3 * 4 * 4];
    for (int i = 0; i < 3 * 4 * 4; i++) gray[i] = 0.5f;
    rc = fh3_y4m_write_rgb(y, gray);
    TEST_ASSERT(rc == FH3_OK);

    fh3_y4m_close(y);

    /* Verify file on disk */
    FILE *fp = fopen(out_path, "rb");
    TEST_ASSERT(fp != NULL);
    fseek(fp, 0, SEEK_END);
    long file_sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    /* Header: "YUV4MPEG2 W4 H4 F24:1 Ip A1:1 C420jpeg\n" = 39 bytes */
    char hdr_buf[64];
    size_t rd = fread(hdr_buf, 1, 39, fp);
    TEST_ASSERT(rd == 39);
    TEST_ASSERT(memcmp(hdr_buf, "YUV4MPEG2 W4 H4 F24:1 Ip A1:1 C420jpeg\n", 39) == 0);

    /* Per frame size: 6 ("FRAME\n") + 16 (Y) + 4 (U) + 4 (V) = 30 bytes */
    long expected_total_sz = 39 + 3 * 30;
    TEST_ASSERT(file_sz == expected_total_sz);

    /* Frame 1 check: white -> Y=255, U=128, V=128 */
    char frame_hdr[6];
    rd = fread(frame_hdr, 1, 6, fp);
    TEST_ASSERT(rd == 6 && memcmp(frame_hdr, "FRAME\n", 6) == 0);
    uint8_t f1_y[16], f1_u[4], f1_v[4];
    rd = fread(f1_y, 1, 16, fp); TEST_ASSERT(rd == 16);
    rd = fread(f1_u, 1, 4, fp);  TEST_ASSERT(rd == 4);
    rd = fread(f1_v, 1, 4, fp);  TEST_ASSERT(rd == 4);
    for (int i = 0; i < 16; i++) TEST_ASSERT(f1_y[i] == 255);
    for (int i = 0; i < 4; i++)  TEST_ASSERT(f1_u[i] == 128);
    for (int i = 0; i < 4; i++)  TEST_ASSERT(f1_v[i] == 128);

    /* Frame 2 check: black -> Y=0, U=128, V=128 */
    rd = fread(frame_hdr, 1, 6, fp);
    TEST_ASSERT(rd == 6 && memcmp(frame_hdr, "FRAME\n", 6) == 0);
    uint8_t f2_y[16], f2_u[4], f2_v[4];
    rd = fread(f2_y, 1, 16, fp); TEST_ASSERT(rd == 16);
    rd = fread(f2_u, 1, 4, fp);  TEST_ASSERT(rd == 4);
    rd = fread(f2_v, 1, 4, fp);  TEST_ASSERT(rd == 4);
    for (int i = 0; i < 16; i++) TEST_ASSERT(f2_y[i] == 0);
    for (int i = 0; i < 4; i++)  TEST_ASSERT(f2_u[i] == 128);
    for (int i = 0; i < 4; i++)  TEST_ASSERT(f2_v[i] == 128);

    fclose(fp);
    remove(out_path);

    printf("[PASS] y4m_header_and_frame_valid\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Memory frame sink collector for tests
 * ------------------------------------------------------------------------- */
typedef struct {
    float *buffer;
    int capacity_frames;
    int frames_written;
    int width;
    int height;
} MemFrameSink;

static int mem_sink_write(void *user, int frame_idx, const float *rgb, int width, int height) {
    MemFrameSink *s = (MemFrameSink *)user;
    if (frame_idx != s->frames_written) return FH3_ERR;
    if (s->frames_written >= s->capacity_frames) return FH3_ERR;

    s->width = width;
    s->height = height;
    size_t frame_sz = (size_t)3 * (size_t)width * (size_t)height;
    memcpy(s->buffer + (size_t)frame_idx * frame_sz, rgb, frame_sz * sizeof(float));
    s->frames_written++;
    return FH3_OK;
}

/* -------------------------------------------------------------------------
 * Test 2: vae_video_tiny_matches_diffusers
 * ------------------------------------------------------------------------- */
static int test_vae_video_tiny_matches_diffusers(void) {
    printf("[RUN] vae_video_tiny_matches_diffusers\n");

    char pbuf[512];
    const char *bin_path = resolve_path("tests/fixtures/tiny_vae_video/vae_video_dec.bin", pbuf, sizeof(pbuf));
    FILE *chk = fopen(bin_path, "rb");
    if (!chk) {
        printf("Generating tiny video VAE fixture...\n");
        int sys_rc = system(".venv\\Scripts\\python tools\\pack_video_vae.py --tiny");
        (void)sys_rc;
        bin_path = resolve_path("tests/fixtures/tiny_vae_video/vae_video_dec.bin", pbuf, sizeof(pbuf));
        chk = fopen(bin_path, "rb");
    }
    TEST_ASSERT(chk != NULL);
    fclose(chk);

    size_t lat_sz = 0;
    float *lat = (float *)read_file_bytes("tests/fixtures/tiny_vae_video/latents.bin", &lat_sz);
    TEST_ASSERT(lat != NULL);
    TEST_ASSERT(lat_sz == 24 * 7 * 2 * 2 * sizeof(float));

    size_t golden_sz = 0;
    float *golden = (float *)read_file_bytes("tests/fixtures/tiny_vae_video/golden_frames.bin", &golden_sz);
    TEST_ASSERT(golden != NULL);
    TEST_ASSERT(golden_sz == 3 * 22 * 32 * 32 * sizeof(float));

    Fh3VaeV *m = NULL;
    int rc = fh3_vae_video_open(bin_path, &m);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(m != NULL);

    int total_frames = 22;
    int width = 32;
    int height = 32;
    size_t total_floats = (size_t)3 * (size_t)total_frames * (size_t)width * (size_t)height;
    float *decoded = (float *)malloc(total_floats * sizeof(float));
    TEST_ASSERT(decoded != NULL);

    MemFrameSink mem_sink = {
        .buffer = decoded,
        .capacity_frames = total_frames,
        .frames_written = 0,
        .width = width,
        .height = height,
    };
    Fh3FrameSink sink = {
        .user = &mem_sink,
        .write_frame = mem_sink_write,
    };

    rc = fh3_vae_video_decode(m, lat, 7, 2, 2, &sink);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(mem_sink.frames_written == 22);

    /* The golden frames are planar [3, T, H, W] in golden_frames.bin.
     * mem_sink stores frame-by-frame: [T, 3, H, W].
     * Let's permute decoded into planar [3, T, H, W] for direct parity check.
     */
    float *decoded_planar = (float *)malloc(total_floats * sizeof(float));
    TEST_ASSERT(decoded_planar != NULL);
    size_t hw = (size_t)width * (size_t)height;

    for (int t = 0; t < total_frames; t++) {
        const float *frame_t = decoded + (size_t)t * (3 * hw);
        for (int c = 0; c < 3; c++) {
            const float *src_c = frame_t + (size_t)c * hw;
            float *dst_c = decoded_planar + ((size_t)c * (size_t)total_frames + (size_t)t) * hw;
            memcpy(dst_c, src_c, hw * sizeof(float));
        }
    }

    float rel_l2 = compute_rel_l2(decoded_planar, golden, total_floats);
    printf("  tiny video VAE vs diffusers rel-L2: %e\n", rel_l2);
    TEST_ASSERT(rel_l2 <= 1e-4f);

    fh3_vae_video_close(m);
    free(lat);
    free(golden);
    free(decoded);
    free(decoded_planar);

    printf("[PASS] vae_video_tiny_matches_diffusers\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: vae_video_chunk_boundaries
 * ------------------------------------------------------------------------- */
static int test_vae_video_chunk_boundaries(void) {
    printf("[RUN] vae_video_chunk_boundaries\n");

    char pbuf[512];
    const char *bin_path = resolve_path("tests/fixtures/tiny_vae_video/vae_video_dec.bin", pbuf, sizeof(pbuf));
    Fh3VaeV *m = NULL;
    int rc = fh3_vae_video_open(bin_path, &m);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(m != NULL);

    /* F = 12 latent frames must produce exactly 17*2 + 5 = 39 pixel frames */
    const int F = 12;
    const int h = 2;
    const int w = 2;
    size_t lat_n = (size_t)24 * (size_t)F * (size_t)(h * w);
    float *lat = (float *)malloc(lat_n * sizeof(float));
    TEST_ASSERT(lat != NULL);

    /* Fill with deterministic smooth wave pattern */
    for (size_t i = 0; i < lat_n; i++) {
        lat[i] = 0.5f * (float)sin(2.0 * M_PI * (double)i / 37.0);
    }

    int expected_frames = 39;
    int patch_s = fh3_vae_video_patch_size(m);
    int pixel_w = w * patch_s;
    int pixel_h = h * patch_s;
    size_t frame_floats = (size_t)3 * (size_t)pixel_w * (size_t)pixel_h;
    float *all_frames = (float *)malloc((size_t)expected_frames * frame_floats * sizeof(float));
    TEST_ASSERT(all_frames != NULL);

    MemFrameSink mem_sink = {
        .buffer = all_frames,
        .capacity_frames = expected_frames,
        .frames_written = 0,
        .width = pixel_w,
        .height = pixel_h,
    };
    Fh3FrameSink sink = {
        .user = &mem_sink,
        .write_frame = mem_sink_write,
    };

    rc = fh3_vae_video_decode(m, lat, F, h, w, &sink);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(mem_sink.frames_written == expected_frames);

    /* Verify no NaNs, no Infs, all in [0, 1] */
    for (size_t i = 0; i < (size_t)expected_frames * frame_floats; i++) {
        float val = all_frames[i];
        TEST_ASSERT(!isnan(val));
        TEST_ASSERT(!isinf(val));
        TEST_ASSERT(val >= 0.0f && val <= 1.0f);
    }

    /* Check boundary continuity: frame 16 (chunk 0 tail) and frame 17 (chunk 1 head)
     * should not have a discontinuous jump in mean intensity
     */
    double mean_f16 = 0.0;
    double mean_f17 = 0.0;
    const float *f16 = all_frames + 16 * frame_floats;
    const float *f17 = all_frames + 17 * frame_floats;
    for (size_t i = 0; i < frame_floats; i++) {
        mean_f16 += (double)f16[i];
        mean_f17 += (double)f17[i];
    }
    mean_f16 /= (double)frame_floats;
    mean_f17 /= (double)frame_floats;
    double seam_diff = fabs(mean_f16 - mean_f17);
    printf("  chunk boundary seam difference (f16 vs f17 mean): %e\n", seam_diff);
    TEST_ASSERT(seam_diff < 0.25);

    fh3_vae_video_close(m);
    free(lat);
    free(all_frames);

    printf("[PASS] vae_video_chunk_boundaries\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 4: vae_video_real_one_clip
 * ------------------------------------------------------------------------- */
static int test_vae_video_real_one_clip(void) {
    printf("[RUN] vae_video_real_one_clip\n");

    char pbuf[512];
    const char *bin_path = resolve_path("packed/vae_video_dec.bin", pbuf, sizeof(pbuf));
    FILE *chk = fopen(bin_path, "rb");
    if (!chk) {
        printf("[SKIP] packed/vae_video_dec.bin not present\n");
        return 0;
    }
    fclose(chk);

    Fh3VaeV *m = NULL;
    int rc = fh3_vae_video_open(bin_path, &m);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(m != NULL);

    /* Run 1 clip: F=7, h=2, w=2 */
    const int F = 7;
    const int h = 2;
    const int w = 2;
    size_t lat_n = (size_t)24 * (size_t)F * (size_t)(h * w);
    float *lat = (float *)calloc(lat_n, sizeof(float));
    TEST_ASSERT(lat != NULL);

    int expected_frames = 22;
    int patch_s = fh3_vae_video_patch_size(m);
    int pixel_w = w * patch_s;
    int pixel_h = h * patch_s;
    size_t frame_floats = (size_t)3 * (size_t)pixel_w * (size_t)pixel_h;
    float *all_frames = (float *)malloc((size_t)expected_frames * frame_floats * sizeof(float));
    TEST_ASSERT(all_frames != NULL);

    MemFrameSink mem_sink = {
        .buffer = all_frames,
        .capacity_frames = expected_frames,
        .frames_written = 0,
        .width = pixel_w,
        .height = pixel_h,
    };
    Fh3FrameSink sink = {
        .user = &mem_sink,
        .write_frame = mem_sink_write,
    };

    rc = fh3_vae_video_decode(m, lat, F, h, w, &sink);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(mem_sink.frames_written == expected_frames);

    fh3_vae_video_close(m);
    free(lat);
    free(all_frames);

    printf("[PASS] vae_video_real_one_clip\n");
    return 0;
}

int main(void) {
    printf("=== FastH3 Video ViT VAE & Y4M Unit Tests ===\n");

    if (test_y4m_header_and_frame_valid() != 0) return 1;
    if (test_vae_video_tiny_matches_diffusers() != 0) return 1;
    if (test_vae_video_chunk_boundaries() != 0) return 1;
    if (test_vae_video_real_one_clip() != 0) return 1;

    if (g_failures == 0) {
        printf("ALL TESTS PASSED.\n");
        return 0;
    }
    return 1;
}
