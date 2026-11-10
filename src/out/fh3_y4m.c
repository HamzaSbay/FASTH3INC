#include "fh3_y4m.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Fh3Y4m {
    FILE *fp;
    int width;
    int height;
    int fps_num;
    int fps_den;
    uint8_t *y_plane;
    uint8_t *u_plane;
    uint8_t *v_plane;
};

int fh3_y4m_open(const char *path, int width, int height, int fps_num, int fps_den, Fh3Y4m **out) {
    if (!path || !out || width <= 0 || height <= 0 || (width & 1) || (height & 1) || fps_num <= 0 || fps_den <= 0) {
        return FH3_E_INVALID;
    }
    *out = NULL;

    FILE *fp = fopen(path, "wb");
    if (!fp) return FH3_E_IO;

    /* Write standard Y4M header */
    int n = fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420jpeg\n", width, height, fps_num, fps_den);
    if (n <= 0) {
        fclose(fp);
        return FH3_E_IO;
    }

    Fh3Y4m *y = (Fh3Y4m *)calloc(1, sizeof(Fh3Y4m));
    if (!y) {
        fclose(fp);
        return FH3_E_NOMEM;
    }

    y->fp = fp;
    y->width = width;
    y->height = height;
    y->fps_num = fps_num;
    y->fps_den = fps_den;

    y->y_plane = (uint8_t *)malloc((size_t)width * (size_t)height);
    y->u_plane = (uint8_t *)malloc((size_t)(width / 2) * (size_t)(height / 2));
    y->v_plane = (uint8_t *)malloc((size_t)(width / 2) * (size_t)(height / 2));

    if (!y->y_plane || !y->u_plane || !y->v_plane) {
        fh3_y4m_close(y);
        return FH3_E_NOMEM;
    }

    *out = y;
    return FH3_OK;
}

static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

int fh3_y4m_write_rgb(Fh3Y4m *y, const float *rgb) {
    if (!y || !y->fp || !rgb) return FH3_E_INVALID;

    /* Write frame header */
    if (fputs("FRAME\n", y->fp) == EOF) return FH3_E_IO;

    int w = y->width;
    int h = y->height;
    size_t wh = (size_t)w * (size_t)h;
    const float *r_ch = rgb;
    const float *g_ch = rgb + wh;
    const float *b_ch = rgb + 2 * wh;

    /* Convert Y plane (BT.601 / JPEG full-range) */
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < h; row++) {
        size_t row_off = (size_t)row * (size_t)w;
        for (int col = 0; col < w; col++) {
            size_t idx = row_off + (size_t)col;
            float r = r_ch[idx];
            float g = g_ch[idx];
            float b = b_ch[idx];
            float y_val = 0.299f * r + 0.587f * g + 0.114f * b;
            y->y_plane[idx] = clamp_u8((int)lrintf(y_val * 255.0f));
        }
    }

    /* Convert U and V planes with 2x2 box average downsampling */
    int uv_w = w / 2;
    int uv_h = h / 2;

    #pragma omp parallel for schedule(static)
    for (int urow = 0; urow < uv_h; urow++) {
        int r0 = urow * 2;
        int r1 = r0 + 1;
        size_t row0_off = (size_t)r0 * (size_t)w;
        size_t row1_off = (size_t)r1 * (size_t)w;
        size_t uv_row_off = (size_t)urow * (size_t)uv_w;

        for (int ucol = 0; ucol < uv_w; ucol++) {
            int c0 = ucol * 2;
            int c1 = c0 + 1;

            size_t idx00 = row0_off + (size_t)c0;
            size_t idx01 = row0_off + (size_t)c1;
            size_t idx10 = row1_off + (size_t)c0;
            size_t idx11 = row1_off + (size_t)c1;

            float r_avg = 0.25f * (r_ch[idx00] + r_ch[idx01] + r_ch[idx10] + r_ch[idx11]);
            float g_avg = 0.25f * (g_ch[idx00] + g_ch[idx01] + g_ch[idx10] + g_ch[idx11]);
            float b_avg = 0.25f * (b_ch[idx00] + b_ch[idx01] + b_ch[idx10] + b_ch[idx11]);

            float u_val = -0.168736f * r_avg - 0.331264f * g_avg + 0.500000f * b_avg + (128.0f / 255.0f);
            float v_val =  0.500000f * r_avg - 0.418688f * g_avg - 0.081312f * b_avg + (128.0f / 255.0f);

            size_t uv_idx = uv_row_off + (size_t)ucol;
            y->u_plane[uv_idx] = clamp_u8((int)lrintf(u_val * 255.0f));
            y->v_plane[uv_idx] = clamp_u8((int)lrintf(v_val * 255.0f));
        }
    }

    if (fwrite(y->y_plane, 1, wh, y->fp) != wh) return FH3_E_IO;
    size_t uv_sz = (size_t)uv_w * (size_t)uv_h;
    if (fwrite(y->u_plane, 1, uv_sz, y->fp) != uv_sz) return FH3_E_IO;
    if (fwrite(y->v_plane, 1, uv_sz, y->fp) != uv_sz) return FH3_E_IO;

    return FH3_OK;
}

void fh3_y4m_close(Fh3Y4m *y) {
    if (!y) return;
    if (y->fp) fclose(y->fp);
    free(y->y_plane);
    free(y->u_plane);
    free(y->v_plane);
    free(y);
}

static int y4m_sink_cb(void *user, int frame_idx, const float *rgb, int width, int height) {
    (void)frame_idx;
    Fh3Y4m *y = (Fh3Y4m *)user;
    if (!y || !rgb || width != y->width || height != y->height) {
        return FH3_E_INVALID;
    }
    return fh3_y4m_write_rgb(y, rgb);
}

Fh3FrameSink fh3_y4m_frame_sink(Fh3Y4m *y) {
    Fh3FrameSink s;
    s.user = y;
    s.write_frame = y4m_sink_cb;
    return s;
}
