#include "fh3_wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHUNK_FRAMES 4096

static void write_u16_le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
}

static void write_u32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)((v >> 24) & 0xFF);
}

int fh3_wav_write(const char *path, const float *pcm, int ch, int64_t n, int rate) {
    if (!path || !pcm || ch <= 0 || n < 0 || rate <= 0) {
        return FH3_E_INVALID;
    }

    uint64_t total_samples = (uint64_t)ch * (uint64_t)n;
    uint64_t data_bytes = total_samples * 2;
    if (data_bytes > 0xFFFFFFFFULL - 36ULL) {
        /* Standard 32-bit RIFF/WAVE size overflow */
        return FH3_E_INVALID;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        return FH3_E_IO;
    }

    /* 44-byte standard RIFF/WAVE header */
    uint8_t hdr[44];
    /* RIFF chunk descriptor */
    memcpy(hdr + 0, "RIFF", 4);
    write_u32_le(hdr + 4, (uint32_t)(36 + data_bytes));
    memcpy(hdr + 8, "WAVE", 4);

    /* "fmt " sub-chunk */
    memcpy(hdr + 12, "fmt ", 4);
    write_u32_le(hdr + 16, 16);               /* Subchunk1Size = 16 for PCM */
    write_u16_le(hdr + 20, 1);                /* AudioFormat = 1 (PCM) */
    write_u16_le(hdr + 22, (uint16_t)ch);     /* NumChannels */
    write_u32_le(hdr + 24, (uint32_t)rate);   /* SampleRate */
    uint32_t byte_rate = (uint32_t)rate * (uint32_t)ch * 2;
    write_u32_le(hdr + 28, byte_rate);        /* ByteRate */
    uint16_t block_align = (uint16_t)(ch * 2);
    write_u16_le(hdr + 32, block_align);      /* BlockAlign */
    write_u16_le(hdr + 34, 16);               /* BitsPerSample = 16 */

    /* "data" sub-chunk */
    memcpy(hdr + 36, "data", 4);
    write_u32_le(hdr + 40, (uint32_t)data_bytes);

    if (fwrite(hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) {
        fclose(fp);
        return FH3_E_IO;
    }

    /* Stream interleaved 16-bit PCM in chunks */
    int16_t buf[CHUNK_FRAMES * 8]; /* supports up to 8 channels comfortably */
    int frames_per_chunk = CHUNK_FRAMES;
    if (ch > 8) {
        frames_per_chunk = (int)(sizeof(buf) / (sizeof(int16_t) * (size_t)ch));
        if (frames_per_chunk < 1) frames_per_chunk = 1;
    }

    int64_t frame_idx = 0;
    while (frame_idx < n) {
        int count = (int)(n - frame_idx);
        if (count > frames_per_chunk) count = frames_per_chunk;

        int buf_idx = 0;
        for (int i = 0; i < count; i++) {
            int64_t cur_frame = frame_idx + i;
            for (int c = 0; c < ch; c++) {
                float s = pcm[(int64_t)c * n + cur_frame];
                if (s > 1.0f) s = 1.0f;
                else if (s < -1.0f) s = -1.0f;
                float scaled = s * 32767.0f;
                long val = lrintf(scaled);
                if (val > 32767) val = 32767;
                else if (val < -32768) val = -32768;
                buf[buf_idx++] = (int16_t)val;
            }
        }

        size_t to_write = (size_t)buf_idx * sizeof(int16_t);
        if (fwrite(buf, 1, to_write, fp) != to_write) {
            fclose(fp);
            return FH3_E_IO;
        }

        frame_idx += count;
    }

    fclose(fp);
    return FH3_OK;
}
