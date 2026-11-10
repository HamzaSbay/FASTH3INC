#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fh3/fh3.h"
#include "model/fh3_seq.h"

int fh3_seq_build(const Fh3SeqSpec *s, Fh3Seq *out) {
    if (!s || !out) return FH3_ERR;
    if (s->num_text < 0 || s->num_audio < 0 || s->num_video < 0) {
        return FH3_E_INVALID;
    }

    int T = s->num_text;
    int A = s->num_audio;
    int V = s->num_video;
    int N = T + A + V;
    if (N <= 0) return FH3_E_INVALID;

    memset(out, 0, sizeof(*out));
    out->seq_len = N;
    out->num_text = T;
    out->num_audio = A;
    out->num_video = V;

    out->tags = (int *)malloc((size_t)N * sizeof(int));
    out->pair_index = (int *)malloc((size_t)N * sizeof(int));
    out->pos = (double *)malloc((size_t)N * 3 * sizeof(double));
    out->cos = (float *)malloc((size_t)N * 96 * sizeof(float));
    out->sin = (float *)malloc((size_t)N * 96 * sizeof(float));

    if (!out->tags || !out->pair_index || !out->pos || !out->cos || !out->sin) {
        fh3_seq_free(out);
        return FH3_E_NOMEM;
    }

    /* Modality tags and pair indices:
     * FastH3 layout: [text (1) | audio (2) | video (0)]
     */
    for (int i = 0; i < T; i++) {
        out->tags[i] = 1;
        out->pair_index[i] = 1;
    }
    for (int i = T; i < T + A; i++) {
        out->tags[i] = 2;
        out->pair_index[i] = 2;
    }
    for (int i = T + A; i < N; i++) {
        out->tags[i] = 0;
        out->pair_index[i] = 0;
    }

    /* Position coordinates */
    if (s->custom_pos) {
        memcpy(out->pos, s->custom_pos, (size_t)N * 3 * sizeof(double));
    } else {
        /* 1. Text rows on time axis */
        for (int i = 0; i < T; i++) {
            out->pos[i * 3 + 0] = (double)i;
            out->pos[i * 3 + 1] = 0.0;
            out->pos[i * 3 + 2] = 0.0;
        }

        /* 2. Audio rows: channel-major on video rotary clock */
        int a_lat = (s->audio_latents > 0) ? s->audio_latents : A;
        int a_chan = (s->audio_channels > 0) ? s->audio_channels : 1;
        double sqrt_area = (s->latent_height > 0 && s->latent_width > 0)
                               ? sqrt((double)(s->latent_height * s->latent_width))
                               : 1.0;
        int pw = (s->patch_w > 0) ? s->patch_w : 2;
        int nw = (s->latent_width > 0) ? (s->latent_width / pw) : 1;
        double ratio_w = (s->latent_width > 0) ? ((double)s->latent_width / sqrt_area) : 1.0;
        double left_w = (1.0 - ratio_w) / 2.0;
        double w0 = left_w * 32.0;
        double w1 = (left_w + ratio_w * (double)(nw > 1 ? (nw - 1) : 0) / (double)nw) * 32.0;

        for (int c = 0; c < a_chan; c++) {
            double w_coord = (c == 0) ? w0 : w1;
            for (int t = 0; t < a_lat; t++) {
                int idx = T + c * a_lat + t;
                if (idx < T + A) {
                    out->pos[idx * 3 + 0] = (double)T + (double)t;
                    out->pos[idx * 3 + 1] = 0.0;
                    out->pos[idx * 3 + 2] = w_coord;
                }
            }
        }

        /* 3. Video rows: temporal + spatial grid */
        int ph = (s->patch_h > 0) ? s->patch_h : 2;
        int nh = (s->latent_height > 0) ? (s->latent_height / ph) : 1;
        int rpf = nh * nw;
        double ratio_h = (s->latent_height > 0) ? ((double)s->latent_height / sqrt_area) : 1.0;
        double left_h = (1.0 - ratio_h) / 2.0;

        static const double frames_per_latent[5] = {1.0, 4.0, 4.0, 4.0, 4.0};
        double cur_time = (double)T;
        int lframes = (s->latent_frames > 0) ? s->latent_frames : 1;

        for (int f = 0; f < lframes; f++) {
            if (f > 0) {
                double span = (5.0 / 3.0) * frames_per_latent[(f - 1) % 5];
                cur_time += span;
            }
            for (int r = 0; r < rpf; r++) {
                int ih = r / nw;
                int iw = r % nw;
                double h_coord = (left_h + ratio_h * ((double)ih / (double)nh)) * 32.0;
                double w_coord = (left_w + ratio_w * ((double)iw / (double)nw)) * 32.0;
                int idx = T + A + f * rpf + r;
                if (idx < N) {
                    out->pos[idx * 3 + 0] = cur_time;
                    out->pos[idx * 3 + 1] = h_coord;
                    out->pos[idx * 3 + 2] = w_coord;
                }
            }
        }
    }

    /* RoPE 3-axis cos and sin tables (96 channels per token) */
    for (int i = 0; i < N; i++) {
        double pt = out->pos[i * 3 + 0];
        double ph = out->pos[i * 3 + 1];
        double pw = out->pos[i * 3 + 2];

        for (int k = 0; k < 16; k++) {
            double inv_freq = 1.0 / pow(10000.0, (double)k / 16.0);
            float ft = (float)(pt * inv_freq);
            float fh = (float)(ph * inv_freq);
            float fw = (float)(pw * inv_freq);

            out->cos[i * 96 + k]      = cosf(ft);
            out->cos[i * 96 + 16 + k] = cosf(fh);
            out->cos[i * 96 + 32 + k] = cosf(fw);
            out->cos[i * 96 + 48 + k] = cosf(ft);
            out->cos[i * 96 + 64 + k] = cosf(fh);
            out->cos[i * 96 + 80 + k] = cosf(fw);

            out->sin[i * 96 + k]      = sinf(ft);
            out->sin[i * 96 + 16 + k] = sinf(fh);
            out->sin[i * 96 + 32 + k] = sinf(fw);
            out->sin[i * 96 + 48 + k] = sinf(ft);
            out->sin[i * 96 + 64 + k] = sinf(fh);
            out->sin[i * 96 + 80 + k] = sinf(fw);
        }
    }

    return FH3_OK;
}

void fh3_seq_free(Fh3Seq *seq) {
    if (!seq) return;
    if (seq->tags) { free(seq->tags); seq->tags = NULL; }
    if (seq->pair_index) { free(seq->pair_index); seq->pair_index = NULL; }
    if (seq->pos) { free(seq->pos); seq->pos = NULL; }
    if (seq->cos) { free(seq->cos); seq->cos = NULL; }
    if (seq->sin) { free(seq->sin); seq->sin = NULL; }
    seq->seq_len = 0;
    seq->num_text = 0;
    seq->num_audio = 0;
    seq->num_video = 0;
}
