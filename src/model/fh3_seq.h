#ifndef FH3_SEQ_H
#define FH3_SEQ_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fh3SeqSpec {
    int num_text;             /* Number of text tokens */
    int num_audio;            /* Number of audio tokens */
    int num_video;            /* Number of video tokens */
    int latent_frames;        /* Target video latent frames */
    int latent_height;        /* Target video latent height */
    int latent_width;         /* Target video latent width */
    int patch_t;              /* Patch size t (e.g. 1) */
    int patch_h;              /* Patch size h (e.g. 2) */
    int patch_w;              /* Patch size w (e.g. 2) */
    int audio_latents;        /* Audio latents per channel */
    int audio_channels;       /* Audio channels (e.g. 2) */
    const double *custom_pos; /* If non-NULL, [N * 3] fp64 positions (t, h, w) */
    int grid_t;               /* Optional explicit video grid (t, h, w) in tokens/patches */
    int grid_h;
    int grid_w;
} Fh3SeqSpec;

struct Fh3Seq {
    int seq_len;              /* N = num_text + num_audio + num_video */
    int num_text;             /* T */
    int num_audio;            /* A */
    int num_video;            /* V */
    int grid_t;               /* video grid (t, h, w) in tokens/patches */
    int grid_h;
    int grid_w;
    int *tags;                /* [N] modality tags: 0 video, 1 text, 2 audio */
    int *pair_index;          /* [N] pair indices for AdaLN table: 0 video, 1 text, 2 audio */
    double *pos;              /* [N * 3] fp64 position coordinates (t, h, w) */
    float *cos;               /* [N * 96] f32 RoPE cosine table */
    float *sin;               /* [N * 96] f32 RoPE sine table */
};

int fh3_seq_build(const Fh3SeqSpec *s, Fh3Seq *out);
void fh3_seq_free(Fh3Seq *seq);

#ifdef __cplusplus
}
#endif

#endif /* FH3_SEQ_H */
