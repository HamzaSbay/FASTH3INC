#ifndef FH3_DIT_H
#define FH3_DIT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "fh3/fh3.h"
#include "model/fh3_seq.h"
#include "io/fh3_pack.h"
#include "io/fh3_trunk.h"
#include "io/fh3_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Fh3DitCfg {
    int hidden_size;          /* e.g. 256 for tiny, 5376 for full */
    int num_heads;            /* e.g. 2 for tiny, 56 for full */
    int head_dim;             /* e.g. 128 */
    int num_layers;           /* e.g. 3 for tiny, 50 for full */
    int num_refiner_layers;   /* e.g. 2 */
    int ffn_dim;              /* e.g. 512 for tiny, 14336 for full */
    int text_dim;             /* e.g. 64 for tiny, 5120 for full */
    int time_embed_dim;       /* e.g. 128 for tiny, 2688 for full */
    int in_channels;          /* e.g. 24 */
    int audio_in_channels;    /* e.g. 32 */
    int patch_size[3];        /* [1, 2, 2] */
    float norm_eps;           /* 1e-5f */
    float qk_norm_eps;        /* 1e-5f */
    float final_norm_eps;     /* 1e-5f */
    int rope_freq_dim;        /* 16 */
    float rope_theta;         /* 10000.0f */
};

typedef struct Fh3RefinerBlock {
    const float *norm1;       /* [hidden_size] */
    Fh3W qkv;                 /* [3 * inner_dim, hidden_size] */
    const float *q_norm;      /* [head_dim] */
    const float *k_norm;      /* [head_dim] */
    Fh3W out;                 /* [hidden_size, inner_dim] */
    const float *norm2;       /* [hidden_size] */
    Fh3W fc1;                 /* [2 * ffn_dim, hidden_size] */
    Fh3W fc2;                 /* [hidden_size, ffn_dim] */
} Fh3RefinerBlock;

struct Fh3Dit {
    Fh3DitCfg cfg;
    Fh3Gguf globals_gguf;
    Fh3Adaln adaln;
    uint8_t *globals_data;
    size_t globals_data_size;

    /* Global projection weights and biases */
    Fh3W video_patch_proj;
    const float *video_patch_proj_bias;

    Fh3W audio_patch_proj;
    const float *audio_patch_proj_bias;

    Fh3W condition_proj;
    const float *condition_proj_bias;

    /* Refiner blocks */
    Fh3RefinerBlock refiner[4];
    const float *refiner_final_norm;
    int refiner_run_count;

    /* Final layer heads and norm */
    const float *final_norm;
    Fh3W video_out;
    const float *video_out_bias;
    Fh3W audio_out;
    const float *audio_out_bias;

    /* Allocated dequant buffers (if any 1D tensor is quantized) */
    void **dequant_allocs;
    int n_dequant_allocs;
};

int fh3_dit_open(const char *globals_path, const char *adaln_path, const Fh3DitCfg *cfg, Fh3Dit **out);
void fh3_dit_close(Fh3Dit *m);
int fh3_dit_refiner_count(const Fh3Dit *m);

int fh3_dit_prepare_text(const Fh3Dit *m, const float *text_emb, int T, float *text_rows);
int fh3_dit_forward(const Fh3Dit *m, Fh3Trunk *tr, const Fh3Seq *seq, int step,
                    const float *x_video, const float *x_audio, const float *text_rows,
                    float *v_video, float *v_audio);

#ifdef __cplusplus
}
#endif

#endif /* FH3_DIT_H */
