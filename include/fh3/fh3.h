#ifndef FH3_H
#define FH3_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations of core engine types */
typedef struct Fh3DitCfg Fh3DitCfg;
typedef struct Fh3Dit Fh3Dit;
typedef struct Fh3TeCfg Fh3TeCfg;
typedef struct Fh3Seq Fh3Seq;
typedef struct Fh3SeqSpec Fh3SeqSpec;

typedef struct Fh3SchedStep {
    float sigma_v;
    float sigma_v_next;
    float sigma_a;
    float sigma_a_next;
    float t_v;
    float t_a;
} Fh3SchedStep;

typedef struct Fh3Sched {
    int n_steps;               /* 8 */
    double video_shift;        /* 10.0 */
    double audio_shift;        /* 3.0 */
    float video_sigmas[9];     /* 8 steps + terminal 0.0f */
    float audio_sigmas[9];     /* 8 steps + terminal 0.0f */
    Fh3SchedStep steps[8];
} Fh3Sched;
typedef enum {
    FH3_ATTN_DENSE = 0,
    FH3_ATTN_VSA   = 1,
} Fh3AttnType;

typedef struct Fh3VsaCfg {
    int tile;      /* default 64 */
    float sparsity; /* default 0.8f */
} Fh3VsaCfg;
typedef struct Fh3File Fh3File;
typedef struct Fh3Gguf Fh3Gguf;
typedef struct Fh3Tensor Fh3Tensor;
typedef struct Fh3Trunk Fh3Trunk;
typedef struct Fh3Tok Fh3Tok;
typedef struct Fh3Te Fh3Te;
typedef struct Fh3VaeA Fh3VaeA;
typedef struct Fh3VaeV Fh3VaeV;

typedef struct Fh3FrameSink {
    void *user;
    int (*write_frame)(void *user, int frame_idx, const float *rgb /*3 x H x W planar in [0, 1]*/, int width, int height);
} Fh3FrameSink;

typedef struct Fh3Pack Fh3Pack;
typedef struct Fh3Adaln Fh3Adaln;

/* Quantization types (matching GGML / GGUF constants) */
typedef enum {
    FH3_QUANT_F32  = 0,
    FH3_QUANT_F16  = 1,
    FH3_QUANT_Q4_0 = 2,
    FH3_QUANT_Q4_1 = 3,
    FH3_QUANT_Q5_0 = 6,
    FH3_QUANT_Q5_1 = 7,
    FH3_QUANT_Q8_0 = 8,
    FH3_QUANT_Q8_1 = 9,
    FH3_QUANT_Q2_K = 10,
    FH3_QUANT_Q3_K = 11,
    FH3_QUANT_Q4_K = 12,
    FH3_QUANT_Q5_K = 13,
    FH3_QUANT_Q6_K = 14,
    FH3_QUANT_Q8_K = 15,
    FH3_QUANT_BF16 = 30,
} Fh3QuantType;

/* Weight matrix representation */
typedef struct Fh3W {
    int type;
    int64_t in;
    int64_t out;
    const uint8_t *data;
} Fh3W;

/* Layer view with Fh3W weight structures pointing into layer memory */
typedef struct Fh3LayerView {
    Fh3W qkv;
    Fh3W gate_compress;
    Fh3W out;
    Fh3W fc1;
    Fh3W fc2;
    Fh3W norm1;
    Fh3W norm2;
    Fh3W q_norm;
    Fh3W k_norm;
} Fh3LayerView;

/* Quantization routines */
size_t fh3_row_bytes(int type, int64_t n);
void fh3_dequant(int type, const void *src, float *dst, int64_t n);

/* Weight-stationary matrix multiplication */
void fh3_matmul(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null);
void fh3_matmul_scalar(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null);
void fh3_matmul_avx2(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null);

/* Elementwise and attention ops */
void fh3_rmsnorm(float *x, const float *w, int n, int dim, float eps);
void fh3_rope3(float *x, const float *cos, const float *sin, int n, int heads, int head_dim, int rot_dim);
void fh3_swiglu(const float *in, float *out, int n, int F);
void fh3_modulate(float *x, const float *shift, const float *scale, int n, int dim);
void fh3_gate_residual(float *h, const float *gate, const float *y, int n, int dim);
void fh3_attention(const float *q, const float *k, const float *v, float *o, int n, int heads, int hd);
void fh3_vsa_attention(const Fh3VsaCfg *c, const float *q, const float *k, const float *v,
                       const float *gate, float *o, const Fh3Seq *seq, int heads, int head_dim);
void fh3_timestep_embed(const double *t, int nt, int freq_dim, float *out);

/* Packed trunk operations */
int fh3_pack_open(const char *path, Fh3Pack *p);
void fh3_pack_close(Fh3Pack *p);
Fh3LayerView fh3_pack_layer(const Fh3Pack *p, int L, const uint8_t *base);

/* AdaLN table operations */
uint64_t fh3_schedule_hash(const int *dmd_steps, int n_steps, double video_shift, double audio_shift);
int fh3_adaln_open(const char *path, uint64_t expect_hash, Fh3Adaln *a);
void fh3_adaln_close(Fh3Adaln *a);
const float *fh3_adaln_row(const Fh3Adaln *a, int step, int block, int pair, int chunk);

/* Trunk streamer operations */
typedef struct Fh3TrunkStats {
    uint64_t hits;
    uint64_t misses;
    uint64_t bytes_read;
    double read_s;
    double wait_s;
} Fh3TrunkStats;

int fh3_trunk_open(const char *path, uint64_t budget_bytes, Fh3Trunk *t);
void fh3_trunk_close(Fh3Trunk *t);
void fh3_trunk_prefetch(Fh3Trunk *t, int L);
const uint8_t *fh3_trunk_acquire(Fh3Trunk *t, int L);
Fh3TrunkStats fh3_trunk_stats(const Fh3Trunk *t);

/* Sequence builder operations */
int fh3_seq_build(const Fh3SeqSpec *s, Fh3Seq *out);
void fh3_seq_free(Fh3Seq *seq);

/* DiT transformer operations */
int fh3_dit_open(const char *globals_path, const char *adaln_path, const Fh3DitCfg *cfg, Fh3Dit **out);
void fh3_dit_close(Fh3Dit *m);
int fh3_dit_refiner_count(const Fh3Dit *m);
int fh3_dit_prepare_text(const Fh3Dit *m, const float *text_emb, int T, float *text_rows);
int fh3_dit_forward(const Fh3Dit *m, Fh3Trunk *tr, const Fh3Seq *seq, int step,
                    const float *x_video, const float *x_audio, const float *text_rows,
                    float *v_video, float *v_audio);

/* Sampler scheduler & noise operations */
int fh3_sched_build(Fh3Sched *s, const int dmd[8], double video_shift, double audio_shift);
void fh3_sched_step(const Fh3Sched *s, int i, float *x, const float *v, int n, int modality, const float *noise_next);
void fh3_rng_normal(uint64_t seed, uint64_t stream, float *out, size_t n);
int fh3_load_noise(const char *path, float *out, size_t n);

/* Tokenizer operations */
int fh3_tok_load(const char *vocab_json, const char *merges_txt, Fh3Tok *t);
void fh3_tok_free(Fh3Tok *t);
int fh3_tok_encode(const Fh3Tok *t, const char *utf8, int **ids, int *n);
int fh3_tok_encode_opt(const Fh3Tok *t, const char *utf8, int normalize_newlines, int **ids, int *n);

/* Text encoder operations */
int fh3_te_open(const char *trunk_path, const char *embed_path, const Fh3TeCfg *cfg, Fh3Te **out);
void fh3_te_close(Fh3Te *m);
int fh3_te_encode(const Fh3Te *m, Fh3Trunk *tr, const int *ids, int T, float *out);
int fh3_te_cache_lookup(const char *cache_dir, const int *ids, int T, float *out);
int fh3_te_cache_save(const char *cache_dir, const int *ids, int T, const float *emb);

/* Audio VAE (BigVGAN) decoder operations */
int fh3_vae_audio_open(const char *path, Fh3VaeA **out);
void fh3_vae_audio_close(Fh3VaeA *m);
int fh3_vae_audio_decode(const Fh3VaeA *m, const float *lat /*2 ch × L × 32*/, int L, float *pcm /*2 × L*800*/);
int fh3_vae_audio_upsample_ratio(const Fh3VaeA *m);

/* WAV audio writer */
int fh3_wav_write(const char *path, const float *pcm /*ch × n*/, int ch, int64_t n, int rate /*32000*/);

/* Video VAE (ViT) decoder operations */
int fh3_vae_video_open(const char *path, Fh3VaeV **out);
void fh3_vae_video_close(Fh3VaeV *m);
int fh3_vae_video_decode(const Fh3VaeV *m, const float *lat /*24 × F × h × w*/, int F, int h, int w, Fh3FrameSink *sink);
int fh3_vae_video_patch_size(const Fh3VaeV *m);
int fh3_vae_video_patch_size_t(const Fh3VaeV *m);

/* Y4M video writer */
typedef struct Fh3Y4m Fh3Y4m;
int fh3_y4m_open(const char *path, int width, int height, int fps_num, int fps_den, Fh3Y4m **out);
int fh3_y4m_write_rgb(Fh3Y4m *y, const float *rgb /*3 × H × W in [0, 1]*/);
void fh3_y4m_close(Fh3Y4m *y);
Fh3FrameSink fh3_y4m_frame_sink(Fh3Y4m *y);

/* Execution plan & memory check */
typedef struct Fh3Plan {
    uint64_t budget_bytes;
    uint64_t min_bytes;
    uint64_t pinned_bytes;
    uint64_t ring_bytes;
    uint64_t activation_bytes;
    uint64_t table_bytes;
    uint64_t total_bytes;
} Fh3Plan;

uint64_t fh3_plan_bytes(const Fh3Plan *p);
int fh3_plan_check(const Fh3Plan *p);

/* Return / Error codes */
#define FH3_OK             0
#define FH3_ERR          (-1)
#define FH3_E_IO         (-2)
#define FH3_E_TRUNCATED  (-3)
#define FH3_E_BAD_MAGIC  (-4)
#define FH3_E_BAD_VERSION (-5)
#define FH3_E_NOT_FOUND  (-6)
#define FH3_E_NOMEM      (-7)
#define FH3_E_INVALID    (-8)

/* Version info */
#define FH3_VERSION_MAJOR 0
#define FH3_VERSION_MINOR 1
#define FH3_VERSION_PATCH 0
#define FH3_VERSION_STRING "0.1.0"

const char *fh3_version(void);

#ifdef __cplusplus
}
#endif

#endif /* FH3_H */
