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
typedef struct Fh3TeCfg Fh3TeCfg;
typedef struct Fh3Seq Fh3Seq;
typedef struct Fh3Sched Fh3Sched;
typedef struct Fh3VsaCfg Fh3VsaCfg;
typedef struct Fh3File Fh3File;
typedef struct Fh3Gguf Fh3Gguf;
typedef struct Fh3Tensor Fh3Tensor;
typedef struct Fh3Trunk Fh3Trunk;
typedef struct Fh3Tok Fh3Tok;
typedef struct Fh3Te Fh3Te;
typedef struct Fh3VaeA Fh3VaeA;
typedef struct Fh3VaeV Fh3VaeV;
typedef struct Fh3FrameSink Fh3FrameSink;
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
