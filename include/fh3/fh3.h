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

/* Quantization routines */
size_t fh3_row_bytes(int type, int64_t n);
void fh3_dequant(int type, const void *src, float *dst, int64_t n);

/* Weight-stationary matrix multiplication */
void fh3_matmul(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null);
void fh3_matmul_scalar(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null);
void fh3_matmul_avx2(const Fh3W *w, const float *x, float *y, int n_tok, const float *bias_or_null);

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
