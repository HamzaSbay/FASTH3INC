#ifndef FH3_GGUF_H
#define FH3_GGUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fh3/fh3.h"
#include "io/fh3_portable_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF Value Types */
typedef enum {
    FH3_GGUF_TYPE_UINT8   = 0,
    FH3_GGUF_TYPE_INT8    = 1,
    FH3_GGUF_TYPE_UINT16  = 2,
    FH3_GGUF_TYPE_INT16   = 3,
    FH3_GGUF_TYPE_UINT32  = 4,
    FH3_GGUF_TYPE_INT32   = 5,
    FH3_GGUF_TYPE_FLOAT32 = 6,
    FH3_GGUF_TYPE_BOOL    = 7,
    FH3_GGUF_TYPE_STRING  = 8,
    FH3_GGUF_TYPE_ARRAY   = 9,
    FH3_GGUF_TYPE_UINT64  = 10,
    FH3_GGUF_TYPE_INT64   = 11,
    FH3_GGUF_TYPE_FLOAT64 = 12,
} Fh3GgufValueType;

struct Fh3Tensor {
    char name[128];
    int ndim;
    int64_t ne[4];
    int type;
    uint64_t off;
    uint64_t nbytes;
};

typedef struct Fh3GgufKv {
    char *key;
    uint32_t type;
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        bool b;
        char *str;
        struct {
            uint32_t elem_type;
            uint64_t count;
            void *data;
        } arr;
    } val;
} Fh3GgufKv;

struct Fh3Gguf {
    Fh3File file;
    uint32_t version;
    uint32_t alignment;
    uint64_t data_offset;
    uint64_t tensor_count;
    Fh3Tensor *tensors;
    uint64_t kv_count;
    Fh3GgufKv *kvs;
};

int fh3_gguf_open(const char *path, Fh3Gguf *g);
const Fh3Tensor *fh3_gguf_find(const Fh3Gguf *g, const char *name);
int fh3_gguf_shape(const Fh3Gguf *g, const Fh3Tensor *t, int64_t out[4]);
void fh3_gguf_close(Fh3Gguf *g);

const Fh3GgufKv *fh3_gguf_find_kv(const Fh3Gguf *g, const char *key);

#ifdef __cplusplus
}
#endif

#endif /* FH3_GGUF_H */
