#ifndef FH3_PACK_H
#define FH3_PACK_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fh3/fh3.h"
#include "io/fh3_portable_io.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FH3_PACK_MAGIC 0x50334846U  /* 'FH3P' */
#define FH3_PACK_VERSION 1U
#define FH3_PACK_ALIGNMENT 4096U
#define FH3_PACK_TENSORS_PER_LAYER 9U

#define FH3_ADALN_MAGIC 0x41334846U  /* 'FH3A' */
#define FH3_ADALN_VERSION 1U
#define FH3_ADALN_PAIRS_PER_STEP 3U
#define FH3_ADALN_CHUNKS_PER_BLOCK 6U
#define FH3_ADALN_CHUNKS_FINAL 2U

/* 
 * Tensor indices inside each layer run:
 * 0: qkv           (blocks.{L}.attn.qkv_proj.weight)
 * 1: gate_compress (blocks.{L}.attn.to_gate_compress.weight)
 * 2: out           (blocks.{L}.attn.out_proj.weight)
 * 3: fc1           (blocks.{L}.mlp.fc1.weight)
 * 4: fc2           (blocks.{L}.mlp.fc2.weight)
 * 5: norm1         (blocks.{L}.norm1.weight)
 * 6: norm2         (blocks.{L}.norm2.weight)
 * 7: q_norm        (blocks.{L}.attn.q_norm.weight)
 * 8: k_norm        (blocks.{L}.attn.k_norm.weight)
 */

#pragma pack(push, 1)

typedef struct Fh3PackTensorDesc {
    char name[64];
    uint32_t type;            /* Fh3QuantType */
    uint32_t n_dims;          /* 1 or 2 */
    int64_t shape[4];         /* Dimension sizes */
    uint64_t offset_in_layer; /* 32-byte aligned */
    uint64_t size_bytes;      /* Data byte size */
    int64_t in_dim;           /* in-dimension for Fh3W */
    int64_t out_dim;          /* out-dimension for Fh3W */
    uint8_t reserved[32];
} Fh3PackTensorDesc;

typedef struct Fh3PackLayerDesc {
    uint64_t file_offset;     /* 4096-aligned file offset */
    uint64_t size_bytes;      /* 4096-aligned run size */
    uint32_t n_tensors;       /* 9 */
    uint32_t reserved;
    Fh3PackTensorDesc tensors[FH3_PACK_TENSORS_PER_LAYER];
} Fh3PackLayerDesc;

typedef struct Fh3PackHeader {
    uint32_t magic;           /* FH3_PACK_MAGIC */
    uint32_t version;         /* FH3_PACK_VERSION */
    uint32_t num_layers;
    uint32_t layer_alignment; /* 4096 */
    uint64_t index_offset;
    uint64_t index_bytes;
    uint64_t data_offset;
    uint64_t total_bytes;
    uint8_t reserved[80];
} Fh3PackHeader;

typedef struct Fh3AdalnHeader {
    uint32_t magic;           /* FH3_ADALN_MAGIC */
    uint32_t version;         /* FH3_ADALN_VERSION */
    uint32_t n_steps;
    uint32_t n_blocks;
    uint32_t pairs_per_step;  /* 3 */
    uint32_t hidden_size;     /* e.g. 5376 */
    uint64_t schedule_hash;   /* 64-bit FNV-1a */
    uint64_t trunk_offset;    /* File offset of trunk table */
    uint64_t trunk_bytes;     /* Byte size of trunk table */
    uint64_t final_offset;    /* File offset of final layer table */
    uint64_t final_bytes;     /* Byte size of final layer table */
    uint8_t reserved[64];
} Fh3AdalnHeader;

#pragma pack(pop)

/* Open packed trunk file handle */
typedef struct Fh3Pack {
    Fh3File file;
    Fh3PackHeader header;
    int num_layers;
    Fh3PackLayerDesc *layers;
    const uint8_t *mapped_base;
    size_t mapped_size;
#if defined(_WIN32)
    void *h_map;
#endif
} Fh3Pack;

/* Open folded AdaLN modulation table */
typedef struct Fh3Adaln {
    Fh3File file;
    Fh3AdalnHeader header;
    int n_steps;
    int n_blocks;
    int pairs_per_step;
    int hidden_size;
    uint64_t schedule_hash;
    const float *trunk_data;  /* [n_steps][n_blocks][pairs_per_step][6][hidden_size] */
    const float *final_data;  /* [n_steps][pairs_per_step][2][hidden_size] */
    const uint8_t *mapped_base;
    size_t mapped_size;
    float *allocated_buf;     /* Non-null if fallback heap allocated */
#if defined(_WIN32)
    void *h_map;
#endif
} Fh3Adaln;

/* Packed trunk operations */
int fh3_pack_open(const char *path, Fh3Pack *p);
void fh3_pack_close(Fh3Pack *p);
Fh3LayerView fh3_pack_layer(const Fh3Pack *p, int L, const uint8_t *base);

/* AdaLN table operations */
uint64_t fh3_schedule_hash(const int *dmd_steps, int n_steps, double video_shift, double audio_shift);
int fh3_adaln_open(const char *path, uint64_t expect_hash, Fh3Adaln *a);
void fh3_adaln_close(Fh3Adaln *a);
const float *fh3_adaln_row(const Fh3Adaln *a, int step, int block, int pair, int chunk);

#ifdef __cplusplus
}
#endif

#endif /* FH3_PACK_H */
