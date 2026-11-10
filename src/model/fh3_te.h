#ifndef FH3_TE_H
#define FH3_TE_H

#include <stddef.h>
#include <stdint.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Fh3TeCfg {
    int hidden_size;          /* 5120 (or tiny 128) */
    int num_heads;            /* 64 (or tiny 4) */
    int num_kv_heads;         /* 8 (or tiny 2) */
    int head_dim;             /* 128 (or tiny 32) */
    int num_layers;           /* 50 (or tiny 3) */
    int intermediate_size;    /* 25600 (or tiny 256) */
    float norm_eps;           /* 1e-6f */
    float qk_norm_eps;        /* 1e-6f */
    float rope_theta;         /* 5000000.0f */
};

typedef struct Fh3Te Fh3Te;

/* Open text encoder with trunk path (optional or pack), embed path, and config */
int fh3_te_open(const char *trunk_path, const char *embed_path, const Fh3TeCfg *cfg, Fh3Te **out);

/* Release text encoder resources */
void fh3_te_close(Fh3Te *m);

/* Forward pass: encode T token IDs into out (T x hidden_size) pre-final-norm hidden state */
int fh3_te_encode(const Fh3Te *m, Fh3Trunk *tr, const int *ids, int T, float *out);

/* Prompt cache: lookup pre-computed embedding from cache_dir/<sha256(ids)>.emb */
int fh3_te_cache_lookup(const char *cache_dir, const int *ids, int T, float *out);

/* Prompt cache: save pre-computed embedding to cache_dir/<sha256(ids)>.emb */
int fh3_te_cache_save(const char *cache_dir, const int *ids, int T, const float *emb);

#ifdef __cplusplus
}
#endif

#endif /* FH3_TE_H */
