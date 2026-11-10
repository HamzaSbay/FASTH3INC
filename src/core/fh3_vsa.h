#ifndef FH3_VSA_H
#define FH3_VSA_H

#include "fh3/fh3.h"
#include "model/fh3_seq.h"

#ifdef __cplusplus
extern "C" {
#endif

void fh3_vsa_attention(const Fh3VsaCfg *c, const float *q, const float *k, const float *v,
                       const float *gate, float *o, const Fh3Seq *seq, int heads, int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* FH3_VSA_H */
