#ifndef FH3_QUANT_H
#define FH3_QUANT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Convert a 16-bit half precision float to 32-bit float */
static inline float fh3_f16_to_f32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000U;
    uint32_t nonsign = w & 0x7FFFFFFFU;
    uint32_t renorm_shift = __builtin_clz(nonsign);
    renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
    int32_t inf_nan_mask = ((int32_t)(nonsign + 0x04000000U) >> 8) & 0x7F800000U;
    int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
    uint32_t res = sign | ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) | inf_nan_mask) & ~zero_mask);
    float out;
    memcpy(&out, &res, sizeof(float));
    return out;
#endif
}

/* Convert a 16-bit bfloat16 to 32-bit float */
static inline float fh3_bf16_to_f32(uint16_t b) {
    uint32_t val32 = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &val32, sizeof(float));
    return f;
}

#ifdef __cplusplus
}
#endif

#endif /* FH3_QUANT_H */
