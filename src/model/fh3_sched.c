#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fh3/fh3.h"
#include "model/fh3_sched.h"

const int FH3_DEFAULT_DMD_STEPS[8] = {999, 874, 749, 624, 500, 375, 250, 125};

int fh3_sched_build(Fh3Sched *s, const int dmd[8], double video_shift, double audio_shift) {
    if (!s) return FH3_E_INVALID;

    const int *steps = dmd ? dmd : FH3_DEFAULT_DMD_STEPS;
    double v_shift = (video_shift > 0.0) ? video_shift : 10.0;
    double a_shift = (audio_shift > 0.0) ? audio_shift : 3.0;

    s->n_steps = 8;
    s->video_shift = v_shift;
    s->audio_shift = a_shift;

    for (int i = 0; i < 8; i++) {
        double base = (double)steps[i] / 1000.0;
        double sv = (v_shift * base) / (1.0 + (v_shift - 1.0) * base);
        double sa = (a_shift * base) / (1.0 + (a_shift - 1.0) * base);
        s->video_sigmas[i] = (float)sv;
        s->audio_sigmas[i] = (float)sa;
    }
    s->video_sigmas[8] = 0.0f;
    s->audio_sigmas[8] = 0.0f;

    for (int i = 0; i < 8; i++) {
        s->steps[i].sigma_v = s->video_sigmas[i];
        s->steps[i].sigma_v_next = s->video_sigmas[i + 1];
        s->steps[i].sigma_a = s->audio_sigmas[i];
        s->steps[i].sigma_a_next = s->audio_sigmas[i + 1];
        s->steps[i].t_v = (float)(1.0 - (double)s->steps[i].sigma_v);
        s->steps[i].t_a = (float)(1.0 - (double)s->steps[i].sigma_a);
    }

    return FH3_OK;
}

void fh3_sched_step(
    const Fh3Sched *s,
    int i,
    float *x,
    const float *v,
    int n,
    int modality,
    const float *noise_next
) {
    if (!s || !x || !v || n <= 0) return;
    if (i < 0 || i >= s->n_steps) return;

    float sigma = (modality == 2) ? s->steps[i].sigma_a : s->steps[i].sigma_v;
    float sigma_next = (modality == 2) ? s->steps[i].sigma_a_next : s->steps[i].sigma_v_next;

    bool is_terminal = (i == s->n_steps - 1) || (sigma_next == 0.0f);

    if (is_terminal) {
        #pragma omp parallel for if(n > 1024)
        for (int j = 0; j < n; j++) {
            x[j] = x[j] + sigma * v[j];
        }
    } else if (noise_next != NULL) {
        float scale_x0 = 1.0f - sigma_next;
        #pragma omp parallel for if(n > 1024)
        for (int j = 0; j < n; j++) {
            float x0 = x[j] + sigma * v[j];
            x[j] = scale_x0 * x0 + sigma_next * noise_next[j];
        }
    } else {
        float ratio = sigma_next / sigma;
        float one_minus_r = 1.0f - ratio;
        #pragma omp parallel for if(n > 1024)
        for (int j = 0; j < n; j++) {
            float x0 = x[j] + sigma * v[j];
            x[j] = ratio * x[j] + one_minus_r * x0;
        }
    }
}

static inline void philox4x32_round(uint32_t ctr[4], const uint32_t key[2]) {
    uint64_t prod0 = 0xD2511F53ULL * (uint64_t)ctr[0];
    uint32_t hi0 = (uint32_t)(prod0 >> 32);
    uint32_t lo0 = (uint32_t)prod0;

    uint64_t prod1 = 0xCD9E8D57ULL * (uint64_t)ctr[2];
    uint32_t hi1 = (uint32_t)(prod1 >> 32);
    uint32_t lo1 = (uint32_t)prod1;

    uint32_t c0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t c1 = lo1;
    uint32_t c2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t c3 = lo0;

    ctr[0] = c0;
    ctr[1] = c1;
    ctr[2] = c2;
    ctr[3] = c3;
}

static inline void philox4x32_10(uint32_t ctr[4], uint32_t key[2]) {
    for (int r = 0; r < 10; r++) {
        philox4x32_round(ctr, key);
        key[0] += 0x9E3779B9U;
        key[1] += 0xBB67AE85U;
    }
}

void fh3_rng_normal(uint64_t seed, uint64_t stream, float *out, size_t n) {
    if (!out || n == 0) return;

    const uint32_t key_base[2] = {
        (uint32_t)(seed & 0xFFFFFFFFU),
        (uint32_t)(seed >> 32)
    };
    const uint32_t stream_lo = (uint32_t)(stream & 0xFFFFFFFFU);
    const uint32_t stream_hi = (uint32_t)(stream >> 32);

    size_t num_blocks = (n + 3) / 4;

    #pragma omp parallel for if(num_blocks > 256)
    for (size_t b = 0; b < num_blocks; b++) {
        uint32_t ctr[4];
        ctr[0] = (uint32_t)(b & 0xFFFFFFFFU);
        ctr[1] = (uint32_t)(b >> 32);
        ctr[2] = stream_lo;
        ctr[3] = stream_hi;

        uint32_t key[2];
        key[0] = key_base[0];
        key[1] = key_base[1];

        philox4x32_10(ctr, key);

        /* Map 4 uint32 words to 4 normal floats using 2 Box-Muller pairs */
        double u0 = ((double)ctr[0] + 1.0) / 4294967298.0;
        double u1 = ((double)ctr[1] + 1.0) / 4294967298.0;
        double r0 = sqrt(-2.0 * log(u0));
        double theta0 = 6.28318530717958647692 * u1;
        float z0 = (float)(r0 * cos(theta0));
        float z1 = (float)(r0 * sin(theta0));

        double u2 = ((double)ctr[2] + 1.0) / 4294967298.0;
        double u3 = ((double)ctr[3] + 1.0) / 4294967298.0;
        double r1 = sqrt(-2.0 * log(u2));
        double theta1 = 6.28318530717958647692 * u3;
        float z2 = (float)(r1 * cos(theta1));
        float z3 = (float)(r1 * sin(theta1));

        size_t base_idx = b * 4;
        if (base_idx < n) out[base_idx] = z0;
        if (base_idx + 1 < n) out[base_idx + 1] = z1;
        if (base_idx + 2 < n) out[base_idx + 2] = z2;
        if (base_idx + 3 < n) out[base_idx + 3] = z3;
    }
}

int fh3_load_noise(const char *path, float *out, size_t n) {
    if (!path || !out || n == 0) return FH3_E_INVALID;

    FILE *fp = fopen(path, "rb");
    if (!fp) return FH3_E_NOT_FOUND;

    size_t expected_bytes = n * sizeof(float);
    size_t rd = fread(out, 1, expected_bytes, fp);
    fclose(fp);

    if (rd != expected_bytes) {
        return FH3_E_TRUNCATED;
    }

    return FH3_OK;
}
