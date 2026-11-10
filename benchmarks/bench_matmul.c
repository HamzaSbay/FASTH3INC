#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "fh3/fh3.h"

int main(int argc, char **argv) {
    int64_t in_dim = 5376;
    int64_t out_dim = 14336;
    int n_tok = 1024;
    int n_iters = 3;

    if (argc >= 2) n_tok = atoi(argv[1]);
    if (argc >= 3) n_iters = atoi(argv[2]);

    printf("================================================================\n");
    printf(" FastH3 Benchmark: Weight-Stationary Matmul (Q4_K)\n");
    printf("================================================================\n");
    printf(" Matrix: in=%lld, out=%lld, n_tok=%d\n", (long long)in_dim, (long long)out_dim, n_tok);
    printf(" Threads: %d (OpenMP max threads)\n", omp_get_max_threads());

    size_t row_bytes = fh3_row_bytes(FH3_QUANT_Q4_K, in_dim);
    size_t total_w_bytes = (size_t)out_dim * row_bytes;
    size_t total_x_bytes = (size_t)n_tok * (size_t)in_dim * sizeof(float);
    size_t total_y_bytes = (size_t)n_tok * (size_t)out_dim * sizeof(float);
    size_t bias_bytes    = (size_t)out_dim * sizeof(float);

    printf(" Memory: W=%.1f MB, X=%.1f MB, Y=%.1f MB\n",
           (double)total_w_bytes / (1024.0 * 1024.0),
           (double)total_x_bytes / (1024.0 * 1024.0),
           (double)total_y_bytes / (1024.0 * 1024.0));

    uint8_t *w_data = (uint8_t *)malloc(total_w_bytes);
    float *x_data   = (float *)malloc(total_x_bytes);
    float *bias     = (float *)malloc(bias_bytes);
    float *y_data   = (float *)malloc(total_y_bytes);

    if (!w_data || !x_data || !bias || !y_data) {
        fprintf(stderr, "Allocation failed!\n");
        return 1;
    }

    /* Initialize Q4_K blocks with valid float16 scales and repeatable data */
    size_t n_blocks = total_w_bytes / 144;
    uint16_t d_val = 0x3800;   /* float16 0.5 */
    uint16_t dmin_val = 0x2c00;/* float16 0.0625 */
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = w_data + b * 144;
        memcpy(blk, &d_val, 2);
        memcpy(blk + 2, &dmin_val, 2);
        memset(blk + 4, 0x12, 12);  /* scales */
        memset(blk + 16, 0x34, 128); /* qs */
    }

    /* Initialize input and bias */
    for (size_t i = 0; i < (size_t)n_tok * in_dim; i++) {
        x_data[i] = ((float)(i % 100) - 50.0f) * 0.001f;
    }
    for (size_t i = 0; i < (size_t)out_dim; i++) {
        bias[i] = ((float)(i % 10) - 5.0f) * 0.01f;
    }

    Fh3W w;
    w.type = FH3_QUANT_Q4_K;
    w.in = in_dim;
    w.out = out_dim;
    w.data = w_data;

    double gflops = 2.0 * (double)in_dim * (double)out_dim * (double)n_tok / 1e9;
    printf(" Workload: %.3f GFLOP per matmul\n", gflops);

    /* Warm-up */
    printf("\n Warming up (1 iteration)...\n");
    double t0 = omp_get_wtime();
    fh3_matmul(&w, x_data, y_data, n_tok, bias);
    double t_warm = omp_get_wtime() - t0;
    printf(" Warmup completed in %.3f s (%.1f GFLOP/s)\n\n", t_warm, gflops / t_warm);

    /* Timed iterations */
    printf(" Running %d timed iterations...\n", n_iters);
    double total_time = 0.0;
    double min_time = 1e9;

    for (int iter = 0; iter < n_iters; iter++) {
        t0 = omp_get_wtime();
        fh3_matmul(&w, x_data, y_data, n_tok, bias);
        double dt = omp_get_wtime() - t0;
        total_time += dt;
        if (dt < min_time) min_time = dt;
        printf("   Iteration %d: %.3f s -> %.2f GFLOP/s\n", iter + 1, dt, gflops / dt);
    }

    double avg_time = total_time / n_iters;
    double avg_gflops = gflops / avg_time;
    double peak_gflops = gflops / min_time;

    printf("\n----------------------------------------------------------------\n");
    printf(" Summary Results:\n");
    printf("   Average Time:   %.3f s\n", avg_time);
    printf("   Best Time:      %.3f s\n", min_time);
    printf("   Average Speed:  %.2f GFLOP/s\n", avg_gflops);
    printf("   Peak Speed:     %.2f GFLOP/s\n", peak_gflops);
    printf("================================================================\n");

    free(w_data);
    free(x_data);
    free(bias);
    free(y_data);

    return 0;
}
