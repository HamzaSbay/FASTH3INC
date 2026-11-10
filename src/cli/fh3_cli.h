#ifndef FH3_CLI_H
#define FH3_CLI_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CLI Exit Codes */
#define FH3_EXIT_OK         0
#define FH3_EXIT_USAGE      2
#define FH3_EXIT_NOMEM      3
#define FH3_EXIT_BAD_MODEL  4
#define FH3_EXIT_CANCELLED  5

/* Subcommands */
typedef enum {
    FH3_CMD_NONE = 0,
    FH3_CMD_RUN,
    FH3_CMD_ENCODE,
    FH3_CMD_DENOISE,
    FH3_CMD_DECODE,
    FH3_CMD_DOCTOR,
    FH3_CMD_HELP,
    FH3_CMD_VERSION
} Fh3Subcommand;

/* Resolved Video & Audio Dimensions */
typedef struct Fh3ResolvedDims {
    int width;              /* Snapped pixel width (multiple of 32) */
    int height;             /* Snapped pixel height (multiple of 32) */
    int frames;             /* Snapped pixel frames (17n + 5) */
    int latent_width;       /* width / 16 */
    int latent_height;      /* height / 16 */
    int latent_frames;      /* 5n + 2 */
    int num_video_tokens;   /* latent_frames * (latent_height / 2) * (latent_width / 2) */
    int num_audio_latents;  /* round(frames / 24.0 * 40.0) */
    int num_audio_tokens;   /* 2 * num_audio_latents */
} Fh3ResolvedDims;

/* Telemetry & Metrics */
typedef struct Fh3Telemetry {
    int exit_code;
    double peak_rss_mb;
    uint64_t bytes_read;
    int pinned_layers;
    double encode_s;
    double denoise_s;
    double decode_s;
    double total_s;
    int cpu_cores;
    bool has_avx2;
    bool has_fma;
    double total_ram_gb;
    double avail_ram_gb;
    double free_disk_gb;
} Fh3Telemetry;

/* CLI Arguments */
typedef struct Fh3CliArgs {
    Fh3Subcommand cmd;
    char prompt[1024];
    char prompt_file[512];
    char ids_in[512];
    char preset[32];        /* "tiny", "small", "480p" */
    int width;              /* User-provided or 0 */
    int height;             /* User-provided or 0 */
    int frames;             /* User-provided or 0 */
    double mem_gb;          /* Memory budget in GB (0 = default auto) */
    int threads;            /* OpenMP threads count (0 = default) */
    uint64_t seed;          /* Philox PRNG seed */
    char noise_in[512];     /* Precomputed noise file */
    bool dense_attn;        /* Bypass VSA for full dense attention */
    bool resume;            /* Resume diffusion loop from checkpoint */
    char out_dir[512];      /* Output directory (default "out") */
    char json_path[512];    /* Telemetry JSON report output path */
    bool is_tiny;           /* --tiny-fixture flag */
    char models_dir[512];   /* Optional model directory override */
    int interrupt_after;    /* For test simulation of Ctrl-C / interrupt (-1 = none) */
    bool dry_run;           /* Only print plan/dimensions and exit */
} Fh3CliArgs;

/* Snapping & Dimension Helpers */
int fh3_cli_preset_dims(const char *preset, Fh3ResolvedDims *out);
int fh3_cli_snap_dims(int width, int height, int frames, Fh3ResolvedDims *out);

/* Memory Planning */
int fh3_cli_plan_memory(const Fh3ResolvedDims *dims, double mem_gb, bool is_tiny, Fh3Plan *out_plan);

/* CLI Parsing */
void fh3_cli_args_init(Fh3CliArgs *args);
int fh3_cli_parse_args(int argc, char **argv, Fh3CliArgs *out, char *err_msg, size_t err_sz);

/* Doctor */
int fh3_cli_doctor(const Fh3CliArgs *args);

/* Phase Execution */
int fh3_cli_run_phase1(const Fh3CliArgs *args, const Fh3ResolvedDims *dims, Fh3Telemetry *telem);
int fh3_cli_run_phase2(const Fh3CliArgs *args, const Fh3ResolvedDims *dims, Fh3Telemetry *telem);
int fh3_cli_run_phase3(const Fh3CliArgs *args, const Fh3ResolvedDims *dims, Fh3Telemetry *telem);

/* Telemetry Reporting */
int fh3_cli_write_telemetry_json(const char *path, const Fh3CliArgs *args,
                                 const Fh3ResolvedDims *dims, const Fh3Telemetry *telem);

/* System Utilities */
double fh3_cli_get_peak_rss_mb(void);
void fh3_cli_detect_cpu(bool *has_avx2, bool *has_fma);
uint64_t fh3_cli_free_disk_space(const char *path);
const char *fh3_cli_resolve_path(const char *path, char *buf, size_t cap);
int fh3_cli_mkdir_p(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* FH3_CLI_H */
