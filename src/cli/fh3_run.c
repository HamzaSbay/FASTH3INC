#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <omp.h>

#include "fh3/fh3.h"
#include "fh3_cli.h"
#include "io/fh3_portable_io.h"

extern volatile int g_fh3_cancel_requested;

static void print_usage(void) {
    printf("FastH3-in-C Engine (v%s)\n\n", fh3_version());
    printf("Usage:\n");
    printf("  fh3 run     [options]   Run prompt to synchronized video (.y4m) + stereo audio (.wav)\n");
    printf("  fh3 encode  [options]   Phase 1: Tokenize and encode prompt -> out/prompt.emb\n");
    printf("  fh3 denoise [options]   Phase 2: 8-step DiT diffusion loop -> latents (supports --resume)\n");
    printf("  fh3 decode  [options]   Phase 3: Decode latents to out.y4m + out.wav\n");
    printf("  fh3 doctor  [options]   System capability diagnostics and NVMe benchmark\n");
    printf("  fh3 help                Show this help message\n");
    printf("  fh3 version             Show version information\n\n");
    printf("Options:\n");
    printf("  --prompt <text>         Input prompt text\n");
    printf("  --prompt-file <path>    Read prompt from file\n");
    printf("  --ids-in <path>         Read token IDs directly from binary/text file\n");
    printf("  --preset <name>         Resolution preset: 'tiny' (256x448, 22f), 'small' (384x640, 39f), '480p' (480x832, 124f)\n");
    printf("  --width <W>             Output video width in pixels (snapped up to multiple of 32)\n");
    printf("  --height <H>            Output video height in pixels (snapped up to multiple of 32)\n");
    printf("  --frames <F>            Output video frame count (snapped up to 17n + 5)\n");
    printf("  --mem-gb <N>            RAM budget in GB (e.g. 9.0). Exits code 3 if insufficient\n");
    printf("  --threads <N>           Number of OpenMP threads to use\n");
    printf("  --seed <S>              PRNG seed for Philox normal noise generation\n");
    printf("  --noise-in <path>       Load precomputed initial noise for bit-exact parity testing\n");
    printf("  --dense-attn            Bypass VSA sparse attention in favor of dense attention\n");
    printf("  --resume                Resume diffusion loop from latest latents.step<N>.bin checkpoint\n");
    printf("  --out <dir>             Output directory for generated files (default: 'out')\n");
    printf("  --json <path>           Write detailed execution telemetry JSON report\n");
    printf("  --tiny-fixture          Use synthetic tiny fixtures from tests/fixtures for fast testing\n");
    printf("  --dry-run               Validate configuration and print resolved dimensions without executing\n");
}

static void sigint_handler(int sig) {
    (void)sig;
    g_fh3_cancel_requested = 1;
}

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT || ctrl_type == CTRL_CLOSE_EVENT) {
        g_fh3_cancel_requested = 1;
        return TRUE;
    }
    return FALSE;
}
#endif

int main(int argc, char **argv) {
    /* 1. Install signal and console interrupt handlers */
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
#if defined(_WIN32)
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#endif

    /* 2. Parse command-line arguments */
    Fh3CliArgs args;
    char err_msg[512] = {0};
    int parse_rc = fh3_cli_parse_args(argc, argv, &args, err_msg, sizeof(err_msg));
    if (parse_rc != FH3_EXIT_OK) {
        if (err_msg[0] != '\0') {
            fprintf(stderr, "fh3: %s\n", err_msg);
        }
        return FH3_EXIT_USAGE;
    }

    if (args.cmd == FH3_CMD_HELP) {
        print_usage();
        return FH3_EXIT_OK;
    }

    if (args.cmd == FH3_CMD_VERSION) {
        printf("FastH3-in-C Engine (v%s)\n", fh3_version());
        return FH3_EXIT_OK;
    }

    if (args.cmd == FH3_CMD_DOCTOR) {
        return fh3_cli_doctor(&args);
    }

    /* 3. Configure OpenMP threads if specified */
    if (args.threads > 0) {
        omp_set_num_threads(args.threads);
    }

    /* 4. Resolve dimensions and presets */
    Fh3ResolvedDims dims;
    memset(&dims, 0, sizeof(dims));

    if (args.width > 0 || args.height > 0 || args.frames > 0) {
        /* User specified explicit dimensions -> snap */
        fh3_cli_snap_dims(args.width, args.height, args.frames, &dims);
    } else {
        /* Use preset */
        fh3_cli_preset_dims(args.preset, &dims);
    }

    printf("[fh3] Resolved size: %dx%d, %d frames (latent: %dx%d, %d frames)\n",
           dims.width, dims.height, dims.frames,
           dims.latent_width, dims.latent_height, dims.latent_frames);

    /* 5. Memory Planning & Preflight Refusal */
    Fh3Plan plan;
    int mem_rc = fh3_cli_plan_memory(&dims, args.mem_gb, args.is_tiny, &plan);
    if (mem_rc != FH3_EXIT_OK) {
        return FH3_EXIT_NOMEM;
    }

    /* Check free disk space in output directory target */
    uint64_t free_disk = fh3_cli_free_disk_space(args.out_dir[0] ? args.out_dir : ".");
    uint64_t min_disk = args.is_tiny ? (10 * 1024 * 1024ULL) : (100 * 1024 * 1024ULL);
    if (free_disk > 0 && free_disk < min_disk) {
        fprintf(stderr, "fh3: error: insufficient disk space on output volume (%llu bytes free, %llu required)\n",
                (unsigned long long)free_disk, (unsigned long long)min_disk);
        return FH3_EXIT_NOMEM;
    }

    if (args.dry_run) {
        printf("[fh3] Dry run complete. Configuration and memory plan verified.\n");
        return FH3_EXIT_OK;
    }

    /* 6. Initialize Telemetry */
    Fh3Telemetry telem;
    memset(&telem, 0, sizeof(telem));
    telem.cpu_cores = omp_get_max_threads();
    fh3_cli_detect_cpu(&telem.has_avx2, &telem.has_fma);
    telem.avail_ram_gb = (double)fh3_mem_available() / (1024.0 * 1024.0 * 1024.0);
#if defined(_WIN32)
    MEMORYSTATUSEX st_mem;
    st_mem.dwLength = sizeof(st_mem);
    if (GlobalMemoryStatusEx(&st_mem)) {
        telem.total_ram_gb = (double)st_mem.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
    }
#endif
    telem.free_disk_gb = (double)free_disk / (1024.0 * 1024.0 * 1024.0);

    double t_start = omp_get_wtime();
    int exit_code = FH3_EXIT_OK;

    /* 7. Execute Requested Subcommand with Strict Phase Isolation */
    if (args.cmd == FH3_CMD_ENCODE) {
        exit_code = fh3_cli_run_phase1(&args, &dims, &telem);
    } else if (args.cmd == FH3_CMD_DENOISE) {
        exit_code = fh3_cli_run_phase2(&args, &dims, &telem);
    } else if (args.cmd == FH3_CMD_DECODE) {
        exit_code = fh3_cli_run_phase3(&args, &dims, &telem);
    } else if (args.cmd == FH3_CMD_RUN) {
        /* Phase 1: Text Encoder */
        exit_code = fh3_cli_run_phase1(&args, &dims, &telem);
        if (exit_code == FH3_EXIT_OK) {
            /* Phase 2: DiT Sampler */
            exit_code = fh3_cli_run_phase2(&args, &dims, &telem);
        }
        if (exit_code == FH3_EXIT_OK) {
            /* Phase 3: Decoders */
            exit_code = fh3_cli_run_phase3(&args, &dims, &telem);
        }
    }

    double t_end = omp_get_wtime();
    telem.total_s = t_end - t_start;
    telem.peak_rss_mb = fh3_cli_get_peak_rss_mb();
    telem.exit_code = exit_code;

    /* 8. Write Telemetry JSON report if requested */
    if (args.json_path[0] != '\0') {
        fh3_cli_write_telemetry_json(args.json_path, &args, &dims, &telem);
        printf("[fh3] Telemetry JSON written to '%s'\n", args.json_path);
    }

    if (exit_code == FH3_EXIT_OK) {
        printf("\n================================================================================\n");
        printf("FastH3 Run Completed Successfully in %.3f s\n", telem.total_s);
        printf("  Peak RSS:       %.2f MB\n", telem.peak_rss_mb);
        printf("  Bytes Streamed: %.2f MB\n", (double)telem.bytes_read / (1024.0 * 1024.0));
        printf("  Pinned Layers:  %d\n", telem.pinned_layers);
        printf("  Outputs:        %s/out.y4m, %s/out.wav\n", args.out_dir, args.out_dir);
        printf("================================================================================\n");
    }

    return exit_code;
}
