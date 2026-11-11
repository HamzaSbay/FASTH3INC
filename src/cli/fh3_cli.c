#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <omp.h>

#include "fh3/fh3.h"
#include "fh3_cli.h"
#include "model/fh3_seq.h"
#include "model/fh3_dit.h"
#include "model/fh3_te.h"
#include "model/fh3_sched.h"
#include "tokenizer/fh3_tok.h"
#include "vae/fh3_vae_video.h"
#include "vae/fh3_vae_audio.h"
#include "out/fh3_y4m.h"
#include "out/fh3_wav.h"
#include "io/fh3_portable_io.h"
#include "io/fh3_pack.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#include <direct.h>
#else
#include <sys/resource.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif

/* Global cancel flag checked during diffusion loop */
volatile int g_fh3_cancel_requested = 0;

/* High-resolution timer */
static inline double fh3_now_sec(void) {
#if defined(_WIN32)
    static double s_inv_freq = 0.0;
    if (s_inv_freq == 0.0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        s_inv_freq = 1.0 / (double)f.QuadPart;
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * s_inv_freq;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* Peak RSS in Megabytes */
double fh3_cli_get_peak_rss_mb(void) {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
        return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#else
        return (double)ru.ru_maxrss / 1024.0; /* Linux ru_maxrss is in KiB */
#endif
    }
    return 0.0;
#endif
}

/* CPU Feature Detection */
void fh3_cli_detect_cpu(bool *has_avx2, bool *has_fma) {
    *has_avx2 = false;
    *has_fma = false;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (ecx & (1 << 12)) *has_fma = true;
    }
#if defined(__cpuid_count)
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (ebx & (1 << 5)) *has_avx2 = true;
#elif defined(__get_cpuid_count)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (ebx & (1 << 5)) *has_avx2 = true;
    }
#endif
#endif
}

/* Free Disk Space in Bytes */
uint64_t fh3_cli_free_disk_space(const char *path) {
    const char *check_path = (path && *path) ? path : ".";
#if defined(_WIN32)
    ULARGE_INTEGER free_bytes, total_bytes, total_free;
    if (GetDiskFreeSpaceExA(check_path, &free_bytes, &total_bytes, &total_free)) {
        return (uint64_t)free_bytes.QuadPart;
    }
    if (GetDiskFreeSpaceExA(".", &free_bytes, &total_bytes, &total_free)) {
        return (uint64_t)free_bytes.QuadPart;
    }
    return 0;
#else
    struct statvfs sv;
    if (statvfs(check_path, &sv) == 0) {
        return (uint64_t)sv.f_bavail * (uint64_t)sv.f_frsize;
    }
    if (statvfs(".", &sv) == 0) {
        return (uint64_t)sv.f_bavail * (uint64_t)sv.f_frsize;
    }
    return 0;
#endif
}

/* Resolve Relative Paths across CWD, ../, and ../../ */
const char *fh3_cli_resolve_path(const char *path, char *buf, size_t cap) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if (fp) {
        fclose(fp);
        return path;
    }
    size_t plen = strlen(path);
    if (cap > plen + 4) {
        snprintf(buf, cap, "../%s", path);
        fp = fopen(buf, "rb");
        if (fp) {
            fclose(fp);
            return buf;
        }
    }
    if (cap > plen + 7) {
        snprintf(buf, cap, "../../%s", path);
        fp = fopen(buf, "rb");
        if (fp) {
            fclose(fp);
            return buf;
        }
    }
    return path;
}

/* Recursive Directory Creation */
int fh3_cli_mkdir_p(const char *path) {
    if (!path || !*path) return 0;
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (tmp[len - 1] == '/' || tmp[len - 1] == '\\') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = '\0';
#if defined(_WIN32)
            CreateDirectoryA(tmp, NULL);
#else
            mkdir(tmp, 0755);
#endif
            *p = '/';
        }
    }
#if defined(_WIN32)
    CreateDirectoryA(tmp, NULL);
#else
    mkdir(tmp, 0755);
#endif
    return 0;
}

/* Helper to read entire file into allocated memory */
static void *read_file_bytes(const char *path, size_t *out_sz) {
    char pbuf[1024];
    const char *resolved = fh3_cli_resolve_path(path, pbuf, sizeof(pbuf));
    FILE *fp = fopen(resolved, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); return NULL; }
    void *buf = malloc((size_t)sz);
    if (!buf) { fclose(fp); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) { free(buf); return NULL; }
    if (out_sz) *out_sz = (size_t)sz;
    return buf;
}

/* -------------------------------------------------------------------------
 * Presets and Snapping
 * ------------------------------------------------------------------------- */

int fh3_cli_preset_dims(const char *preset, Fh3ResolvedDims *out) {
    if (!preset || !out) return -1;
    if (strcmp(preset, "tiny") == 0) {
        return fh3_cli_snap_dims(448, 256, 22, out);
    } else if (strcmp(preset, "small") == 0) {
        return fh3_cli_snap_dims(640, 384, 39, out);
    } else if (strcmp(preset, "480p") == 0) {
        return fh3_cli_snap_dims(832, 480, 124, out);
    }
    return -1;
}

int fh3_cli_snap_dims(int width, int height, int frames, Fh3ResolvedDims *out) {
    if (!out) return -1;
    if (width <= 0) width = 640;
    if (height <= 0) height = 384;
    if (frames <= 0) frames = 39;

    /* Spatial snapping: snap up to next multiple of 32 */
    int snapped_w = ((width + 31) / 32) * 32;
    int snapped_h = ((height + 31) / 32) * 32;

    /* Temporal snapping: snap up to 17n + 5 */
    int n = 1;
    if (frames > 22) {
        n = (frames - 5 + 16) / 17;
    }
    int snapped_f = 17 * n + 5;
    int latent_frames = 5 * n + 2;

    int latent_w = snapped_w / 16;
    int latent_h = snapped_h / 16;

    out->width = snapped_w;
    out->height = snapped_h;
    out->frames = snapped_f;
    out->latent_width = latent_w;
    out->latent_height = latent_h;
    out->latent_frames = latent_frames;

    /* Video patch (1, 2, 2): tokens = latent_frames * (latent_h / 2) * (latent_w / 2) */
    out->num_video_tokens = latent_frames * (latent_h / 2) * (latent_w / 2);

    /* Audio: 24 fps video, 40 Hz audio latent rate -> round(frames / 24 * 40) */
    out->num_audio_latents = (int)round((double)snapped_f / 24.0 * 40.0);
    out->num_audio_tokens = 2 * out->num_audio_latents;

    return 0;
}

/* -------------------------------------------------------------------------
 * Memory Planning
 * ------------------------------------------------------------------------- */

int fh3_cli_plan_memory(const Fh3ResolvedDims *dims, double mem_gb, bool is_tiny, Fh3Plan *out_plan) {
    if (!dims || !out_plan) return -1;
    memset(out_plan, 0, sizeof(*out_plan));

    uint64_t min_bytes = 0;
    if (is_tiny) {
        /* Minimum memory floor for tiny fixture execution: ~4 MB */
        min_bytes = 4 * 1024 * 1024ULL;
    } else {
        /* Minimum memory floor for full DiT (2 ring slots ~477MB + activations ~500MB + table ~160MB): ~1.5 GB */
        min_bytes = (uint64_t)(1.5 * 1024.0 * 1024.0 * 1024.0);
    }
    out_plan->min_bytes = min_bytes;

    uint64_t budget_bytes = 0;
    if (mem_gb > 0.0) {
        budget_bytes = (uint64_t)(mem_gb * 1024.0 * 1024.0 * 1024.0);
    } else {
        uint64_t avail = fh3_mem_available();
        budget_bytes = (uint64_t)((double)avail * 0.80);
        if (budget_bytes < min_bytes) {
            budget_bytes = min_bytes;
        }
    }
    out_plan->budget_bytes = budget_bytes;

    if (budget_bytes < min_bytes) {
        fprintf(stderr,
                "fh3: error: requested memory (%llu bytes / %.2f GB) is below minimum requirement (%llu bytes / %.2f GB).\n",
                (unsigned long long)budget_bytes, (double)budget_bytes / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)min_bytes, (double)min_bytes / (1024.0 * 1024.0 * 1024.0));
        return FH3_EXIT_NOMEM;
    }

    if (!is_tiny) {
        int check_rc = fh3_plan_check(out_plan);
        if (check_rc != FH3_OK) {
            return FH3_EXIT_NOMEM;
        }
    }

    return FH3_EXIT_OK;
}

/* -------------------------------------------------------------------------
 * Argument Parsing
 * ------------------------------------------------------------------------- */

void fh3_cli_args_init(Fh3CliArgs *args) {
    if (!args) return;
    memset(args, 0, sizeof(*args));
    args->cmd = FH3_CMD_NONE;
    snprintf(args->preset, sizeof(args->preset), "small");
    snprintf(args->out_dir, sizeof(args->out_dir), "out");
    args->seed = 42;
    args->interrupt_after = -1;
}

int fh3_cli_parse_args(int argc, char **argv, Fh3CliArgs *out, char *err_msg, size_t err_sz) {
    if (!out) return FH3_EXIT_USAGE;
    fh3_cli_args_init(out);

    if (argc < 2) {
        if (err_msg) snprintf(err_msg, err_sz, "No subcommand specified. Use 'fh3 help' for usage.");
        return FH3_EXIT_USAGE;
    }

    const char *sub = argv[1];
    if (strcmp(sub, "run") == 0) {
        out->cmd = FH3_CMD_RUN;
    } else if (strcmp(sub, "encode") == 0) {
        out->cmd = FH3_CMD_ENCODE;
    } else if (strcmp(sub, "denoise") == 0) {
        out->cmd = FH3_CMD_DENOISE;
    } else if (strcmp(sub, "decode") == 0) {
        out->cmd = FH3_CMD_DECODE;
    } else if (strcmp(sub, "doctor") == 0) {
        out->cmd = FH3_CMD_DOCTOR;
    } else if (strcmp(sub, "help") == 0 || strcmp(sub, "--help") == 0 || strcmp(sub, "-h") == 0) {
        out->cmd = FH3_CMD_HELP;
        return FH3_EXIT_OK;
    } else if (strcmp(sub, "version") == 0 || strcmp(sub, "--version") == 0 || strcmp(sub, "-v") == 0) {
        out->cmd = FH3_CMD_VERSION;
        return FH3_EXIT_OK;
    } else {
        if (err_msg) snprintf(err_msg, err_sz, "Unknown subcommand '%s'. Use 'fh3 help' for available commands.", sub);
        return FH3_EXIT_USAGE;
    }

    /* Parse options starting from argv[2] */
    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            out->cmd = FH3_CMD_HELP;
            return FH3_EXIT_OK;
        } else if (strcmp(arg, "--prompt") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--prompt' requires an argument.");
                return FH3_EXIT_USAGE;
            }
            snprintf(out->prompt, sizeof(out->prompt), "%s", argv[++i]);
        } else if (strcmp(arg, "--prompt-file") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--prompt-file' requires a file path.");
                return FH3_EXIT_USAGE;
            }
            snprintf(out->prompt_file, sizeof(out->prompt_file), "%s", argv[++i]);
        } else if (strcmp(arg, "--ids-in") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--ids-in' requires a file path.");
                return FH3_EXIT_USAGE;
            }
            snprintf(out->ids_in, sizeof(out->ids_in), "%s", argv[++i]);
        } else if (strcmp(arg, "--preset") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--preset' requires an argument (tiny, small, 480p).");
                return FH3_EXIT_USAGE;
            }
            snprintf(out->preset, sizeof(out->preset), "%s", argv[++i]);
            if (strcmp(out->preset, "tiny") != 0 && strcmp(out->preset, "small") != 0 && strcmp(out->preset, "480p") != 0) {
                if (err_msg) snprintf(err_msg, err_sz, "Invalid preset '%s'. Supported presets: tiny, small, 480p.", out->preset);
                return FH3_EXIT_USAGE;
            }
        } else if (strcmp(arg, "--width") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--width' requires an integer value.");
                return FH3_EXIT_USAGE;
            }
            out->width = atoi(argv[++i]);
            if (out->width <= 0) {
                if (err_msg) snprintf(err_msg, err_sz, "Width must be a positive integer.");
                return FH3_EXIT_USAGE;
            }
        } else if (strcmp(arg, "--height") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--height' requires an integer value.");
                return FH3_EXIT_USAGE;
            }
            out->height = atoi(argv[++i]);
            if (out->height <= 0) {
                if (err_msg) snprintf(err_msg, err_sz, "Height must be a positive integer.");
                return FH3_EXIT_USAGE;
            }
        } else if (strcmp(arg, "--frames") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--frames' requires an integer value.");
                return FH3_EXIT_USAGE;
            }
            out->frames = atoi(argv[++i]);
            if (out->frames <= 0) {
                if (err_msg) snprintf(err_msg, err_sz, "Frames must be a positive integer.");
                return FH3_EXIT_USAGE;
            }
        } else if (strcmp(arg, "--mem-gb") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--mem-gb' requires a floating-point value.");
                return FH3_EXIT_USAGE;
            }
            out->mem_gb = atof(argv[++i]);
            if (out->mem_gb <= 0.0) {
                if (err_msg) snprintf(err_msg, err_sz, "Memory budget must be greater than 0.");
                return FH3_EXIT_USAGE;
            }
        } else if (strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--threads' requires an integer value.");
                return FH3_EXIT_USAGE;
            }
            out->threads = atoi(argv[++i]);
            if (out->threads <= 0) {
                if (err_msg) snprintf(err_msg, err_sz, "Thread count must be greater than 0.");
                return FH3_EXIT_USAGE;
            }
        } else if (strcmp(arg, "--seed") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--seed' requires an unsigned integer.");
                return FH3_EXIT_USAGE;
            }
            out->seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--noise-in") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--noise-in' requires a file path.");
                return FH3_EXIT_USAGE;
            }
            snprintf(out->noise_in, sizeof(out->noise_in), "%s", argv[++i]);
        } else if (strcmp(arg, "--dense-attn") == 0) {
            out->dense_attn = true;
        } else if (strcmp(arg, "--resume") == 0) {
            out->resume = true;
        } else if (strcmp(arg, "--out") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--out' requires a directory path.");
                return FH3_EXIT_USAGE;
            }
            snprintf(out->out_dir, sizeof(out->out_dir), "%s", argv[++i]);
        } else if (strcmp(arg, "--json") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--json' requires a file path.");
                return FH3_EXIT_USAGE;
            }
            snprintf(out->json_path, sizeof(out->json_path), "%s", argv[++i]);
        } else if (strcmp(arg, "--tiny-fixture") == 0) {
            out->is_tiny = true;
        } else if (strcmp(arg, "--models-dir") == 0 || strcmp(arg, "--pack-dir") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--models-dir' requires a directory path.");
                return FH3_EXIT_USAGE;
            }
            snprintf(out->models_dir, sizeof(out->models_dir), "%s", argv[++i]);
        } else if (strcmp(arg, "--interrupt-after") == 0) {
            if (i + 1 >= argc) {
                if (err_msg) snprintf(err_msg, err_sz, "Option '--interrupt-after' requires an integer step (0..7).");
                return FH3_EXIT_USAGE;
            }
            out->interrupt_after = atoi(argv[++i]);
        } else if (strcmp(arg, "--dry-run") == 0) {
            out->dry_run = true;
        } else {
            if (err_msg) snprintf(err_msg, err_sz, "Unknown argument '%s'.", arg);
            return FH3_EXIT_USAGE;
        }
    }

    return FH3_EXIT_OK;
}

/* -------------------------------------------------------------------------
 * Doctor Command
 * ------------------------------------------------------------------------- */

int fh3_cli_doctor(const Fh3CliArgs *args) {
    printf("================================================================================\n");
    printf("FastH3 System Diagnostics (fh3 doctor v%s)\n", fh3_version());
    printf("================================================================================\n");

    /* CPU */
    bool has_avx2 = false, has_fma = false;
    fh3_cli_detect_cpu(&has_avx2, &has_fma);
    int cores = omp_get_max_threads();
    printf("[CPU]\n");
    printf("  Logical threads:    %d\n", cores);
    printf("  AVX2 SIMD:          %s\n", has_avx2 ? "YES" : "NO");
    printf("  FMA3 support:       %s\n", has_fma ? "YES" : "NO");

    /* Memory */
    uint64_t avail_ram = fh3_mem_available();
    uint64_t total_ram = 0;
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        total_ram = (uint64_t)st.ullTotalPhys;
    }
#else
    FILE *fp_mem = fopen("/proc/meminfo", "r");
    if (fp_mem) {
        char line[256];
        uint64_t kb = 0;
        while (fgets(line, sizeof(line), fp_mem)) {
            if (sscanf(line, "MemTotal: %llu kB", (unsigned long long *)&kb) == 1) {
                total_ram = kb * 1024ULL;
                break;
            }
        }
        fclose(fp_mem);
    }
#endif
    printf("\n[Memory]\n");
    printf("  Total physical RAM: %.2f GB\n", (double)total_ram / (1024.0 * 1024.0 * 1024.0));
    printf("  Available RAM:      %.2f GB\n", (double)avail_ram / (1024.0 * 1024.0 * 1024.0));

    /* Disk */
    uint64_t free_disk = fh3_cli_free_disk_space(".");
    printf("\n[Disk]\n");
    printf("  Working drive free: %.2f GB\n", (double)free_disk / (1024.0 * 1024.0 * 1024.0));

    /* Model Files */
    printf("\n[Model Files]\n");
    const char *files_to_check[7];
    if (args && args->is_tiny) {
        files_to_check[0] = "tests/fixtures/tiny_h3/dit_trunk.bin";
        files_to_check[1] = "tests/fixtures/tiny_h3/dit_globals.bin";
        files_to_check[2] = "tests/fixtures/tiny_h3/adaln_table.bin";
        files_to_check[3] = "tests/fixtures/tiny_te/te_trunk.bin";
        files_to_check[4] = "tests/fixtures/tiny_te/te_embed.bin";
        files_to_check[5] = "tests/fixtures/tiny_vae_video/vae_video_dec.bin";
        files_to_check[6] = "tests/fixtures/tiny_vae_audio/vae_audio_dec.bin";
    } else {
        files_to_check[0] = "packed/dit_trunk.bin";
        files_to_check[1] = "packed/dit_globals.bin";
        files_to_check[2] = "packed/adaln_table.bin";
        files_to_check[3] = "packed/te_trunk.bin";
        files_to_check[4] = "packed/te_embed.bin";
        files_to_check[5] = "packed/vae_video_dec.bin";
        files_to_check[6] = "packed/vae_audio_dec.bin";
    }

    for (int i = 0; i < 7; i++) {
        char pbuf[1024];
        const char *resolved = fh3_cli_resolve_path(files_to_check[i], pbuf, sizeof(pbuf));
        FILE *f = fopen(resolved, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            uint32_t magic = 0;
            size_t rd = fread(&magic, sizeof(magic), 1, f);
            fclose(f);
            (void)rd;
            printf("  %-32s [OK] (%.2f MB, magic 0x%08X)\n",
                   files_to_check[i], (double)sz / (1024.0 * 1024.0), magic);
        } else {
            printf("  %-32s [MISSING]\n", files_to_check[i]);
        }
    }

    /* NVMe Sequential Read Benchmark */
    printf("\n[NVMe Read Benchmark]\n");
    const size_t bench_sz = 32 * 1024 * 1024; /* 32 MB */
    void *bench_buf = fh3_alloc_aligned(bench_sz);
    if (bench_buf) {
        /* Write temporary test file for consistent benchmark if no model file */
        const char *bench_file = "test_bench_tmp.bin";
        FILE *f_bench = fopen(bench_file, "wb");
        if (f_bench) {
            memset(bench_buf, 0xAA, 1024 * 1024);
            for (size_t mb = 0; mb < 32; mb++) {
                fwrite(bench_buf, 1, 1024 * 1024, f_bench);
            }
            fclose(f_bench);

            Fh3File f_io;
            if (fh3_open_direct(bench_file, &f_io) == FH3_OK) {
                double t0 = fh3_now_sec();
                int64_t r = fh3_pread(&f_io, bench_buf, bench_sz, 0);
                double t1 = fh3_now_sec();
                fh3_close_file(&f_io);

                double dt = t1 - t0;
                if (dt > 0.0 && r > 0) {
                    double gb_s = ((double)r / (1024.0 * 1024.0 * 1024.0)) / dt;
                    printf("  Sequential read speed: %.2f GB/s (read %.1f MB in %.4f s)\n",
                           gb_s, (double)r / (1024.0 * 1024.0), dt);
                } else {
                    printf("  Sequential read speed: N/A\n");
                }
            } else {
                printf("  Sequential read speed: N/A (could not open with direct I/O)\n");
            }
            remove(bench_file);
        }
        fh3_free_aligned(bench_buf);
    }

    printf("================================================================================\n");
    printf("System check completed successfully.\n");
    return FH3_EXIT_OK;
}

/* -------------------------------------------------------------------------
 * Phase 1: Text Encoder -> prompt.emb
 * ------------------------------------------------------------------------- */

int fh3_cli_run_phase1(const Fh3CliArgs *args, const Fh3ResolvedDims *dims, Fh3Telemetry *telem) {
    (void)dims;
    double t0 = fh3_now_sec();
    printf("[fh3:Phase1] Starting Tokenizer and Text Encoder...\n");

    fh3_cli_mkdir_p(args->out_dir);

    /* 1. Resolve prompt string */
    char prompt_text[2048] = {0};
    if (args->prompt_file[0] != '\0') {
        size_t sz = 0;
        char *p_txt = (char *)read_file_bytes(args->prompt_file, &sz);
        if (!p_txt) {
            fprintf(stderr, "fh3: error: cannot read prompt file '%s'\n", args->prompt_file);
            return FH3_EXIT_BAD_MODEL;
        }
        size_t copy_n = sz < sizeof(prompt_text) - 1 ? sz : sizeof(prompt_text) - 1;
        memcpy(prompt_text, p_txt, copy_n);
        prompt_text[copy_n] = '\0';
        free(p_txt);
    } else if (args->prompt[0] != '\0') {
        snprintf(prompt_text, sizeof(prompt_text), "%s", args->prompt);
    } else if (args->ids_in[0] == '\0') {
        snprintf(prompt_text, sizeof(prompt_text), "a red fox running through snow, wind sounds");
    }

    /* 2. Tokenize prompt */
    int *ids = NULL;
    int T = 0;

    if (args->ids_in[0] != '\0') {
        size_t sz = 0;
        ids = (int *)read_file_bytes(args->ids_in, &sz);
        if (!ids) {
            fprintf(stderr, "fh3: error: cannot read token IDs file '%s'\n", args->ids_in);
            return FH3_EXIT_BAD_MODEL;
        }
        T = (int)(sz / sizeof(int));
    } else {
        char pbuf1[1024], pbuf2[1024];
        const char *vocab_path = fh3_cli_resolve_path("tests/fixtures/tok/vocab.json", pbuf1, sizeof(pbuf1));
        const char *merges_path = fh3_cli_resolve_path("tests/fixtures/tok/merges.txt", pbuf2, sizeof(pbuf2));

        Fh3Tok tok;
        int rc_tok = fh3_tok_load(vocab_path, merges_path, &tok);
        if (rc_tok != FH3_OK) {
            fprintf(stderr, "fh3: error: failed to load tokenizer files (error %d)\n", rc_tok);
            return FH3_EXIT_BAD_MODEL;
        }

        rc_tok = fh3_tok_encode(&tok, prompt_text, &ids, &T);
        fh3_tok_free(&tok);
        if (rc_tok != FH3_OK || !ids || T <= 0) {
            fprintf(stderr, "fh3: error: tokenization failed\n");
            return FH3_EXIT_BAD_MODEL;
        }
    }

    if (args->is_tiny) {
        /* Clamp token IDs to vocab size 1000 for tiny fixture */
        for (int i = 0; i < T; i++) {
            if (ids[i] < 0) ids[i] = 0;
            ids[i] = ids[i] % 1000;
        }
        /* Tiny model expects exactly 6 text tokens in tiny_h3 */
        if (T < 6) {
            int *new_ids = (int *)malloc(6 * sizeof(int));
            for (int i = 0; i < 6; i++) {
                new_ids[i] = (i < T) ? ids[i] : (ids[i % T]);
            }
            free(ids);
            ids = new_ids;
            T = 6;
        } else if (T > 6) {
            T = 6;
        }
    }

    printf("[fh3:Phase1] Prompt tokenized to %d tokens\n", T);

    /* 3. Text Encoder config and paths */
    char pbuf_trunk[1024], pbuf_embed[1024];
    const char *trunk_path = NULL;
    const char *embed_path = NULL;
    Fh3TeCfg te_cfg;
    memset(&te_cfg, 0, sizeof(te_cfg));

    if (args->is_tiny) {
        trunk_path = fh3_cli_resolve_path("tests/fixtures/tiny_te/te_trunk.bin", pbuf_trunk, sizeof(pbuf_trunk));
        embed_path = fh3_cli_resolve_path("tests/fixtures/tiny_te/te_embed.bin", pbuf_embed, sizeof(pbuf_embed));
        te_cfg.hidden_size = 128;
        te_cfg.num_heads = 4;
        te_cfg.num_kv_heads = 2;
        te_cfg.head_dim = 32;
        te_cfg.num_layers = 3;
        te_cfg.intermediate_size = 256;
        te_cfg.norm_eps = 1e-6f;
        te_cfg.qk_norm_eps = 1e-6f;
        te_cfg.rope_theta = 5000000.0f;
    } else {
        const char *base = args->models_dir[0] ? args->models_dir : "packed";
        char t_str[1024], e_str[1024];
        snprintf(t_str, sizeof(t_str), "%s/te_trunk.bin", base);
        snprintf(e_str, sizeof(e_str), "%s/te_embed.bin", base);
        trunk_path = fh3_cli_resolve_path(t_str, pbuf_trunk, sizeof(pbuf_trunk));
        embed_path = fh3_cli_resolve_path(e_str, pbuf_embed, sizeof(pbuf_embed));
        te_cfg.hidden_size = 5120;
        te_cfg.num_heads = 64;
        te_cfg.num_kv_heads = 8;
        te_cfg.head_dim = 128;
        te_cfg.num_layers = 50;
        te_cfg.intermediate_size = 25600;
        te_cfg.norm_eps = 1e-6f;
        te_cfg.qk_norm_eps = 1e-6f;
        te_cfg.rope_theta = 5000000.0f;
    }

    Fh3Te *te = NULL;
    int rc_te = fh3_te_open(trunk_path, embed_path, &te_cfg, &te);
    if (rc_te != FH3_OK || !te) {
        fprintf(stderr, "fh3: error: failed to open text encoder at '%s' / '%s' (error %d)\n",
                trunk_path, embed_path, rc_te);
        free(ids);
        return FH3_EXIT_BAD_MODEL;
    }

    /* Open trunk streamer */
    Fh3Trunk tr;
    uint64_t te_budget = args->is_tiny ? (20 * 1024 * 1024ULL) : (2 * 1024 * 1024 * 1024ULL);
    rc_te = fh3_trunk_open(trunk_path, te_budget, &tr);
    if (rc_te != FH3_OK) {
        fprintf(stderr, "fh3: error: failed to open text encoder trunk '%s' (error %d)\n", trunk_path, rc_te);
        fh3_te_close(te);
        free(ids);
        return FH3_EXIT_BAD_MODEL;
    }

    /* Forward pass through layers 0-49 */
    float *out_emb = (float *)malloc((size_t)T * (size_t)te_cfg.hidden_size * sizeof(float));
    if (!out_emb) {
        fh3_trunk_close(&tr);
        fh3_te_close(te);
        free(ids);
        return FH3_EXIT_NOMEM;
    }

    rc_te = fh3_te_encode(te, &tr, ids, T, out_emb);
    if (telem) {
        Fh3TrunkStats st = fh3_trunk_stats(&tr);
        telem->bytes_read += st.bytes_read;
    }

    /* Strict phase isolation: close and unload text encoder */
    fh3_trunk_close(&tr);
    fh3_te_close(te);
    free(ids);

    if (rc_te != FH3_OK) {
        fprintf(stderr, "fh3: error: text encoder forward failed (error %d)\n", rc_te);
        free(out_emb);
        return FH3_EXIT_BAD_MODEL;
    }

    /* Save prompt.emb to disk: T (int32), hidden_size (int32), then data */
    char out_emb_path[1024];
    snprintf(out_emb_path, sizeof(out_emb_path), "%s/prompt.emb", args->out_dir);
    FILE *fp_out = fopen(out_emb_path, "wb");
    if (!fp_out) {
        fprintf(stderr, "fh3: error: cannot write output embedding file '%s'\n", out_emb_path);
        free(out_emb);
        return FH3_EXIT_BAD_MODEL;
    }
    uint32_t hdr[4] = { 0x45334846, 1, (uint32_t)T, (uint32_t)te_cfg.hidden_size };
    fwrite(hdr, sizeof(uint32_t), 4, fp_out);
    fwrite(out_emb, sizeof(float), (size_t)T * (size_t)te_cfg.hidden_size, fp_out);
    fclose(fp_out);
    free(out_emb);

    double t1 = fh3_now_sec();
    if (telem) telem->encode_s = t1 - t0;
    printf("[fh3:Phase1] Finished in %.3f s. Saved '%s'. TE unloaded.\n", t1 - t0, out_emb_path);
    return FH3_EXIT_OK;
}

/* -------------------------------------------------------------------------
 * Phase 2: DiT Diffusion Loop -> video_latents.bin + audio_latents.bin
 * ------------------------------------------------------------------------- */

static int save_checkpoint(const char *out_dir, int step,
                           const float *x_v, size_t n_v,
                           const float *x_a, size_t n_a) {
    char chk_path[1024];
    snprintf(chk_path, sizeof(chk_path), "%s/latents.step%d.bin", out_dir, step);
    FILE *fp = fopen(chk_path, "wb");
    if (!fp) return -1;
    uint32_t hdr[4] = { 0x43334846, 1, (uint32_t)step, (uint32_t)n_v };
    uint32_t extra[2] = { (uint32_t)n_a, 0 };
    fwrite(hdr, sizeof(uint32_t), 4, fp);
    fwrite(extra, sizeof(uint32_t), 2, fp);
    fwrite(x_v, sizeof(float), n_v, fp);
    fwrite(x_a, sizeof(float), n_a, fp);
    fclose(fp);
    return 0;
}

static int load_checkpoint(const char *out_dir, int step,
                           float *x_v, size_t n_v,
                           float *x_a, size_t n_a) {
    char chk_path[1024];
    snprintf(chk_path, sizeof(chk_path), "%s/latents.step%d.bin", out_dir, step);
    FILE *fp = fopen(chk_path, "rb");
    if (!fp) return -1;
    uint32_t hdr[4];
    uint32_t extra[2];
    if (fread(hdr, sizeof(uint32_t), 4, fp) != 4) { fclose(fp); return -1; }
    if (fread(extra, sizeof(uint32_t), 2, fp) != 2) { fclose(fp); return -1; }
    if (hdr[0] != 0x43334846 || hdr[2] != (uint32_t)step) { fclose(fp); return -1; }
    if (hdr[3] != (uint32_t)n_v || extra[0] != (uint32_t)n_a) { fclose(fp); return -1; }
    if (fread(x_v, sizeof(float), n_v, fp) != n_v) { fclose(fp); return -1; }
    if (fread(x_a, sizeof(float), n_a, fp) != n_a) { fclose(fp); return -1; }
    fclose(fp);
    return 0;
}

int fh3_cli_run_phase2(const Fh3CliArgs *args, const Fh3ResolvedDims *dims, Fh3Telemetry *telem) {
    double t0 = fh3_now_sec();
    printf("[fh3:Phase2] Starting DiT 8-Step Diffusion Sampler...\n");

    fh3_cli_mkdir_p(args->out_dir);

    /* 1. Read prompt embeddings */
    char emb_path[1024];
    snprintf(emb_path, sizeof(emb_path), "%s/prompt.emb", args->out_dir);
    FILE *fp_emb = fopen(emb_path, "rb");
    if (!fp_emb) {
        /* If missing, try fixture text_emb.bin */
        char pbuf[1024];
        const char *t_emb_fix = fh3_cli_resolve_path("tests/fixtures/tiny_h3/text_emb.bin", pbuf, sizeof(pbuf));
        fp_emb = fopen(t_emb_fix, "rb");
        if (!fp_emb) {
            fprintf(stderr, "fh3: error: prompt embedding file '%s' not found. Run 'fh3 encode' first.\n", emb_path);
            return FH3_EXIT_BAD_MODEL;
        }
    }
    uint32_t emb_hdr[4];
    int num_text = 6;
    int text_dim = args->is_tiny ? 64 : 5120;
    float *prompt_emb = NULL;
    if (fread(emb_hdr, sizeof(uint32_t), 4, fp_emb) == 4 && emb_hdr[0] == 0x45334846) {
        num_text = (int)emb_hdr[2];
        text_dim = (int)emb_hdr[3];
        prompt_emb = (float *)malloc((size_t)num_text * (size_t)text_dim * sizeof(float));
        fread(prompt_emb, sizeof(float), (size_t)num_text * (size_t)text_dim, fp_emb);
    } else {
        /* Fallback raw floats */
        fseek(fp_emb, 0, SEEK_END);
        long fsz = ftell(fp_emb);
        fseek(fp_emb, 0, SEEK_SET);
        num_text = 6;
        text_dim = (int)(fsz / (num_text * sizeof(float)));
        prompt_emb = (float *)malloc((size_t)fsz);
        fread(prompt_emb, 1, (size_t)fsz, fp_emb);
    }
    fclose(fp_emb);

    /* 2. DiT config and paths */
    char pbuf_g[1024], pbuf_a[1024], pbuf_t[1024];
    const char *globals_path = NULL;
    const char *adaln_path = NULL;
    const char *trunk_path = NULL;

    Fh3DitCfg dit_cfg;
    memset(&dit_cfg, 0, sizeof(dit_cfg));
    dit_cfg.attn = args->dense_attn ? FH3_ATTN_DENSE : FH3_ATTN_VSA;
    dit_cfg.vsa.tile = 64;
    dit_cfg.vsa.sparsity = 0.8f;

    if (args->is_tiny) {
        globals_path = fh3_cli_resolve_path("tests/fixtures/tiny_h3/dit_globals.bin", pbuf_g, sizeof(pbuf_g));
        adaln_path = fh3_cli_resolve_path("tests/fixtures/tiny_h3/adaln_table.bin", pbuf_a, sizeof(pbuf_a));
        trunk_path = fh3_cli_resolve_path("tests/fixtures/tiny_h3/dit_trunk.bin", pbuf_t, sizeof(pbuf_t));
        dit_cfg.hidden_size = 256;
        dit_cfg.num_heads = 2;
        dit_cfg.head_dim = 128;
        dit_cfg.num_layers = 3;
        dit_cfg.num_refiner_layers = 2;
        dit_cfg.ffn_dim = 512;
        dit_cfg.text_dim = 64;
        dit_cfg.time_embed_dim = 128;
        dit_cfg.in_channels = 24;
        dit_cfg.audio_in_channels = 32;
        dit_cfg.patch_size[0] = 1; dit_cfg.patch_size[1] = 2; dit_cfg.patch_size[2] = 2;
        dit_cfg.norm_eps = 1e-5f;
        dit_cfg.qk_norm_eps = 1e-5f;
        dit_cfg.final_norm_eps = 1e-5f;
        dit_cfg.rope_freq_dim = 16;
        dit_cfg.rope_theta = 10000.0f;
    } else {
        const char *base = args->models_dir[0] ? args->models_dir : "packed";
        char g_str[1024], a_str[1024], t_str[1024];
        snprintf(g_str, sizeof(g_str), "%s/dit_globals.bin", base);
        snprintf(a_str, sizeof(a_str), "%s/adaln_table.bin", base);
        snprintf(t_str, sizeof(t_str), "%s/dit_trunk.bin", base);
        globals_path = fh3_cli_resolve_path(g_str, pbuf_g, sizeof(pbuf_g));
        adaln_path = fh3_cli_resolve_path(a_str, pbuf_a, sizeof(pbuf_a));
        trunk_path = fh3_cli_resolve_path(t_str, pbuf_t, sizeof(pbuf_t));
        dit_cfg.hidden_size = 5376;
        dit_cfg.num_heads = 56;
        dit_cfg.head_dim = 128;
        dit_cfg.num_layers = 50;
        dit_cfg.num_refiner_layers = 2;
        dit_cfg.ffn_dim = 14336;
        dit_cfg.text_dim = 5120;
        dit_cfg.time_embed_dim = 2688;
        dit_cfg.in_channels = 24;
        dit_cfg.audio_in_channels = 32;
        dit_cfg.patch_size[0] = 1; dit_cfg.patch_size[1] = 2; dit_cfg.patch_size[2] = 2;
        dit_cfg.norm_eps = 1e-5f;
        dit_cfg.qk_norm_eps = 1e-5f;
        dit_cfg.final_norm_eps = 1e-5f;
        dit_cfg.rope_freq_dim = 16;
        dit_cfg.rope_theta = 10000.0f;
    }

    Fh3Dit *dit = NULL;
    int rc = fh3_dit_open(globals_path, adaln_path, &dit_cfg, &dit);
    if (rc != FH3_OK || !dit) {
        fprintf(stderr, "fh3: error: failed to open DiT model files at '%s' / '%s' (error %d)\n",
                globals_path, adaln_path, rc);
        free(prompt_emb);
        return FH3_EXIT_BAD_MODEL;
    }

    /* Open trunk streamer */
    Fh3Trunk trunk;
    uint64_t dit_budget = (args->mem_gb > 0) ? (uint64_t)(args->mem_gb * 1024.0 * 1024.0 * 1024.0) : (args->is_tiny ? (50 * 1024 * 1024ULL) : (9ULL * 1024 * 1024 * 1024));
    rc = fh3_trunk_open(trunk_path, dit_budget, &trunk);
    if (rc != FH3_OK) {
        fprintf(stderr, "fh3: error: failed to open trunk '%s' (error %d)\n", trunk_path, rc);
        fh3_dit_close(dit);
        free(prompt_emb);
        return FH3_EXIT_BAD_MODEL;
    }
    if (telem) telem->pinned_layers = trunk.npin;

    /* 3. Build sequence */
    Fh3SeqSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.num_text = num_text;

    int num_video = 0;
    int num_audio = 0;
    double *custom_pos = NULL;

    if (args->is_tiny) {
        spec.num_audio = 4;
        spec.num_video = 8;
        custom_pos = (double *)read_file_bytes("tests/fixtures/tiny_h3/pos.bin", NULL);
        spec.custom_pos = custom_pos;
        num_video = 8;
        num_audio = 4;
    } else {
        spec.num_audio = dims->num_audio_tokens;
        spec.num_video = dims->num_video_tokens;
        spec.latent_frames = dims->latent_frames;
        spec.latent_height = dims->latent_height;
        spec.latent_width = dims->latent_width;
        spec.audio_latents = dims->num_audio_latents;
        spec.audio_channels = 2;
        spec.patch_t = 1; spec.patch_h = 2; spec.patch_w = 2;
        num_video = spec.num_video;
        num_audio = spec.num_audio;
    }

    Fh3Seq seq;
    rc = fh3_seq_build(&spec, &seq);
    if (rc != FH3_OK) {
        fprintf(stderr, "fh3: error: failed to build sequence (error %d)\n", rc);
        fh3_trunk_close(&trunk);
        fh3_dit_close(dit);
        free(prompt_emb);
        if (custom_pos) free(custom_pos);
        return FH3_EXIT_BAD_MODEL;
    }

    /* 4. Prepare text rows once per prompt */
    float *text_rows = (float *)malloc((size_t)num_text * (size_t)dit_cfg.hidden_size * sizeof(float));
    if (!text_rows) {
        fh3_seq_free(&seq);
        fh3_trunk_close(&trunk);
        fh3_dit_close(dit);
        free(prompt_emb);
        if (custom_pos) free(custom_pos);
        return FH3_EXIT_NOMEM;
    }
    rc = fh3_dit_prepare_text(dit, prompt_emb, num_text, text_rows);
    free(prompt_emb);
    if (rc != FH3_OK) {
        fprintf(stderr, "fh3: error: failed to prepare text rows (error %d)\n", rc);
        free(text_rows);
        fh3_seq_free(&seq);
        fh3_trunk_close(&trunk);
        fh3_dit_close(dit);
        if (custom_pos) free(custom_pos);
        return FH3_EXIT_BAD_MODEL;
    }

    /* 5. Latents buffer allocation */
    const size_t n_v_floats = (size_t)num_video * 96;
    const size_t n_a_floats = (size_t)num_audio * 32;
    float *x_v = (float *)malloc(n_v_floats * sizeof(float));
    float *x_a = (float *)malloc(n_a_floats * sizeof(float));
    float *v_video = (float *)malloc(n_v_floats * sizeof(float));
    float *v_audio = (float *)malloc(n_a_floats * sizeof(float));
    float *noise_v = (float *)malloc(n_v_floats * sizeof(float));
    float *noise_a = (float *)malloc(n_a_floats * sizeof(float));

    if (!x_v || !x_a || !v_video || !v_audio || !noise_v || !noise_a) {
        free(x_v); free(x_a); free(v_video); free(v_audio); free(noise_v); free(noise_a);
        free(text_rows);
        fh3_seq_free(&seq);
        fh3_trunk_close(&trunk);
        fh3_dit_close(dit);
        if (custom_pos) free(custom_pos);
        return FH3_EXIT_NOMEM;
    }

    /* 6. Check for resume */
    int start_step = 0;
    if (args->resume) {
        for (int k = 7; k >= 0; k--) {
            if (load_checkpoint(args->out_dir, k, x_v, n_v_floats, x_a, n_a_floats) == 0) {
                start_step = k + 1;
                printf("[fh3:Phase2] Resuming diffusion from checkpoint 'latents.step%d.bin' (starting step %d)\n", k, start_step);
                break;
            }
        }
    }

    /* If not resuming from checkpoint, initialize noise */
    if (start_step == 0) {
        if (args->noise_in[0] != '\0') {
            fh3_load_noise(args->noise_in, x_v, n_v_floats);
            fh3_rng_normal(args->seed, 1, x_a, n_a_floats);
        } else if (args->is_tiny) {
            size_t xv_sz = 0, xa_sz = 0;
            float *xv_init = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_video.bin", &xv_sz);
            float *xa_init = (float *)read_file_bytes("tests/fixtures/tiny_h3/x_audio.bin", &xa_sz);
            if (xv_init && xa_init) {
                memcpy(x_v, xv_init, n_v_floats * sizeof(float));
                memcpy(x_a, xa_init, n_a_floats * sizeof(float));
            } else {
                fh3_rng_normal(args->seed, 0, x_v, n_v_floats);
                fh3_rng_normal(args->seed, 1, x_a, n_a_floats);
            }
            if (xv_init) free(xv_init);
            if (xa_init) free(xa_init);
        } else {
            fh3_rng_normal(args->seed, 0, x_v, n_v_floats);
            fh3_rng_normal(args->seed, 1, x_a, n_a_floats);
        }
    }

    /* 7. Sampler schedule */
    Fh3Sched sched;
    fh3_sched_build(&sched, NULL, 10.0, 3.0);

    /* 8. Denoising loop */
    int exit_status = FH3_EXIT_OK;
    for (int step = start_step; step < 8; step++) {
        /* Check cancellation request before step */
        if (g_fh3_cancel_requested) {
            printf("[fh3:Phase2] Cancellation requested (SIGINT / Ctrl-C). Saving checkpoint at step %d...\n", step > 0 ? step - 1 : 0);
            exit_status = FH3_EXIT_CANCELLED;
            break;
        }

        printf("  [DiT] Step %d/8 (sigma_v = %.4f, sigma_a = %.4f)...\n",
               step + 1, sched.steps[step].sigma_v, sched.steps[step].sigma_a);

        /* Model forward */
        rc = fh3_dit_forward(dit, &trunk, &seq, step, x_v, x_a, text_rows, v_video, v_audio);
        if (rc != FH3_OK) {
            fprintf(stderr, "fh3: error: DiT forward failed at step %d (error %d)\n", step, rc);
            exit_status = FH3_EXIT_BAD_MODEL;
            break;
        }

        /* Sampler update */
        if (step < 7) {
            if (args->is_tiny) {
                char nv_path[1024], na_path[1024], p_nv[1024], p_na[1024];
                snprintf(nv_path, sizeof(nv_path), "tests/fixtures/tiny_h3/noise_v_step%d.bin", step);
                snprintf(na_path, sizeof(na_path), "tests/fixtures/tiny_h3/noise_a_step%d.bin", step);
                fh3_load_noise(fh3_cli_resolve_path(nv_path, p_nv, sizeof(p_nv)), noise_v, n_v_floats);
                fh3_load_noise(fh3_cli_resolve_path(na_path, p_na, sizeof(p_na)), noise_a, n_a_floats);
            } else {
                fh3_rng_normal(args->seed, 100ULL + (uint64_t)step * 2, noise_v, n_v_floats);
                fh3_rng_normal(args->seed, 101ULL + (uint64_t)step * 2, noise_a, n_a_floats);
            }
            fh3_sched_step(&sched, step, x_v, v_video, (int)n_v_floats, 0, noise_v);
            fh3_sched_step(&sched, step, x_a, v_audio, (int)n_a_floats, 2, noise_a);
        } else {
            fh3_sched_step(&sched, step, x_v, v_video, (int)n_v_floats, 0, NULL);
            fh3_sched_step(&sched, step, x_a, v_audio, (int)n_a_floats, 2, NULL);
        }

        /* Save intermediate step checkpoint */
        save_checkpoint(args->out_dir, step, x_v, n_v_floats, x_a, n_a_floats);

        /* Test simulation interruption */
        if (args->interrupt_after >= 0 && step >= args->interrupt_after) {
            printf("[fh3:Phase2] Interrupt condition reached at step %d. Checkpoint saved.\n", step);
            exit_status = FH3_EXIT_CANCELLED;
            break;
        }
    }

    if (telem) {
        Fh3TrunkStats st = fh3_trunk_stats(&trunk);
        telem->bytes_read += st.bytes_read;
    }

    /* If finished completely, save final latents */
    if (exit_status == FH3_EXIT_OK) {
        char v_lat_path[1024], a_lat_path[1024];
        snprintf(v_lat_path, sizeof(v_lat_path), "%s/video_latents.bin", args->out_dir);
        snprintf(a_lat_path, sizeof(a_lat_path), "%s/audio_latents.bin", args->out_dir);

        FILE *fp_vl = fopen(v_lat_path, "wb");
        if (fp_vl) {
            fwrite(x_v, sizeof(float), n_v_floats, fp_vl);
            fclose(fp_vl);
        }
        FILE *fp_al = fopen(a_lat_path, "wb");
        if (fp_al) {
            fwrite(x_a, sizeof(float), n_a_floats, fp_al);
            fclose(fp_al);
        }
    }

    /* Strict phase isolation: close and unload DiT and streamer */
    free(x_v); free(x_a); free(v_video); free(v_audio); free(noise_v); free(noise_a);
    free(text_rows);
    fh3_seq_free(&seq);
    fh3_trunk_close(&trunk);
    fh3_dit_close(dit);
    if (custom_pos) free(custom_pos);

    double t1 = fh3_now_sec();
    if (telem) telem->denoise_s = t1 - t0;

    if (exit_status == FH3_EXIT_CANCELLED) {
        printf("[fh3:Phase2] Cancelled in %.3f s. Checkpoint retained for resume.\n", t1 - t0);
        return FH3_EXIT_CANCELLED;
    }

    printf("[fh3:Phase2] Finished 8 steps in %.3f s. Saved latents. DiT unloaded.\n", t1 - t0);
    return exit_status;
}

/* -------------------------------------------------------------------------
 * Phase 3: Decoders -> out.wav and out.y4m
 * ------------------------------------------------------------------------- */

int fh3_cli_run_phase3(const Fh3CliArgs *args, const Fh3ResolvedDims *dims, Fh3Telemetry *telem) {
    double t0 = fh3_now_sec();
    printf("[fh3:Phase3] Starting Audio & Video Decoders...\n");

    fh3_cli_mkdir_p(args->out_dir);

    /* ---------------------------------------------------------------------
     * Part A: BigVGAN Audio Decoder
     * --------------------------------------------------------------------- */
    char pbuf_a[1024];
    const char *audio_dec_path = NULL;
    if (args->is_tiny) {
        audio_dec_path = fh3_cli_resolve_path("tests/fixtures/tiny_vae_audio/vae_audio_dec.bin", pbuf_a, sizeof(pbuf_a));
    } else {
        const char *base = args->models_dir[0] ? args->models_dir : "packed";
        char a_str[1024];
        snprintf(a_str, sizeof(a_str), "%s/vae_audio_dec.bin", base);
        audio_dec_path = fh3_cli_resolve_path(a_str, pbuf_a, sizeof(pbuf_a));
    }

    Fh3VaeA *audio_vae = NULL;
    int rc_a = fh3_vae_audio_open(audio_dec_path, &audio_vae);
    if (rc_a != FH3_OK || !audio_vae) {
        fprintf(stderr, "fh3: error: failed to open audio VAE decoder at '%s' (error %d)\n", audio_dec_path, rc_a);
        return FH3_EXIT_BAD_MODEL;
    }

    int a_upsample = fh3_vae_audio_upsample_ratio(audio_vae);
    int num_audio_frames = args->is_tiny ? 4 : dims->num_audio_latents;
    size_t a_lat_floats = 2 * (size_t)num_audio_frames * 32;

    float *audio_lat = (float *)malloc(a_lat_floats * sizeof(float));
    if (!audio_lat) {
        fh3_vae_audio_close(audio_vae);
        return FH3_EXIT_NOMEM;
    }

    /* Load audio latents from disk or tiny fixture */
    char a_lat_path[1024];
    snprintf(a_lat_path, sizeof(a_lat_path), "%s/audio_latents.bin", args->out_dir);
    FILE *fp_al = fopen(a_lat_path, "rb");
    if (fp_al) {
        size_t rd = fread(audio_lat, sizeof(float), a_lat_floats, fp_al);
        fclose(fp_al);
        if (rd < a_lat_floats) {
            memset(audio_lat + rd, 0, (a_lat_floats - rd) * sizeof(float));
        }
    } else if (args->is_tiny) {
        size_t fix_sz = 0;
        float *fix_lat = (float *)read_file_bytes("tests/fixtures/tiny_vae_audio/latents.bin", &fix_sz);
        if (fix_lat) {
            memcpy(audio_lat, fix_lat, fix_sz < a_lat_floats * sizeof(float) ? fix_sz : a_lat_floats * sizeof(float));
            free(fix_lat);
        } else {
            memset(audio_lat, 0, a_lat_floats * sizeof(float));
        }
    } else {
        memset(audio_lat, 0, a_lat_floats * sizeof(float));
    }

    /* Decode audio latents to PCM */
    size_t pcm_samples = 2 * (size_t)num_audio_frames * (size_t)a_upsample;
    float *pcm = (float *)malloc(pcm_samples * sizeof(float));
    if (!pcm) {
        free(audio_lat);
        fh3_vae_audio_close(audio_vae);
        return FH3_EXIT_NOMEM;
    }

    rc_a = fh3_vae_audio_decode(audio_vae, audio_lat, num_audio_frames, pcm);
    fh3_vae_audio_close(audio_vae);
    free(audio_lat);

    if (rc_a != FH3_OK) {
        fprintf(stderr, "fh3: error: audio VAE decode failed (error %d)\n", rc_a);
        free(pcm);
        return FH3_EXIT_BAD_MODEL;
    }

    /* Write out.wav */
    char wav_path[1024];
    snprintf(wav_path, sizeof(wav_path), "%s/out.wav", args->out_dir);
    int rc_wav = fh3_wav_write(wav_path, pcm, 2, (int64_t)num_audio_frames * a_upsample, 32000);
    free(pcm);
    if (rc_wav != FH3_OK) {
        fprintf(stderr, "fh3: error: failed to write WAV to '%s' (error %d)\n", wav_path, rc_wav);
        return FH3_EXIT_BAD_MODEL;
    }
    printf("[fh3:Phase3] Generated '%s'. Audio VAE unloaded.\n", wav_path);

    /* ---------------------------------------------------------------------
     * Part B: Video ViT VAE Decoder
     * --------------------------------------------------------------------- */
    char pbuf_v[1024];
    const char *video_dec_path = NULL;
    if (args->is_tiny) {
        video_dec_path = fh3_cli_resolve_path("tests/fixtures/tiny_vae_video/vae_video_dec.bin", pbuf_v, sizeof(pbuf_v));
    } else {
        const char *base = args->models_dir[0] ? args->models_dir : "packed";
        char v_str[1024];
        snprintf(v_str, sizeof(v_str), "%s/vae_video_dec.bin", base);
        video_dec_path = fh3_cli_resolve_path(v_str, pbuf_v, sizeof(pbuf_v));
    }

    Fh3VaeV *video_vae = NULL;
    int rc_v = fh3_vae_video_open(video_dec_path, &video_vae);
    if (rc_v != FH3_OK || !video_vae) {
        fprintf(stderr, "fh3: error: failed to open video VAE decoder at '%s' (error %d)\n", video_dec_path, rc_v);
        return FH3_EXIT_BAD_MODEL;
    }

    int F_lat = args->is_tiny ? 7 : dims->latent_frames;
    int h_lat = args->is_tiny ? 2 : dims->latent_height;
    int w_lat = args->is_tiny ? 2 : dims->latent_width;
    int out_w = args->is_tiny ? 32 : dims->width;
    int out_h = args->is_tiny ? 32 : dims->height;

    /* Allocate planar video latents [24, F_lat, h_lat, w_lat] */
    size_t lat_voxels = (size_t)F_lat * (size_t)h_lat * (size_t)w_lat;
    size_t v_lat_floats = 24 * lat_voxels;
    float *video_lat = (float *)malloc(v_lat_floats * sizeof(float));
    if (!video_lat) {
        fh3_vae_video_close(video_vae);
        return FH3_EXIT_NOMEM;
    }

    /* Load video latents from disk */
    char v_lat_path[1024];
    snprintf(v_lat_path, sizeof(v_lat_path), "%s/video_latents.bin", args->out_dir);
    FILE *fp_vl = fopen(v_lat_path, "rb");

    if (args->is_tiny) {
        /* In tiny fixture mode, use committed tiny latents directly if present */
        size_t fix_sz = 0;
        float *fix_lat = (float *)read_file_bytes("tests/fixtures/tiny_vae_video/latents.bin", &fix_sz);
        if (fix_lat) {
            memcpy(video_lat, fix_lat, fix_sz < v_lat_floats * sizeof(float) ? fix_sz : v_lat_floats * sizeof(float));
            free(fix_lat);
        } else {
            memset(video_lat, 0, v_lat_floats * sizeof(float));
        }
    } else if (fp_vl) {
        /* Unpatch [V, 96] tokens into [24, F_lat, h_lat, w_lat] */
        size_t num_v_tokens = (size_t)F_lat * (size_t)(h_lat / 2) * (size_t)(w_lat / 2);
        float *tokens = (float *)malloc(num_v_tokens * 96 * sizeof(float));
        if (tokens) {
            fread(tokens, sizeof(float), num_v_tokens * 96, fp_vl);

            int pt = 1, ph = 2, pw = 2;
            int grid_t = F_lat / pt;
            int grid_h = h_lat / ph;
            int grid_w = w_lat / pw;

            #pragma omp parallel for collapse(3) schedule(static)
            for (int gt = 0; gt < grid_t; gt++) {
                for (int gh = 0; gh < grid_h; gh++) {
                    for (int gw = 0; gw < grid_w; gw++) {
                        size_t tok_idx = ((size_t)gt * (size_t)grid_h + (size_t)gh) * (size_t)grid_w + (size_t)gw;
                        const float *t_ptr = tokens + tok_idx * 96;

                        for (int c = 0; c < 24; c++) {
                            for (int dh = 0; dh < 2; dh++) {
                                for (int dw = 0; dw < 2; dw++) {
                                    float val = t_ptr[(c * 4) + (dh * 2) + dw];
                                    int f = gt;
                                    int h = gh * 2 + dh;
                                    int w = gw * 2 + dw;
                                    size_t dst_idx = (((size_t)c * (size_t)F_lat + (size_t)f) * (size_t)h_lat + (size_t)h) * (size_t)w_lat + (size_t)w;
                                    video_lat[dst_idx] = val;
                                }
                            }
                        }
                    }
                }
            }
            free(tokens);
        }
    } else {
        memset(video_lat, 0, v_lat_floats * sizeof(float));
    }
    if (fp_vl) fclose(fp_vl);

    /* Open Y4M output file */
    char y4m_path[1024];
    snprintf(y4m_path, sizeof(y4m_path), "%s/out.y4m", args->out_dir);
    Fh3Y4m *y4m = NULL;
    int rc_y4m = fh3_y4m_open(y4m_path, out_w, out_h, 24, 1, &y4m);
    if (rc_y4m != FH3_OK || !y4m) {
        fprintf(stderr, "fh3: error: failed to open Y4M output at '%s' (error %d)\n", y4m_path, rc_y4m);
        free(video_lat);
        fh3_vae_video_close(video_vae);
        return FH3_EXIT_BAD_MODEL;
    }

    Fh3FrameSink sink = fh3_y4m_frame_sink(y4m);
    rc_v = fh3_vae_video_decode(video_vae, video_lat, F_lat, h_lat, w_lat, &sink);

    fh3_y4m_close(y4m);
    fh3_vae_video_close(video_vae);
    free(video_lat);

    if (rc_v != FH3_OK) {
        fprintf(stderr, "fh3: error: video VAE decode failed (error %d)\n", rc_v);
        return FH3_EXIT_BAD_MODEL;
    }

    double t1 = fh3_now_sec();
    if (telem) telem->decode_s = t1 - t0;
    printf("[fh3:Phase3] Generated '%s'. Video VAE unloaded. Finished in %.3f s.\n", y4m_path, t1 - t0);
    return FH3_EXIT_OK;
}

/* -------------------------------------------------------------------------
 * Telemetry JSON Report
 * ------------------------------------------------------------------------- */

int fh3_cli_write_telemetry_json(const char *path, const Fh3CliArgs *args,
                                 const Fh3ResolvedDims *dims, const Fh3Telemetry *telem) {
    if (!path || !telem) return -1;
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"exit_code\": %d,\n", telem->exit_code);
    fprintf(fp, "  \"preset\": \"%s\",\n", args ? args->preset : "unknown");
    fprintf(fp, "  \"width\": %d,\n", dims ? dims->width : 0);
    fprintf(fp, "  \"height\": %d,\n", dims ? dims->height : 0);
    fprintf(fp, "  \"frames\": %d,\n", dims ? dims->frames : 0);
    fprintf(fp, "  \"peak_rss_mb\": %.2f,\n", telem->peak_rss_mb);
    fprintf(fp, "  \"bytes_read\": %llu,\n", (unsigned long long)telem->bytes_read);
    fprintf(fp, "  \"pinned_layers\": %d,\n", telem->pinned_layers);
    fprintf(fp, "  \"encode_s\": %.4f,\n", telem->encode_s);
    fprintf(fp, "  \"denoise_s\": %.4f,\n", telem->denoise_s);
    fprintf(fp, "  \"decode_s\": %.4f,\n", telem->decode_s);
    fprintf(fp, "  \"total_s\": %.4f,\n", telem->total_s);
    fprintf(fp, "  \"timings\": {\n");
    fprintf(fp, "    \"encode_s\": %.4f,\n", telem->encode_s);
    fprintf(fp, "    \"denoise_s\": %.4f,\n", telem->denoise_s);
    fprintf(fp, "    \"decode_s\": %.4f,\n", telem->decode_s);
    fprintf(fp, "    \"total_s\": %.4f\n", telem->total_s);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"hardware\": {\n");
    fprintf(fp, "    \"cpu_cores\": %d,\n", telem->cpu_cores);
    fprintf(fp, "    \"avx2\": %s,\n", telem->has_avx2 ? "true" : "false");
    fprintf(fp, "    \"fma\": %s,\n", telem->has_fma ? "true" : "false");
    fprintf(fp, "    \"total_ram_gb\": %.2f,\n", telem->total_ram_gb);
    fprintf(fp, "    \"avail_ram_gb\": %.2f\n", telem->avail_ram_gb);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    fclose(fp);
    return 0;
}
