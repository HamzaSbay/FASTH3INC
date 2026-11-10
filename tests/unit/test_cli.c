#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "fh3/fh3.h"
#include "cli/fh3_cli.h"
#include "io/fh3_portable_io.h"

static int g_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return 1; \
    } \
} while (0)

/* -------------------------------------------------------------------------
 * Test 1: Preset Dimensions
 * ------------------------------------------------------------------------- */
static int test_cli_presets(void) {
    printf("[RUN] test_cli_presets\n");
    Fh3ResolvedDims dims;

    /* Tiny */
    int rc = fh3_cli_preset_dims("tiny", &dims);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(dims.width == 448);
    TEST_ASSERT(dims.height == 256);
    TEST_ASSERT(dims.frames == 22);
    TEST_ASSERT(dims.latent_width == 28);
    TEST_ASSERT(dims.latent_height == 16);
    TEST_ASSERT(dims.latent_frames == 7);
    TEST_ASSERT(dims.num_video_tokens == 784);
    TEST_ASSERT(dims.num_audio_latents == 37);
    TEST_ASSERT(dims.num_audio_tokens == 74);

    /* Small */
    rc = fh3_cli_preset_dims("small", &dims);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(dims.width == 640);
    TEST_ASSERT(dims.height == 384);
    TEST_ASSERT(dims.frames == 39);
    TEST_ASSERT(dims.latent_width == 40);
    TEST_ASSERT(dims.latent_height == 24);
    TEST_ASSERT(dims.latent_frames == 12);
    TEST_ASSERT(dims.num_video_tokens == 2880);
    TEST_ASSERT(dims.num_audio_latents == 65);
    TEST_ASSERT(dims.num_audio_tokens == 130);

    /* 480p */
    rc = fh3_cli_preset_dims("480p", &dims);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(dims.width == 832);
    TEST_ASSERT(dims.height == 480);
    TEST_ASSERT(dims.frames == 124);
    TEST_ASSERT(dims.latent_width == 52);
    TEST_ASSERT(dims.latent_height == 30);
    TEST_ASSERT(dims.latent_frames == 37);
    TEST_ASSERT(dims.num_video_tokens == 14430);
    TEST_ASSERT(dims.num_audio_latents == 207);
    TEST_ASSERT(dims.num_audio_tokens == 414);

    /* Invalid preset */
    rc = fh3_cli_preset_dims("invalid_preset", &dims);
    TEST_ASSERT(rc != 0);

    printf("[PASS] test_cli_presets\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 2: Resolution Snapping
 * ------------------------------------------------------------------------- */
static int test_cli_snapping(void) {
    printf("[RUN] test_cli_snapping\n");
    Fh3ResolvedDims dims;

    /* Test 500x300, 30 frames -> snaps to 512x320, 39 frames */
    int rc = fh3_cli_snap_dims(500, 300, 30, &dims);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(dims.width == 512);
    TEST_ASSERT(dims.height == 320);
    TEST_ASSERT(dims.frames == 39);
    TEST_ASSERT(dims.latent_width == 32);
    TEST_ASSERT(dims.latent_height == 20);
    TEST_ASSERT(dims.latent_frames == 12);
    TEST_ASSERT(dims.num_video_tokens == 12 * 10 * 16); /* 1920 */

    /* Test exact multiple: 640x384, 39 frames -> stays 640x384, 39 frames */
    rc = fh3_cli_snap_dims(640, 384, 39, &dims);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(dims.width == 640);
    TEST_ASSERT(dims.height == 384);
    TEST_ASSERT(dims.frames == 39);

    /* Test low frame boundary: frames 5 -> snaps to 22 (n=1) */
    rc = fh3_cli_snap_dims(200, 200, 5, &dims);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(dims.frames == 22);
    TEST_ASSERT(dims.latent_frames == 7);

    /* Test snapping up from 23 frames -> snaps to 39 frames (n=2) */
    rc = fh3_cli_snap_dims(256, 256, 23, &dims);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(dims.frames == 39);
    TEST_ASSERT(dims.latent_frames == 12);

    printf("[PASS] test_cli_snapping\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 3: CLI Argument Parsing
 * ------------------------------------------------------------------------- */
static int test_cli_arg_parsing(void) {
    printf("[RUN] test_cli_arg_parsing\n");
    Fh3CliArgs args;
    char err_msg[256];

    /* Subcommand: run with full arguments */
    char *argv1[] = {
        "fh3", "run",
        "--prompt", "a red fox running through snow",
        "--preset", "tiny",
        "--seed", "12345",
        "--threads", "4",
        "--mem-gb", "8.0",
        "--dense-attn",
        "--resume",
        "--out", "my_out",
        "--json", "my_out/report.json",
        "--tiny-fixture"
    };
    int argc1 = sizeof(argv1) / sizeof(argv1[0]);

    int rc = fh3_cli_parse_args(argc1, argv1, &args, err_msg, sizeof(err_msg));
    TEST_ASSERT(rc == FH3_EXIT_OK);
    TEST_ASSERT(args.cmd == FH3_CMD_RUN);
    TEST_ASSERT(strcmp(args.prompt, "a red fox running through snow") == 0);
    TEST_ASSERT(strcmp(args.preset, "tiny") == 0);
    TEST_ASSERT(args.seed == 12345);
    TEST_ASSERT(args.threads == 4);
    TEST_ASSERT(fabs(args.mem_gb - 8.0) < 1e-6);
    TEST_ASSERT(args.dense_attn == true);
    TEST_ASSERT(args.resume == true);
    TEST_ASSERT(strcmp(args.out_dir, "my_out") == 0);
    TEST_ASSERT(strcmp(args.json_path, "my_out/report.json") == 0);
    TEST_ASSERT(args.is_tiny == true);

    /* Subcommand: doctor */
    char *argv2[] = { "fh3", "doctor", "--tiny-fixture" };
    rc = fh3_cli_parse_args(3, argv2, &args, err_msg, sizeof(err_msg));
    TEST_ASSERT(rc == FH3_EXIT_OK);
    TEST_ASSERT(args.cmd == FH3_CMD_DOCTOR);
    TEST_ASSERT(args.is_tiny == true);

    /* Subcommand: encode */
    char *argv3[] = { "fh3", "encode", "--prompt-file", "prompt.txt" };
    rc = fh3_cli_parse_args(4, argv3, &args, err_msg, sizeof(err_msg));
    TEST_ASSERT(rc == FH3_EXIT_OK);
    TEST_ASSERT(args.cmd == FH3_CMD_ENCODE);
    TEST_ASSERT(strcmp(args.prompt_file, "prompt.txt") == 0);

    /* Unknown subcommand -> error 2 */
    char *argv_bad_cmd[] = { "fh3", "nonexistent_subcommand" };
    rc = fh3_cli_parse_args(2, argv_bad_cmd, &args, err_msg, sizeof(err_msg));
    TEST_ASSERT(rc == FH3_EXIT_USAGE);

    /* Unknown option -> error 2 */
    char *argv_bad_opt[] = { "fh3", "run", "--unknown-option-xyz" };
    rc = fh3_cli_parse_args(3, argv_bad_opt, &args, err_msg, sizeof(err_msg));
    TEST_ASSERT(rc == FH3_EXIT_USAGE);

    /* Missing argument value -> error 2 */
    char *argv_missing_val[] = { "fh3", "run", "--width" };
    rc = fh3_cli_parse_args(3, argv_missing_val, &args, err_msg, sizeof(err_msg));
    TEST_ASSERT(rc == FH3_EXIT_USAGE);

    printf("[PASS] test_cli_arg_parsing\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Test 4: Memory Budget Planning & Refusal
 * ------------------------------------------------------------------------- */
static int test_cli_memory_planning(void) {
    printf("[RUN] test_cli_memory_planning\n");
    Fh3ResolvedDims dims;
    fh3_cli_preset_dims("tiny", &dims);
    Fh3Plan plan;

    /* Low memory budget (0.0001 GB ~ 100 KB) must be refused (exit code 3) */
    int rc = fh3_cli_plan_memory(&dims, 0.0001, true, &plan);
    TEST_ASSERT(rc == FH3_EXIT_NOMEM);

    /* Low memory on full model must also be refused */
    rc = fh3_cli_plan_memory(&dims, 0.0001, false, &plan);
    TEST_ASSERT(rc == FH3_EXIT_NOMEM);

    /* Valid budget that fits within physical available RAM */
    uint64_t avail = fh3_mem_available();
    double safe_gb = (double)avail * 0.5 / (1024.0 * 1024.0 * 1024.0);
    if (safe_gb < 0.1) safe_gb = 0.1;
    rc = fh3_cli_plan_memory(&dims, safe_gb, true, &plan);
    TEST_ASSERT(rc == FH3_EXIT_OK);
    TEST_ASSERT(plan.budget_bytes == (uint64_t)(safe_gb * 1024.0 * 1024.0 * 1024.0));

    printf("[PASS] test_cli_memory_planning\n");
    return 0;
}

int main(void) {
    printf("=== Starting test_cli unit test suite ===\n");
    test_cli_presets();
    test_cli_snapping();
    test_cli_arg_parsing();
    test_cli_memory_planning();

    if (g_failures > 0) {
        fprintf(stderr, "=== test_cli FAILED with %d failure(s) ===\n", g_failures);
        return 1;
    }
    printf("=== All test_cli tests PASSED ===\n");
    return 0;
}
