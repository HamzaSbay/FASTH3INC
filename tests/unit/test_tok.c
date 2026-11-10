#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "fh3/fh3.h"
#include "tokenizer/fh3_tok.h"

static int g_failures = 0;

static const char *resolve_path(const char *path, char *buf, size_t cap) {
    FILE *fp = fopen(path, "rb");
    if (fp) {
        fclose(fp);
        return path;
    }
    snprintf(buf, cap, "../%s", path);
    fp = fopen(buf, "rb");
    if (fp) {
        fclose(fp);
        return buf;
    }
    snprintf(buf, cap, "../../%s", path);
    fp = fopen(buf, "rb");
    if (fp) {
        fclose(fp);
        return buf;
    }
    return path;
}

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return 1; \
    } \
} while (0)

static int test_tok_invalid_args(void) {
    Fh3Tok tok;
    int *ids = NULL;
    int n = 0;

    TEST_ASSERT(fh3_tok_load(NULL, NULL, &tok) == FH3_E_INVALID);
    TEST_ASSERT(fh3_tok_load("nonexistent_vocab.json", "nonexistent_merges.txt", &tok) == FH3_E_IO);
    TEST_ASSERT(fh3_tok_encode(NULL, "hello", &ids, &n) == FH3_E_INVALID);
    TEST_ASSERT(fh3_tok_encode(&tok, NULL, &ids, &n) == FH3_E_INVALID);
    TEST_ASSERT(fh3_tok_encode(&tok, "hello", NULL, &n) == FH3_E_INVALID);
    TEST_ASSERT(fh3_tok_encode(&tok, "hello", &ids, NULL) == FH3_E_INVALID);

    return 0;
}

static int test_tok_parity_corpus(void) {
    char pbuf1[512], pbuf2[512], pbuf3[512];
    const char *vocab_path = resolve_path("tests/fixtures/tok/vocab.json", pbuf1, sizeof(pbuf1));
    const char *merges_path = resolve_path("tests/fixtures/tok/merges.txt", pbuf2, sizeof(pbuf2));
    const char *bin_path = resolve_path("tests/fixtures/tok/corpus_ids.bin", pbuf3, sizeof(pbuf3));

    Fh3Tok tok;
    int rc = fh3_tok_load(vocab_path, merges_path, &tok);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(tok.vocab_size >= 151643);
    TEST_ASSERT(tok.merges_count >= 151000);

    /* Open golden binary fixtures */
    FILE *f = fopen(bin_path, "rb");
    TEST_ASSERT(f != NULL);

    uint32_t n_lines = 0;
    TEST_ASSERT(fread(&n_lines, sizeof(uint32_t), 1, f) == 1);
    TEST_ASSERT(n_lines >= 30);

    int total_tokens = 0;
    for (uint32_t i = 0; i < n_lines; i++) {
        uint32_t text_len = 0;
        TEST_ASSERT(fread(&text_len, sizeof(uint32_t), 1, f) == 1);

        char *text = (char *)malloc(text_len + 1);
        TEST_ASSERT(text != NULL);
        if (text_len > 0) {
            TEST_ASSERT(fread(text, 1, text_len, f) == text_len);
        }
        text[text_len] = '\0';

        uint32_t n_ids = 0;
        TEST_ASSERT(fread(&n_ids, sizeof(uint32_t), 1, f) == 1);

        int32_t *golden_ids = NULL;
        if (n_ids > 0) {
            golden_ids = (int32_t *)malloc(n_ids * sizeof(int32_t));
            TEST_ASSERT(golden_ids != NULL);
            TEST_ASSERT(fread(golden_ids, sizeof(int32_t), n_ids, f) == n_ids);
        }

        /* Encode with C tokenizer */
        int *encoded_ids = NULL;
        int encoded_n = 0;
        rc = fh3_tok_encode(&tok, text, &encoded_ids, &encoded_n);
        TEST_ASSERT(rc == FH3_OK);

        if (encoded_n != (int)n_ids) {
            fprintf(stderr, "[FAIL] Line %u mismatch in count: got %d, expected %u\n", i, encoded_n, n_ids);
            fprintf(stderr, "  Text: %s\n", text);
            fprintf(stderr, "  Got IDs: ");
            for (int k = 0; k < encoded_n; k++) fprintf(stderr, "%d ", encoded_ids[k]);
            fprintf(stderr, "\n  Exp IDs: ");
            for (uint32_t k = 0; k < n_ids; k++) fprintf(stderr, "%d ", golden_ids[k]);
            fprintf(stderr, "\n");
            free(text);
            if (golden_ids) free(golden_ids);
            if (encoded_ids) free(encoded_ids);
            fh3_tok_free(&tok);
            fclose(f);
            g_failures++;
            return 1;
        }

        for (int k = 0; k < encoded_n; k++) {
            if (encoded_ids[k] != golden_ids[k]) {
                fprintf(stderr, "[FAIL] Line %u token %d mismatch: got %d, expected %d\n",
                        i, k, encoded_ids[k], golden_ids[k]);
                fprintf(stderr, "  Text: %s\n", text);
                free(text);
                free(golden_ids);
                free(encoded_ids);
                fh3_tok_free(&tok);
                fclose(f);
                g_failures++;
                return 1;
            }
        }

        total_tokens += encoded_n;
        free(text);
        if (golden_ids) free(golden_ids);
        if (encoded_ids) free(encoded_ids);
    }

    fclose(f);
    printf("test_tok_parity_corpus: verified %u lines, %d tokens bit-identical to HF (bin)\n", n_lines, total_tokens);

    /* Also verify reading tests/fixtures/tok/corpus.txt directly */
    char pbuf4[512];
    const char *txt_path = resolve_path("tests/fixtures/tok/corpus.txt", pbuf4, sizeof(pbuf4));
    FILE *ftxt = fopen(txt_path, "rb");
    TEST_ASSERT(ftxt != NULL);
    char line_buf[8192];
    int txt_line_idx = 0;
    while (fgets(line_buf, sizeof(line_buf), ftxt)) {
        size_t llen = strlen(line_buf);
        while (llen > 0 && (line_buf[llen - 1] == '\r' || line_buf[llen - 1] == '\n')) {
            line_buf[--llen] = '\0';
        }
        int *tids = NULL;
        int tn = 0;
        rc = fh3_tok_encode(&tok, line_buf, &tids, &tn);
        TEST_ASSERT(rc == FH3_OK);
        TEST_ASSERT(tn > 0);
        free(tids);
        txt_line_idx++;
    }
    fclose(ftxt);
    TEST_ASSERT(txt_line_idx == (int)n_lines);
    printf("test_tok_parity_corpus: verified %d lines directly from corpus.txt\n", txt_line_idx);

    /* Also verify empty string edge case */
    int *empty_ids = NULL;
    int empty_n = 0;
    rc = fh3_tok_encode(&tok, "", &empty_ids, &empty_n);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(empty_n == 0);
    if (empty_ids) free(empty_ids);

    fh3_tok_free(&tok);
    return 0;
}

static int test_tok_windows_crlf(void) {
    char pbuf1[512], pbuf2[512];
    const char *vocab_path = resolve_path("tests/fixtures/tok/vocab.json", pbuf1, sizeof(pbuf1));
    const char *merges_path = resolve_path("tests/fixtures/tok/merges.txt", pbuf2, sizeof(pbuf2));

    Fh3Tok tok;
    printf("Loading vocab: %s, merges: %s\n", vocab_path, merges_path);
    int rc = fh3_tok_load(vocab_path, merges_path, &tok);
    if (rc != FH3_OK) {
        fprintf(stderr, "fh3_tok_load failed with rc=%d\n", rc);
    }
    TEST_ASSERT(rc == FH3_OK);

    const char *prompt_crlf = "A peaceful mountain view.\r\nSunlight breaking through pine trees.\r\nCrystal lake below.\r\n";
    const char *prompt_lf   = "A peaceful mountain view.\nSunlight breaking through pine trees.\nCrystal lake below.\n";

    int *ids_lf = NULL;
    int n_lf = 0;
    rc = fh3_tok_encode(&tok, prompt_lf, &ids_lf, &n_lf);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(n_lf > 0);

    /* 1. Default (normalize_newlines = 0): CRLF is NOT normalized to LF */
    int *ids_crlf_raw = NULL;
    int n_crlf_raw = 0;
    rc = fh3_tok_encode(&tok, prompt_crlf, &ids_crlf_raw, &n_crlf_raw);
    TEST_ASSERT(rc == FH3_OK);

    /* Token IDs should differ because CRLF produces \r\n token (319) vs \n token (198) */
    bool are_identical = (n_lf == n_crlf_raw && memcmp(ids_lf, ids_crlf_raw, n_lf * sizeof(int)) == 0);
    TEST_ASSERT(!are_identical);

    /* 2. With normalize_newlines = 1: CRLF prompt tokenizes bit-identically to LF */
    int *ids_crlf_norm = NULL;
    int n_crlf_norm = 0;
    rc = fh3_tok_encode_opt(&tok, prompt_crlf, 1, &ids_crlf_norm, &n_crlf_norm);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(n_crlf_norm == n_lf);
    TEST_ASSERT(memcmp(ids_lf, ids_crlf_norm, n_lf * sizeof(int)) == 0);

    /* 3. Also verify reading a prompt file with CRLF on disk */
    const char *tmp_prompt = "tests/fixtures/tok/test_prompt_crlf.tmp";
    FILE *ftmp = fopen(tmp_prompt, "wb");
    TEST_ASSERT(ftmp != NULL);
    fwrite(prompt_crlf, 1, strlen(prompt_crlf), ftmp);
    fclose(ftmp);

    /* Read back file */
    FILE *fr = fopen(tmp_prompt, "rb");
    TEST_ASSERT(fr != NULL);
    fseek(fr, 0, SEEK_END);
    long fsz = ftell(fr);
    rewind(fr);
    char *fcontent = (char *)malloc(fsz + 1);
    TEST_ASSERT(fcontent != NULL);
    TEST_ASSERT(fread(fcontent, 1, fsz, fr) == (size_t)fsz);
    fcontent[fsz] = '\0';
    fclose(fr);
    remove(tmp_prompt);

    int *file_ids_raw = NULL;
    int file_n_raw = 0;
    rc = fh3_tok_encode(&tok, fcontent, &file_ids_raw, &file_n_raw);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(file_n_raw == n_crlf_raw);
    TEST_ASSERT(memcmp(file_ids_raw, ids_crlf_raw, file_n_raw * sizeof(int)) == 0);

    int *file_ids_norm = NULL;
    int file_n_norm = 0;
    rc = fh3_tok_encode_opt(&tok, fcontent, 1, &file_ids_norm, &file_n_norm);
    TEST_ASSERT(rc == FH3_OK);
    TEST_ASSERT(file_n_norm == n_lf);
    TEST_ASSERT(memcmp(file_ids_norm, ids_lf, file_n_norm * sizeof(int)) == 0);

    free(fcontent);
    free(file_ids_raw);
    free(file_ids_norm);

    printf("test_tok_windows_crlf: verified raw CRLF differs, normalized CRLF matches LF bit-identically (%d tokens)\n", n_lf);

    free(ids_lf);
    free(ids_crlf_raw);
    free(ids_crlf_norm);
    fh3_tok_free(&tok);
    return 0;
}

int main(void) {
    printf("=== Running Tokenizer Unit Tests ===\n");

    if (test_tok_invalid_args() != 0) return 1;
    if (test_tok_parity_corpus() != 0) return 1;
    if (test_tok_windows_crlf() != 0) return 1;

    if (g_failures == 0) {
        printf("ALL TOKENIZER UNIT TESTS PASSED!\n");
        return 0;
    } else {
        fprintf(stderr, "%d tests failed!\n", g_failures);
        return 1;
    }
}
