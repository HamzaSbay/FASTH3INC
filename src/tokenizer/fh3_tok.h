#ifndef FH3_TOK_H
#define FH3_TOK_H

#include <stddef.h>
#include <stdint.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fh3MergeEntry {
    uint64_t key;       /* ((uint64_t)id1 << 32) | (uint32_t)id2 */
    int32_t merged_id;  /* Token ID of the merged pair */
    int32_t rank;       /* 0-based rank from merges.txt, or -1 if empty */
} Fh3MergeEntry;

struct Fh3Tok {
    Fh3MergeEntry *merges;
    size_t merges_cap;   /* Capacity of hash table (power of 2) */
    size_t merges_count; /* Number of entries inserted */
    int32_t byte_to_id[256];
    int vocab_size;
};

/* Load vocab.json and merges.txt into Fh3Tok. Returns FH3_OK (0) on success. */
int fh3_tok_load(const char *vocab_json, const char *merges_txt, Fh3Tok *t);

/* Free resources associated with Fh3Tok. */
void fh3_tok_free(Fh3Tok *t);

/* Encode UTF-8 string into token IDs. Allocates *ids via malloc. Returns FH3_OK (0). */
int fh3_tok_encode(const Fh3Tok *t, const char *utf8, int **ids, int *n);

/* Encode with optional CRLF -> LF newline normalization. */
int fh3_tok_encode_opt(const Fh3Tok *t, const char *utf8, int normalize_newlines, int **ids, int *n);

#ifdef __cplusplus
}
#endif

#endif /* FH3_TOK_H */
