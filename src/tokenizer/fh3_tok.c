#include "fh3_tok.h"
#include "fh3_tok_unicode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* SplitMix64 integer hash for fast 64-bit integer table lookup */
static inline uint64_t fh3_hash_u64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* FNV-1a hash for string lookup during vocab load */
static inline uint64_t fh3_hash_str(const char *s, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Temporary string map slot for vocab.json loading */
typedef struct {
    const char *str;
    size_t len;
    int32_t id;
} Fh3StrSlot;

typedef struct {
    Fh3StrSlot *slots;
    size_t cap;
    size_t count;
} Fh3StrMap;

static int fh3_strmap_init(Fh3StrMap *m, size_t cap) {
    m->slots = (Fh3StrSlot *)calloc(cap, sizeof(Fh3StrSlot));
    if (!m->slots) return FH3_E_NOMEM;
    m->cap = cap;
    m->count = 0;
    return FH3_OK;
}

static void fh3_strmap_free(Fh3StrMap *m) {
    if (m->slots) {
        free(m->slots);
        m->slots = NULL;
    }
    m->cap = 0;
    m->count = 0;
}

static int fh3_strmap_insert(Fh3StrMap *m, const char *str, size_t len, int32_t id) {
    uint64_t mask = m->cap - 1;
    uint64_t h = fh3_hash_str(str, len) & mask;
    while (m->slots[h].str != NULL) {
        if (m->slots[h].len == len && memcmp(m->slots[h].str, str, len) == 0) {
            m->slots[h].id = id;
            return FH3_OK;
        }
        h = (h + 1) & mask;
    }
    m->slots[h].str = str;
    m->slots[h].len = len;
    m->slots[h].id = id;
    m->count++;
    return FH3_OK;
}

static int fh3_strmap_find(const Fh3StrMap *m, const char *str, size_t len, int32_t *out_id) {
    uint64_t mask = m->cap - 1;
    uint64_t h = fh3_hash_str(str, len) & mask;
    while (m->slots[h].str != NULL) {
        if (m->slots[h].len == len && memcmp(m->slots[h].str, str, len) == 0) {
            *out_id = m->slots[h].id;
            return 1;
        }
        h = (h + 1) & mask;
    }
    return 0;
}

/* Reads an entire file into a null-terminated memory buffer */
static char *fh3_read_file_text(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t read_bytes = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[read_bytes] = '\0';
    if (out_size) *out_size = read_bytes;
    return buf;
}

/* GPT-2 / Qwen2 byte-to-unicode character string mapping */
static void fh3_build_bytes_to_unicode(char b2u_str[256][8], size_t b2u_len[256]) {
    int bs[256];
    int cs[256];
    int n_bs = 0;

    for (int b = (int)'!'; b <= (int)'~'; b++) bs[n_bs++] = b;
    for (int b = 161; b <= 172; b++) bs[n_bs++] = b;
    for (int b = 174; b <= 255; b++) bs[n_bs++] = b;

    for (int i = 0; i < n_bs; i++) cs[i] = bs[i];

    int extra = 0;
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int i = 0; i < n_bs; i++) {
            if (bs[i] == b) { found = 1; break; }
        }
        if (!found) {
            bs[n_bs] = b;
            cs[n_bs] = 256 + extra;
            extra++;
            n_bs++;
        }
    }

    for (int i = 0; i < 256; i++) {
        int b = bs[i];
        int cp = cs[i];
        if (cp < 0x80) {
            b2u_str[b][0] = (char)cp;
            b2u_str[b][1] = '\0';
            b2u_len[b] = 1;
        } else if (cp < 0x800) {
            b2u_str[b][0] = (char)(0xC0 | (cp >> 6));
            b2u_str[b][1] = (char)(0x80 | (cp & 0x3F));
            b2u_str[b][2] = '\0';
            b2u_len[b] = 2;
        } else {
            /* Not needed for cp <= 323 */
            b2u_str[b][0] = '\0';
            b2u_len[b] = 0;
        }
    }
}

int fh3_tok_load(const char *vocab_json, const char *merges_txt, Fh3Tok *t) {
    if (!vocab_json || !merges_txt || !t) return FH3_E_INVALID;
    memset(t, 0, sizeof(*t));

    /* 1. Read vocab.json */
    size_t vocab_size = 0;
    char *vocab_buf = fh3_read_file_text(vocab_json, &vocab_size);
    if (!vocab_buf) return FH3_E_IO;

    /* Initialize temporary string map: capacity 524288 (power of 2) */
    Fh3StrMap strmap;
    if (fh3_strmap_init(&strmap, 524288) != FH3_OK) {
        free(vocab_buf);
        return FH3_E_NOMEM;
    }

    /* Parse vocab.json in-place: {"token": id, ...} */
    size_t pos = 0;
    while (pos < vocab_size && vocab_buf[pos] != '{') pos++;
    if (pos < vocab_size) pos++; /* Skip '{' */

    int32_t max_id = -1;
    while (pos < vocab_size) {
        while (pos < vocab_size && (vocab_buf[pos] == ' ' || vocab_buf[pos] == '\t' ||
               vocab_buf[pos] == '\r' || vocab_buf[pos] == '\n' || vocab_buf[pos] == ',')) {
            pos++;
        }
        if (pos >= vocab_size || vocab_buf[pos] == '}') break;

        if (vocab_buf[pos] != '"') {
            fh3_strmap_free(&strmap);
            free(vocab_buf);
            return FH3_ERR;
        }
        pos++; /* Skip opening '"' */

        /* Unescape string in-place */
        char *tok_start = &vocab_buf[pos];
        char *dst = tok_start;
        while (pos < vocab_size && vocab_buf[pos] != '"') {
            if (vocab_buf[pos] == '\\') {
                pos++;
                if (pos >= vocab_size) break;
                char c = vocab_buf[pos++];
                if (c == 'n') *dst++ = '\n';
                else if (c == 'r') *dst++ = '\r';
                else if (c == 't') *dst++ = '\t';
                else *dst++ = c;
            } else {
                *dst++ = vocab_buf[pos++];
            }
        }
        size_t tok_len = (size_t)(dst - tok_start);
        if (pos < vocab_size && vocab_buf[pos] == '"') {
            pos++; /* Skip closing '"' BEFORE null-terminating dst */
        }
        *dst = '\0'; /* Null terminate unescaped string */

        while (pos < vocab_size && (vocab_buf[pos] == ' ' || vocab_buf[pos] == '\t' || vocab_buf[pos] == ':')) {
            pos++;
        }

        /* Parse integer ID */
        int32_t id = 0;
        int has_digits = 0;
        while (pos < vocab_size && vocab_buf[pos] >= '0' && vocab_buf[pos] <= '9') {
            id = id * 10 + (vocab_buf[pos] - '0');
            pos++;
            has_digits = 1;
        }
        if (has_digits) {
            fh3_strmap_insert(&strmap, tok_start, tok_len, id);
            if (id > max_id) max_id = id;
        }

        while (pos < vocab_size && (vocab_buf[pos] == ' ' || vocab_buf[pos] == '\t' ||
               vocab_buf[pos] == '\r' || vocab_buf[pos] == '\n' || vocab_buf[pos] == ',')) {
            pos++;
        }
    }
    t->vocab_size = max_id + 1;

    /* 2. Populate byte_to_id mapping */
    char b2u_str[256][8];
    size_t b2u_len[256];
    fh3_build_bytes_to_unicode(b2u_str, b2u_len);

    for (int b = 0; b < 256; b++) {
        int32_t id = -1;
        if (fh3_strmap_find(&strmap, b2u_str[b], b2u_len[b], &id)) {
            t->byte_to_id[b] = id;
        } else {
            t->byte_to_id[b] = 0;
        }
    }

    /* 3. Read merges.txt */
    size_t merges_size = 0;
    char *merges_buf = fh3_read_file_text(merges_txt, &merges_size);
    if (!merges_buf) {
        fh3_strmap_free(&strmap);
        free(vocab_buf);
        return FH3_E_IO;
    }

    /* Allocate merges hash table: 524288 slots */
    size_t merges_cap = 524288;
    t->merges = (Fh3MergeEntry *)malloc(merges_cap * sizeof(Fh3MergeEntry));
    if (!t->merges) {
        free(merges_buf);
        fh3_strmap_free(&strmap);
        free(vocab_buf);
        return FH3_E_NOMEM;
    }
    t->merges_cap = merges_cap;
    t->merges_count = 0;
    for (size_t i = 0; i < merges_cap; i++) {
        t->merges[i].key = 0;
        t->merges[i].merged_id = -1;
        t->merges[i].rank = -1;
    }

    /* Parse merges line by line */
    size_t mpos = 0;
    int32_t current_rank = 0;
    uint64_t mask = merges_cap - 1;

    while (mpos < merges_size) {
        /* Find line start and end */
        while (mpos < merges_size && (merges_buf[mpos] == '\r' || merges_buf[mpos] == '\n')) mpos++;
        if (mpos >= merges_size) break;

        size_t line_start = mpos;
        while (mpos < merges_size && merges_buf[mpos] != '\r' && merges_buf[mpos] != '\n') mpos++;
        size_t line_end = mpos;
        if (mpos < merges_size) mpos++;
        merges_buf[line_end] = '\0';

        char *line = &merges_buf[line_start];
        if (line[0] == '\0' || strncmp(line, "#version", 8) == 0) continue;

        char *sep = strchr(line, ' ');
        if (!sep) continue;

        *sep = '\0';
        char *p1 = line;
        char *p2 = sep + 1;
        size_t len1 = (size_t)(sep - p1);
        size_t len2 = strlen(p2);

        int32_t id1 = -1, id2 = -1;
        if (!fh3_strmap_find(&strmap, p1, len1, &id1)) continue;
        if (!fh3_strmap_find(&strmap, p2, len2, &id2)) continue;

        char merged[1024];
        if (len1 + len2 >= sizeof(merged)) continue;
        memcpy(merged, p1, len1);
        memcpy(merged + len1, p2, len2);
        merged[len1 + len2] = '\0';

        int32_t merged_id = -1;
        if (!fh3_strmap_find(&strmap, merged, len1 + len2, &merged_id)) continue;

        uint64_t key = ((uint64_t)(uint32_t)id1 << 32) | (uint32_t)id2;
        uint64_t h = fh3_hash_u64(key) & mask;
        while (t->merges[h].rank >= 0) {
            if (t->merges[h].key == key) break;
            h = (h + 1) & mask;
        }
        t->merges[h].key = key;
        t->merges[h].merged_id = merged_id;
        t->merges[h].rank = current_rank++;
        t->merges_count++;
    }

    /* Cleanup temporary structures */
    free(merges_buf);
    fh3_strmap_free(&strmap);
    free(vocab_buf);

    return FH3_OK;
}

void fh3_tok_free(Fh3Tok *t) {
    if (!t) return;
    if (t->merges) {
        free(t->merges);
        t->merges = NULL;
    }
    t->merges_cap = 0;
    t->merges_count = 0;
    t->vocab_size = 0;
}

static inline int fh3_tok_find_merge(const Fh3Tok *t, uint64_t key, int32_t *out_mid, int32_t *out_rank) {
    uint64_t mask = t->merges_cap - 1;
    uint64_t h = fh3_hash_u64(key) & mask;
    while (t->merges[h].rank >= 0) {
        if (t->merges[h].key == key) {
            *out_mid = t->merges[h].merged_id;
            *out_rank = t->merges[h].rank;
            return 1;
        }
        h = (h + 1) & mask;
    }
    return 0;
}

typedef struct {
    uint32_t cp;
    uint32_t byte_offset;
    uint8_t byte_len;
    uint8_t flags;
} Fh3Char;

/* Pre-tokenizer matching case-insensitive contractions */
static inline size_t match_contraction(const Fh3Char *chars, size_t pos, size_t n) {
    if (chars[pos].cp != '\'') return 0;
    if (pos + 1 >= n) return 0;

    uint32_t c1 = chars[pos + 1].cp;
    if (c1 >= 'A' && c1 <= 'Z') c1 += 32;

    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        return 2;
    }
    if (pos + 2 < n) {
        uint32_t c2 = chars[pos + 2].cp;
        if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
        if (c1 == 'r' && c2 == 'e') return 3;
        if (c1 == 'v' && c2 == 'e') return 3;
        if (c1 == 'l' && c2 == 'l') return 3;
    }
    return 0;
}

int fh3_tok_encode_opt(const Fh3Tok *t, const char *utf8, int normalize_newlines, int **out_ids, int *out_n) {
    if (!t || !utf8 || !out_ids || !out_n) return FH3_E_INVALID;
    *out_ids = NULL;
    *out_n = 0;

    size_t utf8_len = strlen(utf8);
    if (utf8_len == 0) {
        return FH3_OK;
    }

    /* Handle newline normalization if requested */
    char *norm_buf = NULL;
    const char *src = utf8;
    if (normalize_newlines) {
        norm_buf = (char *)malloc(utf8_len + 1);
        if (!norm_buf) return FH3_E_NOMEM;
        size_t di = 0;
        for (size_t si = 0; si < utf8_len; si++) {
            if (utf8[si] == '\r' && si + 1 < utf8_len && utf8[si + 1] == '\n') {
                norm_buf[di++] = '\n';
                si++;
            } else {
                norm_buf[di++] = utf8[si];
            }
        }
        norm_buf[di] = '\0';
        src = norm_buf;
        utf8_len = di;
    }

    /* Decode UTF-8 string into Fh3Char array */
    size_t chars_cap = utf8_len + 16;
    Fh3Char *chars = (Fh3Char *)malloc(chars_cap * sizeof(Fh3Char));
    if (!chars) {
        if (norm_buf) free(norm_buf);
        return FH3_E_NOMEM;
    }

    size_t n_chars = 0;
    size_t byte_idx = 0;
    while (byte_idx < utf8_len) {
        uint8_t b0 = (uint8_t)src[byte_idx];
        uint32_t cp = 0;
        uint8_t blen = 1;

        if (b0 < 0x80) {
            cp = b0;
            blen = 1;
        } else if ((b0 & 0xE0) == 0xC0 && byte_idx + 1 < utf8_len) {
            cp = ((b0 & 0x1F) << 6) | ((uint8_t)src[byte_idx + 1] & 0x3F);
            blen = 2;
        } else if ((b0 & 0xF0) == 0xE0 && byte_idx + 2 < utf8_len) {
            cp = ((b0 & 0x0F) << 12) | (((uint8_t)src[byte_idx + 1] & 0x3F) << 6) |
                 ((uint8_t)src[byte_idx + 2] & 0x3F);
            blen = 3;
        } else if ((b0 & 0xF8) == 0xF0 && byte_idx + 3 < utf8_len) {
            cp = ((b0 & 0x07) << 18) | (((uint8_t)src[byte_idx + 1] & 0x3F) << 12) |
                 (((uint8_t)src[byte_idx + 2] & 0x3F) << 6) | ((uint8_t)src[byte_idx + 3] & 0x3F);
            blen = 4;
        } else {
            cp = b0;
            blen = 1;
        }

        if (n_chars >= chars_cap) {
            chars_cap *= 2;
            Fh3Char *nc = (Fh3Char *)realloc(chars, chars_cap * sizeof(Fh3Char));
            if (!nc) {
                free(chars);
                if (norm_buf) free(norm_buf);
                return FH3_E_NOMEM;
            }
            chars = nc;
        }

        chars[n_chars].cp = cp;
        chars[n_chars].byte_offset = (uint32_t)byte_idx;
        chars[n_chars].byte_len = blen;
        chars[n_chars].flags = fh3_unicode_flags(cp);
        n_chars++;
        byte_idx += blen;
    }

    /* Output token ID array */
    size_t ids_cap = 64;
    int32_t *res_ids = (int32_t *)malloc(ids_cap * sizeof(int32_t));
    if (!res_ids) {
        free(chars);
        if (norm_buf) free(norm_buf);
        return FH3_E_NOMEM;
    }
    size_t total_ids = 0;

    /* Pre-tokenizer chunk loop matching Qwen2 regex */
    size_t c_pos = 0;
    while (c_pos < n_chars) {
        size_t match_len = 0;

        /* Branch 1: (?i:'s|'t|'re|'ve|'m|'ll|'d) */
        size_t cont_len = match_contraction(chars, c_pos, n_chars);
        if (cont_len > 0) {
            match_len = cont_len;
        }
        /* Branch 2: [^\r\n\p{L}\p{N}]?\p{L}+ */
        else if (chars[c_pos].flags & FH3_IS_L) {
            size_t end = c_pos + 1;
            while (end < n_chars && (chars[end].flags & FH3_IS_L)) end++;
            match_len = end - c_pos;
        } else if (chars[c_pos].cp != '\r' && chars[c_pos].cp != '\n' && !(chars[c_pos].flags & FH3_IS_N) &&
                   c_pos + 1 < n_chars && (chars[c_pos + 1].flags & FH3_IS_L)) {
            size_t end = c_pos + 2;
            while (end < n_chars && (chars[end].flags & FH3_IS_L)) end++;
            match_len = end - c_pos;
        }
        /* Branch 3: \p{N} */
        else if (chars[c_pos].flags & FH3_IS_N) {
            match_len = 1;
        }
        /* Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n]* */
        else {
            size_t p4 = c_pos;
            if (chars[p4].cp == ' ') p4++;
            size_t count_non_sln = 0;
            while (p4 < n_chars && !(chars[p4].flags & (FH3_IS_S | FH3_IS_L | FH3_IS_N))) {
                count_non_sln++;
                p4++;
            }
            if (count_non_sln > 0) {
                while (p4 < n_chars && (chars[p4].cp == '\r' || chars[p4].cp == '\n')) {
                    p4++;
                }
                match_len = p4 - c_pos;
            }
            /* Branch 5, 6, 7: whitespace */
            else if (chars[c_pos].flags & FH3_IS_S) {
                size_t run_end = c_pos;
                while (run_end < n_chars && (chars[run_end].flags & FH3_IS_S)) {
                    run_end++;
                }
                int last_rn_end = -1;
                for (size_t i = c_pos; i < run_end; i++) {
                    if (chars[i].cp == '\r' || chars[i].cp == '\n') {
                        size_t j = i;
                        while (j < run_end && (chars[j].cp == '\r' || chars[j].cp == '\n')) {
                            j++;
                        }
                        last_rn_end = (int)j;
                        i = j - 1;
                    }
                }
                if (last_rn_end != -1) {
                    match_len = (size_t)last_rn_end - c_pos;
                } else if (run_end == n_chars) {
                    match_len = run_end - c_pos;
                } else if (run_end - c_pos >= 2) {
                    match_len = (run_end - 1) - c_pos;
                } else {
                    match_len = run_end - c_pos;
                }
            } else {
                /* Fallback single char */
                match_len = 1;
            }
        }

        if (match_len == 0) match_len = 1;

        /* Get chunk byte slice */
        size_t b_start = chars[c_pos].byte_offset;
        size_t b_end = chars[c_pos + match_len - 1].byte_offset + chars[c_pos + match_len - 1].byte_len;
        size_t chunk_bytes_len = b_end - b_start;
        const uint8_t *chunk_raw = (const uint8_t *)&src[b_start];

        /* Map bytes to initial token IDs (no VLAs: stack buffer with dynamic fallback) */
        int32_t stack_buf[256];
        int32_t *chunk_ids = stack_buf;
        if (chunk_bytes_len > 256) {
            chunk_ids = (int32_t *)malloc(chunk_bytes_len * sizeof(int32_t));
            if (!chunk_ids) {
                free(res_ids);
                free(chars);
                if (norm_buf) free(norm_buf);
                return FH3_E_NOMEM;
            }
        }

        for (size_t i = 0; i < chunk_bytes_len; i++) {
            chunk_ids[i] = t->byte_to_id[chunk_raw[i]];
        }
        size_t chunk_n = chunk_bytes_len;

        /* BPE merge loop */
        while (chunk_n > 1) {
            int32_t best_rank = 0x7FFFFFFF;
            int best_idx = -1;
            int32_t best_new_id = -1;

            for (size_t i = 0; i < chunk_n - 1; i++) {
                uint64_t key = ((uint64_t)(uint32_t)chunk_ids[i] << 32) | (uint32_t)chunk_ids[i + 1];
                int32_t mid = -1, rk = -1;
                if (fh3_tok_find_merge(t, key, &mid, &rk)) {
                    if (rk < best_rank) {
                        best_rank = rk;
                        best_idx = (int)i;
                        best_new_id = mid;
                    }
                }
            }

            if (best_idx < 0) {
                break;
            }

            chunk_ids[best_idx] = best_new_id;
            memmove(&chunk_ids[best_idx + 1], &chunk_ids[best_idx + 2],
                    (chunk_n - (size_t)best_idx - 2) * sizeof(int32_t));
            chunk_n--;
        }

        /* Append merged IDs to output list */
        if (total_ids + chunk_n > ids_cap) {
            while (total_ids + chunk_n > ids_cap) ids_cap *= 2;
            int32_t *nids = (int32_t *)realloc(res_ids, ids_cap * sizeof(int32_t));
            if (!nids) {
                if (chunk_ids != stack_buf) free(chunk_ids);
                free(res_ids);
                free(chars);
                if (norm_buf) free(norm_buf);
                return FH3_E_NOMEM;
            }
            res_ids = nids;
        }

        memcpy(&res_ids[total_ids], chunk_ids, chunk_n * sizeof(int32_t));
        total_ids += chunk_n;

        if (chunk_ids != stack_buf) {
            free(chunk_ids);
        }

        c_pos += match_len;
    }

    free(chars);
    if (norm_buf) free(norm_buf);

    *out_ids = res_ids;
    *out_n = (int)total_ids;
    return FH3_OK;
}

int fh3_tok_encode(const Fh3Tok *t, const char *utf8, int **ids, int *n) {
    return fh3_tok_encode_opt(t, utf8, 0, ids, n);
}
