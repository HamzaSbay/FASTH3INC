#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "io/fh3_gguf.h"
#include "io/fh3_portable_io.h"

/* GGUF magic bytes: 'G', 'G', 'U', 'F' = 0x46554747 */
#define GGUF_MAGIC 0x46554747U

/* GGML type block sizes: returns elements_per_block and bytes_per_block */
static bool fh3_ggml_type_size(int type, uint64_t *elements_per_block, uint64_t *bytes_per_block) {
    switch (type) {
        case 0:  /* F32 */     *elements_per_block = 1;   *bytes_per_block = 4;   return true;
        case 1:  /* F16 */     *elements_per_block = 1;   *bytes_per_block = 2;   return true;
        case 2:  /* Q4_0 */    *elements_per_block = 32;  *bytes_per_block = 18;  return true;
        case 3:  /* Q4_1 */    *elements_per_block = 32;  *bytes_per_block = 20;  return true;
        case 6:  /* Q5_0 */    *elements_per_block = 32;  *bytes_per_block = 22;  return true;
        case 7:  /* Q5_1 */    *elements_per_block = 32;  *bytes_per_block = 24;  return true;
        case 8:  /* Q8_0 */    *elements_per_block = 32;  *bytes_per_block = 34;  return true;
        case 9:  /* Q8_1 */    *elements_per_block = 32;  *bytes_per_block = 40;  return true;
        case 10: /* Q2_K */    *elements_per_block = 256; *bytes_per_block = 84;  return true;
        case 11: /* Q3_K */    *elements_per_block = 256; *bytes_per_block = 110; return true;
        case 12: /* Q4_K */    *elements_per_block = 256; *bytes_per_block = 144; return true;
        case 13: /* Q5_K */    *elements_per_block = 256; *bytes_per_block = 176; return true;
        case 14: /* Q6_K */    *elements_per_block = 256; *bytes_per_block = 210; return true;
        case 15: /* Q8_K */    *elements_per_block = 256; *bytes_per_block = 292; return true;
        case 16: /* IQ2_XXS */ *elements_per_block = 256; *bytes_per_block = 66;  return true;
        case 17: /* IQ2_XS */  *elements_per_block = 256; *bytes_per_block = 74;  return true;
        case 18: /* IQ3_XXS */ *elements_per_block = 256; *bytes_per_block = 98;  return true;
        case 19: /* IQ1_S */   *elements_per_block = 256; *bytes_per_block = 50;  return true;
        case 20: /* IQ4_NL */  *elements_per_block = 32;  *bytes_per_block = 18;  return true;
        case 21: /* IQ3_S */   *elements_per_block = 256; *bytes_per_block = 110; return true;
        case 22: /* IQ2_S */   *elements_per_block = 256; *bytes_per_block = 82;  return true;
        case 23: /* IQ4_XS */  *elements_per_block = 256; *bytes_per_block = 136; return true;
        case 24: /* I8 */      *elements_per_block = 1;   *bytes_per_block = 1;   return true;
        case 25: /* I16 */     *elements_per_block = 1;   *bytes_per_block = 2;   return true;
        case 26: /* I32 */     *elements_per_block = 1;   *bytes_per_block = 4;   return true;
        case 27: /* I64 */     *elements_per_block = 1;   *bytes_per_block = 8;   return true;
        case 28: /* F64 */     *elements_per_block = 1;   *bytes_per_block = 8;   return true;
        case 29: /* IQ1_M */   *elements_per_block = 256; *bytes_per_block = 56;  return true;
        case 30: /* BF16 */    *elements_per_block = 1;   *bytes_per_block = 2;   return true;
        default:
            *elements_per_block = 1;
            *bytes_per_block = 1;
            return false;
    }
}

/* Streaming incremental reader */
typedef struct {
    Fh3File *file;
    uint64_t file_size;
    uint64_t buf_file_off;
    size_t buf_len;
    size_t cursor;
    size_t buf_cap;
    uint8_t *buf;
} Fh3Reader;

static void reader_free(Fh3Reader *r) {
    if (r->buf) {
        free(r->buf);
        r->buf = NULL;
    }
    r->buf_cap = 0;
    r->buf_len = 0;
    r->cursor = 0;
}

static int reader_ensure(Fh3Reader *r, size_t needed) {
    if (r->cursor + needed <= r->buf_len) {
        return FH3_OK;
    }

    uint64_t cur_file_pos = r->buf_file_off + (uint64_t)r->cursor;
    if (cur_file_pos + (uint64_t)needed > r->file_size) {
        return FH3_E_TRUNCATED;
    }

    /* Shift unread bytes to start of buffer */
    size_t unread = r->buf_len - r->cursor;
    if (unread > 0 && r->cursor > 0) {
        memmove(r->buf, r->buf + r->cursor, unread);
    }
    r->buf_file_off += (uint64_t)r->cursor;
    r->cursor = 0;
    r->buf_len = unread;

    size_t min_cap = needed + 65536;
    if (r->buf_cap < min_cap) {
        size_t new_cap = r->buf_cap ? (r->buf_cap * 2) : 65536;
        while (new_cap < min_cap) {
            new_cap *= 2;
        }
        uint8_t *new_buf = (uint8_t *)realloc(r->buf, new_cap);
        if (!new_buf) return FH3_E_NOMEM;
        r->buf = new_buf;
        r->buf_cap = new_cap;
    }

    size_t to_read = r->buf_cap - r->buf_len;
    if (r->buf_file_off + (uint64_t)r->buf_len + (uint64_t)to_read > r->file_size) {
        to_read = (size_t)(r->file_size - (r->buf_file_off + (uint64_t)r->buf_len));
    }

    int64_t nread = fh3_pread(r->file, r->buf + r->buf_len, to_read, r->buf_file_off + (uint64_t)r->buf_len);
    if (nread < 0) return FH3_E_IO;
    r->buf_len += (size_t)nread;

    if (r->cursor + needed > r->buf_len) {
        return FH3_E_TRUNCATED;
    }

    return FH3_OK;
}

static int read_u8(Fh3Reader *r, uint8_t *out) {
    int rc = reader_ensure(r, 1);
    if (rc != FH3_OK) return rc;
    *out = r->buf[r->cursor++];
    return FH3_OK;
}

static int read_i8(Fh3Reader *r, int8_t *out) {
    int rc = reader_ensure(r, 1);
    if (rc != FH3_OK) return rc;
    *out = (int8_t)r->buf[r->cursor++];
    return FH3_OK;
}

static int read_u16(Fh3Reader *r, uint16_t *out) {
    int rc = reader_ensure(r, 2);
    if (rc != FH3_OK) return rc;
    const uint8_t *p = r->buf + r->cursor;
    *out = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    r->cursor += 2;
    return FH3_OK;
}

static int read_i16(Fh3Reader *r, int16_t *out) {
    uint16_t u = 0;
    int rc = read_u16(r, &u);
    if (rc != FH3_OK) return rc;
    *out = (int16_t)u;
    return FH3_OK;
}

static int read_u32(Fh3Reader *r, uint32_t *out) {
    int rc = reader_ensure(r, 4);
    if (rc != FH3_OK) return rc;
    const uint8_t *p = r->buf + r->cursor;
    *out = (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
    r->cursor += 4;
    return FH3_OK;
}

static int read_i32(Fh3Reader *r, int32_t *out) {
    uint32_t u = 0;
    int rc = read_u32(r, &u);
    if (rc != FH3_OK) return rc;
    *out = (int32_t)u;
    return FH3_OK;
}

static int read_u64(Fh3Reader *r, uint64_t *out) {
    int rc = reader_ensure(r, 8);
    if (rc != FH3_OK) return rc;
    const uint8_t *p = r->buf + r->cursor;
    *out = (uint64_t)p[0] |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
    r->cursor += 8;
    return FH3_OK;
}

static int read_i64(Fh3Reader *r, int64_t *out) {
    uint64_t u = 0;
    int rc = read_u64(r, &u);
    if (rc != FH3_OK) return rc;
    *out = (int64_t)u;
    return FH3_OK;
}

static int read_f32(Fh3Reader *r, float *out) {
    int rc = reader_ensure(r, 4);
    if (rc != FH3_OK) return rc;
    memcpy(out, r->buf + r->cursor, 4);
    r->cursor += 4;
    return FH3_OK;
}

static int read_f64(Fh3Reader *r, double *out) {
    int rc = reader_ensure(r, 8);
    if (rc != FH3_OK) return rc;
    memcpy(out, r->buf + r->cursor, 8);
    r->cursor += 8;
    return FH3_OK;
}

static int read_str(Fh3Reader *r, char **out_str) {
    uint64_t len = 0;
    int rc = read_u64(r, &len);
    if (rc != FH3_OK) return rc;
    if (len > 100 * 1024 * 1024) return FH3_E_TRUNCATED;

    rc = reader_ensure(r, (size_t)len);
    if (rc != FH3_OK) return rc;

    char *s = (char *)malloc((size_t)len + 1);
    if (!s) return FH3_E_NOMEM;
    memcpy(s, r->buf + r->cursor, (size_t)len);
    s[len] = '\0';
    r->cursor += (size_t)len;

    *out_str = s;
    return FH3_OK;
}

static int read_kv_value(Fh3Reader *r, uint32_t val_type, Fh3GgufKv *kv) {
    kv->type = val_type;
    switch (val_type) {
        case FH3_GGUF_TYPE_UINT8: {
            uint8_t v = 0;
            int rc = read_u8(r, &v);
            if (rc != FH3_OK) return rc;
            kv->val.u64 = v;
            return FH3_OK;
        }
        case FH3_GGUF_TYPE_INT8: {
            int8_t v = 0;
            int rc = read_i8(r, &v);
            if (rc != FH3_OK) return rc;
            kv->val.i64 = v;
            return FH3_OK;
        }
        case FH3_GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            int rc = read_u16(r, &v);
            if (rc != FH3_OK) return rc;
            kv->val.u64 = v;
            return FH3_OK;
        }
        case FH3_GGUF_TYPE_INT16: {
            int16_t v = 0;
            int rc = read_i16(r, &v);
            if (rc != FH3_OK) return rc;
            kv->val.i64 = v;
            return FH3_OK;
        }
        case FH3_GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            int rc = read_u32(r, &v);
            if (rc != FH3_OK) return rc;
            kv->val.u64 = v;
            return FH3_OK;
        }
        case FH3_GGUF_TYPE_INT32: {
            int32_t v = 0;
            int rc = read_i32(r, &v);
            if (rc != FH3_OK) return rc;
            kv->val.i64 = v;
            return FH3_OK;
        }
        case FH3_GGUF_TYPE_FLOAT32: {
            float v = 0;
            int rc = read_f32(r, &v);
            if (rc != FH3_OK) return rc;
            kv->val.f64 = (double)v;
            return FH3_OK;
        }
        case FH3_GGUF_TYPE_BOOL: {
            uint8_t v = 0;
            int rc = read_u8(r, &v);
            if (rc != FH3_OK) return rc;
            kv->val.b = (v != 0);
            return FH3_OK;
        }
        case FH3_GGUF_TYPE_STRING: {
            return read_str(r, &kv->val.str);
        }
        case FH3_GGUF_TYPE_ARRAY: {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            int rc = read_u32(r, &elem_type);
            if (rc != FH3_OK) return rc;
            rc = read_u64(r, &count);
            if (rc != FH3_OK) return rc;

            kv->val.arr.elem_type = elem_type;
            kv->val.arr.count = count;
            kv->val.arr.data = NULL;

            if (count == 0) return FH3_OK;
            if (count > 100 * 1024 * 1024) return FH3_E_TRUNCATED;

            /* Check if integer element type */
            if (elem_type == FH3_GGUF_TYPE_UINT8 || elem_type == FH3_GGUF_TYPE_INT8 ||
                elem_type == FH3_GGUF_TYPE_UINT16 || elem_type == FH3_GGUF_TYPE_INT16 ||
                elem_type == FH3_GGUF_TYPE_UINT32 || elem_type == FH3_GGUF_TYPE_INT32 ||
                elem_type == FH3_GGUF_TYPE_UINT64 || elem_type == FH3_GGUF_TYPE_INT64) {
                int64_t *arr = (int64_t *)malloc((size_t)count * sizeof(int64_t));
                if (!arr) return FH3_E_NOMEM;
                for (uint64_t c = 0; c < count; c++) {
                    int64_t val = 0;
                    if (elem_type == FH3_GGUF_TYPE_UINT8) { uint8_t x; rc = read_u8(r, &x); val = x; }
                    else if (elem_type == FH3_GGUF_TYPE_INT8) { int8_t x; rc = read_i8(r, &x); val = x; }
                    else if (elem_type == FH3_GGUF_TYPE_UINT16) { uint16_t x; rc = read_u16(r, &x); val = x; }
                    else if (elem_type == FH3_GGUF_TYPE_INT16) { int16_t x; rc = read_i16(r, &x); val = x; }
                    else if (elem_type == FH3_GGUF_TYPE_UINT32) { uint32_t x; rc = read_u32(r, &x); val = x; }
                    else if (elem_type == FH3_GGUF_TYPE_INT32) { int32_t x; rc = read_i32(r, &x); val = x; }
                    else if (elem_type == FH3_GGUF_TYPE_UINT64) { uint64_t x; rc = read_u64(r, &x); val = (int64_t)x; }
                    else if (elem_type == FH3_GGUF_TYPE_INT64) { rc = read_i64(r, &val); }

                    if (rc != FH3_OK) {
                        free(arr);
                        return rc;
                    }
                    arr[c] = val;
                }
                kv->val.arr.data = arr;
                return FH3_OK;
            } else if (elem_type == FH3_GGUF_TYPE_STRING) {
                char **arr = (char **)calloc((size_t)count, sizeof(char *));
                if (!arr) return FH3_E_NOMEM;
                for (uint64_t c = 0; c < count; c++) {
                    rc = read_str(r, &arr[c]);
                    if (rc != FH3_OK) {
                        for (uint64_t j = 0; j < c; j++) free(arr[j]);
                        free(arr);
                        return rc;
                    }
                }
                kv->val.arr.data = arr;
                return FH3_OK;
            } else if (elem_type == FH3_GGUF_TYPE_FLOAT32) {
                double *arr = (double *)malloc((size_t)count * sizeof(double));
                if (!arr) return FH3_E_NOMEM;
                for (uint64_t c = 0; c < count; c++) {
                    float f;
                    rc = read_f32(r, &f);
                    if (rc != FH3_OK) { free(arr); return rc; }
                    arr[c] = (double)f;
                }
                kv->val.arr.data = arr;
                return FH3_OK;
            } else if (elem_type == FH3_GGUF_TYPE_FLOAT64) {
                double *arr = (double *)malloc((size_t)count * sizeof(double));
                if (!arr) return FH3_E_NOMEM;
                for (uint64_t c = 0; c < count; c++) {
                    rc = read_f64(r, &arr[c]);
                    if (rc != FH3_OK) { free(arr); return rc; }
                }
                kv->val.arr.data = arr;
                return FH3_OK;
            } else {
                /* Unsupported array element type */
                return FH3_ERR;
            }
        }
        case FH3_GGUF_TYPE_UINT64: {
            return read_u64(r, &kv->val.u64);
        }
        case FH3_GGUF_TYPE_INT64: {
            return read_i64(r, &kv->val.i64);
        }
        case FH3_GGUF_TYPE_FLOAT64: {
            return read_f64(r, &kv->val.f64);
        }
        default:
            return FH3_ERR;
    }
}

const Fh3GgufKv *fh3_gguf_find_kv(const Fh3Gguf *g, const char *key) {
    if (!g || !key || !g->kvs) return NULL;
    for (uint64_t i = 0; i < g->kv_count; i++) {
        if (g->kvs[i].key && strcmp(g->kvs[i].key, key) == 0) {
            return &g->kvs[i];
        }
    }
    return NULL;
}

const Fh3Tensor *fh3_gguf_find(const Fh3Gguf *g, const char *name) {
    if (!g || !name || !g->tensors) return NULL;
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) {
            return &g->tensors[i];
        }
    }
    return NULL;
}

int fh3_gguf_shape(const Fh3Gguf *g, const Fh3Tensor *t, int64_t out[4]) {
    if (!g || !t || !out) return FH3_ERR;

    char key[256];
    snprintf(key, sizeof(key), "comfy.gguf.orig_shape.%s", t->name);
    const Fh3GgufKv *kv = fh3_gguf_find_kv(g, key);
    if (kv && kv->type == FH3_GGUF_TYPE_ARRAY && kv->val.arr.count > 0 && kv->val.arr.data) {
        int n = (int)kv->val.arr.count;
        if (n > 4) n = 4;
        const int64_t *dims = (const int64_t *)kv->val.arr.data;
        for (int i = 0; i < n; i++) {
            out[i] = dims[i];
        }
        for (int i = n; i < 4; i++) {
            out[i] = 0;
        }
        return n;
    }

    int n = t->ndim;
    if (n > 4) n = 4;
    for (int i = 0; i < n; i++) {
        out[i] = t->ne[i];
    }
    for (int i = n; i < 4; i++) {
        out[i] = 0;
    }
    return n;
}

void fh3_gguf_close(Fh3Gguf *g) {
    if (!g) return;

    if (g->kvs) {
        for (uint64_t i = 0; i < g->kv_count; i++) {
            free(g->kvs[i].key);
            if (g->kvs[i].type == FH3_GGUF_TYPE_STRING) {
                free(g->kvs[i].val.str);
            } else if (g->kvs[i].type == FH3_GGUF_TYPE_ARRAY) {
                if (g->kvs[i].val.arr.elem_type == FH3_GGUF_TYPE_STRING && g->kvs[i].val.arr.data) {
                    char **s_arr = (char **)g->kvs[i].val.arr.data;
                    for (uint64_t j = 0; j < g->kvs[i].val.arr.count; j++) {
                        free(s_arr[j]);
                    }
                }
                free(g->kvs[i].val.arr.data);
            }
        }
        free(g->kvs);
        g->kvs = NULL;
    }
    g->kv_count = 0;

    if (g->tensors) {
        free(g->tensors);
        g->tensors = NULL;
    }
    g->tensor_count = 0;

    fh3_close_file(&g->file);
}

int fh3_gguf_open(const char *path, Fh3Gguf *g) {
    if (!path || !g) return FH3_ERR;
    memset(g, 0, sizeof(*g));

    int rc = fh3_open_file(path, &g->file);
    if (rc != FH3_OK) return rc;

    if (g->file.size < 24) {
        fh3_close_file(&g->file);
        return FH3_E_TRUNCATED;
    }

    Fh3Reader r;
    memset(&r, 0, sizeof(r));
    r.file = &g->file;
    r.file_size = g->file.size;

    /* Verify Magic */
    uint32_t magic = 0;
    rc = read_u32(&r, &magic);
    if (rc != FH3_OK) {
        reader_free(&r);
        fh3_close_file(&g->file);
        return rc;
    }
    if (magic != GGUF_MAGIC) {
        reader_free(&r);
        fh3_close_file(&g->file);
        return FH3_E_BAD_MAGIC;
    }

    /* Version */
    uint32_t version = 0;
    rc = read_u32(&r, &version);
    if (rc != FH3_OK) {
        reader_free(&r);
        fh3_close_file(&g->file);
        return rc;
    }
    if (version != 2 && version != 3) {
        reader_free(&r);
        fh3_close_file(&g->file);
        return FH3_E_BAD_VERSION;
    }
    g->version = version;

    /* Tensor count and KV count */
    rc = read_u64(&r, &g->tensor_count);
    if (rc != FH3_OK) {
        reader_free(&r);
        fh3_close_file(&g->file);
        return rc;
    }
    rc = read_u64(&r, &g->kv_count);
    if (rc != FH3_OK) {
        reader_free(&r);
        fh3_close_file(&g->file);
        return rc;
    }

    /* Parse KV metadata */
    if (g->kv_count > 0) {
        g->kvs = (Fh3GgufKv *)calloc((size_t)g->kv_count, sizeof(Fh3GgufKv));
        if (!g->kvs) {
            reader_free(&r);
            fh3_close_file(&g->file);
            return FH3_E_NOMEM;
        }

        for (uint64_t i = 0; i < g->kv_count; i++) {
            rc = read_str(&r, &g->kvs[i].key);
            if (rc != FH3_OK) {
                reader_free(&r);
                fh3_gguf_close(g);
                return rc;
            }

            uint32_t val_type = 0;
            rc = read_u32(&r, &val_type);
            if (rc != FH3_OK) {
                reader_free(&r);
                fh3_gguf_close(g);
                return rc;
            }

            rc = read_kv_value(&r, val_type, &g->kvs[i]);
            if (rc != FH3_OK) {
                reader_free(&r);
                fh3_gguf_close(g);
                return rc;
            }
        }
    }

    /* Parse Tensor info table */
    if (g->tensor_count > 0) {
        g->tensors = (Fh3Tensor *)calloc((size_t)g->tensor_count, sizeof(Fh3Tensor));
        if (!g->tensors) {
            reader_free(&r);
            fh3_gguf_close(g);
            return FH3_E_NOMEM;
        }

        for (uint64_t i = 0; i < g->tensor_count; i++) {
            char *tname = NULL;
            rc = read_str(&r, &tname);
            if (rc != FH3_OK) {
                reader_free(&r);
                fh3_gguf_close(g);
                return rc;
            }
            strncpy(g->tensors[i].name, tname, sizeof(g->tensors[i].name) - 1);
            g->tensors[i].name[sizeof(g->tensors[i].name) - 1] = '\0';
            free(tname);

            uint32_t n_dims = 0;
            rc = read_u32(&r, &n_dims);
            if (rc != FH3_OK) {
                reader_free(&r);
                fh3_gguf_close(g);
                return rc;
            }
            if (n_dims > 4) {
                reader_free(&r);
                fh3_gguf_close(g);
                return FH3_E_INVALID;
            }
            g->tensors[i].ndim = (int)n_dims;

            for (uint32_t d = 0; d < n_dims; d++) {
                uint64_t dim = 0;
                rc = read_u64(&r, &dim);
                if (rc != FH3_OK) {
                    reader_free(&r);
                    fh3_gguf_close(g);
                    return rc;
                }
                if (d < 4) {
                    g->tensors[i].ne[d] = (int64_t)dim;
                }
            }
            for (int d = (int)n_dims; d < 4; d++) {
                g->tensors[i].ne[d] = 0;
            }

            uint32_t t_type = 0;
            rc = read_u32(&r, &t_type);
            if (rc != FH3_OK) {
                reader_free(&r);
                fh3_gguf_close(g);
                return rc;
            }
            g->tensors[i].type = (int)t_type;

            rc = read_u64(&r, &g->tensors[i].off);
            if (rc != FH3_OK) {
                reader_free(&r);
                fh3_gguf_close(g);
                return rc;
            }

            /* Calculate nbytes */
            uint64_t nelem = 1;
            for (int d = 0; d < g->tensors[i].ndim; d++) {
                nelem *= (uint64_t)g->tensors[i].ne[d];
            }
            uint64_t be = 1, bb = 1;
            fh3_ggml_type_size(g->tensors[i].type, &be, &bb);
            g->tensors[i].nbytes = (nelem / be) * bb;
        }
    }

    /* Calculate data_offset */
    uint64_t header_end = r.buf_file_off + (uint64_t)r.cursor;
    uint32_t alignment = 32;
    const Fh3GgufKv *align_kv = fh3_gguf_find_kv(g, "general.alignment");
    if (align_kv) {
        if (align_kv->type == FH3_GGUF_TYPE_UINT32) {
            alignment = (uint32_t)align_kv->val.u64;
        } else if (align_kv->type == FH3_GGUF_TYPE_INT32) {
            alignment = (uint32_t)align_kv->val.i64;
        }
    }
    if (alignment == 0) alignment = 32;
    g->alignment = alignment;
    g->data_offset = (header_end + (uint64_t)alignment - 1) / (uint64_t)alignment * (uint64_t)alignment;

    reader_free(&r);

    /* Validate data_off + max(off + nbytes) <= file_size up front */
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        if (g->tensors[i].off > g->file.size || g->tensors[i].nbytes > g->file.size) {
            fh3_gguf_close(g);
            return FH3_E_TRUNCATED;
        }
        uint64_t tensor_end = g->data_offset + g->tensors[i].off + g->tensors[i].nbytes;
        if (tensor_end < g->data_offset || tensor_end > g->file.size) {
            fh3_gguf_close(g);
            return FH3_E_TRUNCATED;
        }
    }

    return FH3_OK;
}
