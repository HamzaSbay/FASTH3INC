#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "io/fh3_pack.h"
#include "io/fh3_portable_io.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

/* -------------------------------------------------------------------------
 * Packed DiT Trunk (FH3P) Reader
 * ------------------------------------------------------------------------- */

int fh3_pack_open(const char *path, Fh3Pack *p) {
    if (!path || !p) return FH3_ERR;
    memset(p, 0, sizeof(*p));

    int rc = fh3_open_file(path, &p->file);
    if (rc != FH3_OK) return rc;

    if (p->file.size < sizeof(Fh3PackHeader)) {
        fh3_pack_close(p);
        return FH3_E_TRUNCATED;
    }

    int64_t rd = fh3_pread(&p->file, &p->header, sizeof(Fh3PackHeader), 0);
    if (rd != (int64_t)sizeof(Fh3PackHeader)) {
        fh3_pack_close(p);
        return FH3_E_IO;
    }

    if (p->header.magic != FH3_PACK_MAGIC) {
        fh3_pack_close(p);
        return FH3_E_BAD_MAGIC;
    }

    if (p->header.version != FH3_PACK_VERSION) {
        fh3_pack_close(p);
        return FH3_E_BAD_VERSION;
    }

    if (p->header.layer_alignment != FH3_PACK_ALIGNMENT || p->header.num_layers == 0) {
        fh3_pack_close(p);
        return FH3_E_INVALID;
    }

    uint64_t needed_index = (uint64_t)p->header.num_layers * sizeof(Fh3PackLayerDesc);
    if (p->header.index_bytes < needed_index ||
        p->header.index_offset + needed_index > p->file.size) {
        fh3_pack_close(p);
        return FH3_E_TRUNCATED;
    }

    p->layers = (Fh3PackLayerDesc *)malloc((size_t)needed_index);
    if (!p->layers) {
        fh3_pack_close(p);
        return FH3_E_NOMEM;
    }

    rd = fh3_pread(&p->file, p->layers, (size_t)needed_index, p->header.index_offset);
    if (rd != (int64_t)needed_index) {
        fh3_pack_close(p);
        return FH3_E_IO;
    }

    p->num_layers = (int)p->header.num_layers;

    /* Optional memory mapping for convenient in-memory inspection */
#if defined(_WIN32)
    p->h_map = CreateFileMappingA(p->file.handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (p->h_map) {
        p->mapped_base = (const uint8_t *)MapViewOfFile((HANDLE)p->h_map, FILE_MAP_READ, 0, 0, 0);
        if (p->mapped_base) {
            p->mapped_size = (size_t)p->file.size;
        }
    }
#else
    void *map = mmap(NULL, (size_t)p->file.size, PROT_READ, MAP_SHARED, p->file.fd, 0);
    if (map != MAP_FAILED) {
        p->mapped_base = (const uint8_t *)map;
        p->mapped_size = (size_t)p->file.size;
    }
#endif

    return FH3_OK;
}

void fh3_pack_close(Fh3Pack *p) {
    if (!p) return;

#if defined(_WIN32)
    if (p->mapped_base) {
        UnmapViewOfFile(p->mapped_base);
        p->mapped_base = NULL;
    }
    if (p->h_map) {
        CloseHandle((HANDLE)p->h_map);
        p->h_map = NULL;
    }
#else
    if (p->mapped_base) {
        munmap((void *)p->mapped_base, p->mapped_size);
        p->mapped_base = NULL;
    }
#endif

    if (p->layers) {
        free(p->layers);
        p->layers = NULL;
    }

    fh3_close_file(&p->file);
    p->num_layers = 0;
}

Fh3LayerView fh3_pack_layer(const Fh3Pack *p, int L, const uint8_t *base) {
    Fh3LayerView v;
    memset(&v, 0, sizeof(v));

    if (!p || L < 0 || L >= p->num_layers || !p->layers) {
        return v;
    }

    const uint8_t *l_base = base;
    if (!l_base) {
        if (p->mapped_base) {
            if (p->layers[L].file_offset + p->layers[L].size_bytes > p->mapped_size) {
                return v;
            }
            l_base = p->mapped_base + p->layers[L].file_offset;
        } else {
            return v;
        }
    }

    const Fh3PackLayerDesc *ld = &p->layers[L];

    /* 0: qkv */
    v.qkv.type = (int)ld->tensors[0].type;
    v.qkv.in   = ld->tensors[0].in_dim;
    v.qkv.out  = ld->tensors[0].out_dim;
    v.qkv.data = l_base + ld->tensors[0].offset_in_layer;

    /* 1: gate_compress */
    v.gate_compress.type = (int)ld->tensors[1].type;
    v.gate_compress.in   = ld->tensors[1].in_dim;
    v.gate_compress.out  = ld->tensors[1].out_dim;
    v.gate_compress.data = l_base + ld->tensors[1].offset_in_layer;

    /* 2: out */
    v.out.type = (int)ld->tensors[2].type;
    v.out.in   = ld->tensors[2].in_dim;
    v.out.out  = ld->tensors[2].out_dim;
    v.out.data = l_base + ld->tensors[2].offset_in_layer;

    /* 3: fc1 */
    v.fc1.type = (int)ld->tensors[3].type;
    v.fc1.in   = ld->tensors[3].in_dim;
    v.fc1.out  = ld->tensors[3].out_dim;
    v.fc1.data = l_base + ld->tensors[3].offset_in_layer;

    /* 4: fc2 */
    v.fc2.type = (int)ld->tensors[4].type;
    v.fc2.in   = ld->tensors[4].in_dim;
    v.fc2.out  = ld->tensors[4].out_dim;
    v.fc2.data = l_base + ld->tensors[4].offset_in_layer;

    /* 5: norm1 */
    v.norm1.type = (int)ld->tensors[5].type;
    v.norm1.in   = ld->tensors[5].in_dim;
    v.norm1.out  = ld->tensors[5].out_dim;
    v.norm1.data = l_base + ld->tensors[5].offset_in_layer;

    /* 6: norm2 */
    v.norm2.type = (int)ld->tensors[6].type;
    v.norm2.in   = ld->tensors[6].in_dim;
    v.norm2.out  = ld->tensors[6].out_dim;
    v.norm2.data = l_base + ld->tensors[6].offset_in_layer;

    /* 7: q_norm */
    v.q_norm.type = (int)ld->tensors[7].type;
    v.q_norm.in   = ld->tensors[7].in_dim;
    v.q_norm.out  = ld->tensors[7].out_dim;
    v.q_norm.data = l_base + ld->tensors[7].offset_in_layer;

    /* 8: k_norm */
    v.k_norm.type = (int)ld->tensors[8].type;
    v.k_norm.in   = ld->tensors[8].in_dim;
    v.k_norm.out  = ld->tensors[8].out_dim;
    v.k_norm.data = l_base + ld->tensors[8].offset_in_layer;

    return v;
}


/* -------------------------------------------------------------------------
 * Folded AdaLN (FH3A) Reader & Schedule Hash
 * ------------------------------------------------------------------------- */

uint64_t fh3_schedule_hash(const int *dmd_steps, int n_steps, double video_shift, double audio_shift) {
    if (!dmd_steps || n_steps <= 0) return 0;

    size_t buf_size = sizeof(int32_t) + (size_t)n_steps * sizeof(int32_t) + sizeof(double) * 2;
    uint8_t stack_buf[256];
    uint8_t *buf = stack_buf;
    if (buf_size > sizeof(stack_buf)) {
        buf = (uint8_t *)malloc(buf_size);
        if (!buf) return 0;
    }

    int32_t n32 = (int32_t)n_steps;
    memcpy(buf, &n32, sizeof(int32_t));
    for (int i = 0; i < n_steps; i++) {
        int32_t s32 = (int32_t)dmd_steps[i];
        memcpy(buf + sizeof(int32_t) + (size_t)i * sizeof(int32_t), &s32, sizeof(int32_t));
    }
    double vs = video_shift;
    double as_ = audio_shift;
    size_t off = sizeof(int32_t) + (size_t)n_steps * sizeof(int32_t);
    memcpy(buf + off, &vs, sizeof(double));
    memcpy(buf + off + sizeof(double), &as_, sizeof(double));

    /* FNV-1a 64-bit */
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < buf_size; i++) {
        h ^= (uint64_t)buf[i];
        h *= 0x100000001b3ULL;
    }

    if (buf != stack_buf) {
        free(buf);
    }
    return h;
}

int fh3_adaln_open(const char *path, uint64_t expect_hash, Fh3Adaln *a) {
    if (!path || !a) return FH3_ERR;
    memset(a, 0, sizeof(*a));

    int rc = fh3_open_file(path, &a->file);
    if (rc != FH3_OK) return rc;

    if (a->file.size < sizeof(Fh3AdalnHeader)) {
        fh3_adaln_close(a);
        return FH3_E_TRUNCATED;
    }

    int64_t rd = fh3_pread(&a->file, &a->header, sizeof(Fh3AdalnHeader), 0);
    if (rd != (int64_t)sizeof(Fh3AdalnHeader)) {
        fh3_adaln_close(a);
        return FH3_E_IO;
    }

    if (a->header.magic != FH3_ADALN_MAGIC) {
        fh3_adaln_close(a);
        return FH3_E_BAD_MAGIC;
    }

    if (a->header.version != FH3_ADALN_VERSION) {
        fh3_adaln_close(a);
        return FH3_E_BAD_VERSION;
    }

    if (expect_hash != 0 && a->header.schedule_hash != expect_hash) {
        fh3_adaln_close(a);
        return FH3_E_INVALID;
    }

    uint64_t needed_trunk = a->header.trunk_offset + a->header.trunk_bytes;
    uint64_t needed_final = a->header.final_offset + a->header.final_bytes;
    if (needed_trunk > a->file.size || needed_final > a->file.size) {
        fh3_adaln_close(a);
        return FH3_E_TRUNCATED;
    }

    a->n_steps = (int)a->header.n_steps;
    a->n_blocks = (int)a->header.n_blocks;
    a->pairs_per_step = (int)a->header.pairs_per_step;
    a->hidden_size = (int)a->header.hidden_size;
    a->schedule_hash = a->header.schedule_hash;

    /* Memory map table */
#if defined(_WIN32)
    a->h_map = CreateFileMappingA(a->file.handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (a->h_map) {
        a->mapped_base = (const uint8_t *)MapViewOfFile((HANDLE)a->h_map, FILE_MAP_READ, 0, 0, 0);
        if (a->mapped_base) {
            a->mapped_size = (size_t)a->file.size;
        }
    }
#else
    void *map = mmap(NULL, (size_t)a->file.size, PROT_READ, MAP_SHARED, a->file.fd, 0);
    if (map != MAP_FAILED) {
        a->mapped_base = (const uint8_t *)map;
        a->mapped_size = (size_t)a->file.size;
    }
#endif

    if (a->mapped_base) {
        a->trunk_data = (const float *)(a->mapped_base + a->header.trunk_offset);
        a->final_data = a->header.final_bytes > 0 ? (const float *)(a->mapped_base + a->header.final_offset) : NULL;
    } else {
        /* Heap fallback if memory mapping unavailable */
        size_t total_floats_bytes = (size_t)(a->header.trunk_bytes + a->header.final_bytes);
        a->allocated_buf = (float *)malloc(total_floats_bytes);
        if (!a->allocated_buf) {
            fh3_adaln_close(a);
            return FH3_E_NOMEM;
        }
        rd = fh3_pread(&a->file, a->allocated_buf, (size_t)a->header.trunk_bytes, a->header.trunk_offset);
        if (rd != (int64_t)a->header.trunk_bytes) {
            fh3_adaln_close(a);
            return FH3_E_IO;
        }
        a->trunk_data = a->allocated_buf;

        if (a->header.final_bytes > 0) {
            float *final_ptr = a->allocated_buf + (a->header.trunk_bytes / sizeof(float));
            rd = fh3_pread(&a->file, final_ptr, (size_t)a->header.final_bytes, a->header.final_offset);
            if (rd != (int64_t)a->header.final_bytes) {
                fh3_adaln_close(a);
                return FH3_E_IO;
            }
            a->final_data = final_ptr;
        }
    }

    return FH3_OK;
}

void fh3_adaln_close(Fh3Adaln *a) {
    if (!a) return;

#if defined(_WIN32)
    if (a->mapped_base) {
        UnmapViewOfFile(a->mapped_base);
        a->mapped_base = NULL;
    }
    if (a->h_map) {
        CloseHandle((HANDLE)a->h_map);
        a->h_map = NULL;
    }
#else
    if (a->mapped_base) {
        munmap((void *)a->mapped_base, a->mapped_size);
        a->mapped_base = NULL;
    }
#endif

    if (a->allocated_buf) {
        free(a->allocated_buf);
        a->allocated_buf = NULL;
    }

    fh3_close_file(&a->file);
    a->trunk_data = NULL;
    a->final_data = NULL;
}

const float *fh3_adaln_row(const Fh3Adaln *a, int step, int block, int pair, int chunk) {
    if (!a) return NULL;
    if (step < 0 || step >= a->n_steps) return NULL;
    if (pair < 0 || pair >= a->pairs_per_step) return NULL;

    /* Trunk blocks: 0 <= block < n_blocks, chunk in 0..5 */
    if (block >= 0 && block < a->n_blocks) {
        if (chunk < 0 || chunk >= (int)FH3_ADALN_CHUNKS_PER_BLOCK) return NULL;
        if (!a->trunk_data) return NULL;

        size_t step_stride = (size_t)a->n_blocks * (size_t)a->pairs_per_step * 6 * (size_t)a->hidden_size;
        size_t block_stride = (size_t)a->pairs_per_step * 6 * (size_t)a->hidden_size;
        size_t pair_stride = 6 * (size_t)a->hidden_size;
        size_t chunk_stride = (size_t)a->hidden_size;

        size_t idx = (size_t)step * step_stride +
                     (size_t)block * block_stride +
                     (size_t)pair * pair_stride +
                     (size_t)chunk * chunk_stride;
        return a->trunk_data + idx;
    }

    /* Final layer: block == a->n_blocks or block == -1, chunk in 0..1 */
    if (block == a->n_blocks || block == -1) {
        if (chunk < 0 || chunk >= (int)FH3_ADALN_CHUNKS_FINAL) return NULL;
        if (!a->final_data) return NULL;

        size_t step_stride = (size_t)a->pairs_per_step * 2 * (size_t)a->hidden_size;
        size_t pair_stride = 2 * (size_t)a->hidden_size;
        size_t chunk_stride = (size_t)a->hidden_size;

        size_t idx = (size_t)step * step_stride +
                     (size_t)pair * pair_stride +
                     (size_t)chunk * chunk_stride;
        return a->final_data + idx;
    }

    return NULL;
}
