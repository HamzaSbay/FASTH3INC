#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "io/fh3_trunk.h"
#include "io/fh3_portable_io.h"

/* High-resolution timer */
static inline double fh3_trunk_time_sec(void) {
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

/* Background worker thread */
#if defined(_WIN32)
static DWORD WINAPI fh3_trunk_worker_win32(LPVOID param) {
    Fh3Trunk *t = (Fh3Trunk *)param;
    while (1) {
        WaitForSingleObject(t->start_event, INFINITE);
        if (t->worker_stop) {
            break;
        }

        int L = t->worker_req_layer;
        int slot = t->worker_req_slot;
        uint8_t *dst = t->ring_buf[slot];
        uint64_t off = t->pack.layers[L].file_offset;
        size_t sz = (size_t)t->pack.layers[L].size_bytes;

        double t0 = fh3_trunk_time_sec();
        int64_t rd = fh3_pread(&t->direct_file, dst, sz, off);
        double t1 = fh3_trunk_time_sec();

        t->worker_read_bytes = (rd > 0) ? rd : 0;
        t->prefetch_io_time = t1 - t0;

        SetEvent(t->done_event);
    }
    return 0;
}
#else
static void *fh3_trunk_worker_posix(void *param) {
    Fh3Trunk *t = (Fh3Trunk *)param;
    pthread_mutex_lock(&t->mutex);
    while (!t->worker_stop) {
        while (!t->worker_has_work && !t->worker_stop) {
            pthread_cond_wait(&t->start_cond, &t->mutex);
        }
        if (t->worker_stop) break;

        int L = t->worker_req_layer;
        int slot = t->worker_req_slot;
        uint8_t *dst = t->ring_buf[slot];
        uint64_t off = t->pack.layers[L].file_offset;
        size_t sz = (size_t)t->pack.layers[L].size_bytes;
        pthread_mutex_unlock(&t->mutex);

        double t0 = fh3_trunk_time_sec();
        int64_t rd = fh3_pread(&t->direct_file, dst, sz, off);
        double t1 = fh3_trunk_time_sec();

        pthread_mutex_lock(&t->mutex);
        t->worker_read_bytes = (rd > 0) ? rd : 0;
        t->prefetch_io_time = t1 - t0;
        t->worker_has_work = false;
        t->worker_done = true;
        pthread_cond_signal(&t->done_cond);
    }
    pthread_mutex_unlock(&t->mutex);
    return NULL;
}
#endif

/* -------------------------------------------------------------------------
 * Execution plan & memory check
 * ------------------------------------------------------------------------- */

uint64_t fh3_plan_bytes(const Fh3Plan *p) {
    if (!p) return 0;
    if (p->total_bytes > 0) return p->total_bytes;
    if (p->budget_bytes > 0) return p->budget_bytes;
    return p->pinned_bytes + p->ring_bytes + p->activation_bytes + p->table_bytes;
}

int fh3_plan_check(const Fh3Plan *p) {
    if (!p) return FH3_ERR;

    uint64_t req = fh3_plan_bytes(p);

    if (p->min_bytes > 0 && req < p->min_bytes) {
        uint64_t shortfall = p->min_bytes - req;
        fprintf(stderr,
                "fh3: error: requested memory (%llu bytes / %.2f GB) is below minimum requirement (%llu bytes / %.2f GB). Shortfall: %llu bytes (%.2f GB)\n",
                (unsigned long long)req, (double)req / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)p->min_bytes, (double)p->min_bytes / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)shortfall, (double)shortfall / (1024.0 * 1024.0 * 1024.0));
        return FH3_E_NOMEM;
    }

    uint64_t avail = fh3_mem_available();
    uint64_t limit = (uint64_t)((double)avail * 0.95);
    if (limit > 0 && req > limit) {
        uint64_t shortfall = req - limit;
        fprintf(stderr,
                "fh3: error: requested memory (%llu bytes / %.2f GB) exceeds 95%% of available RAM (%llu bytes / %.2f GB). Shortfall: %llu bytes (%.2f GB)\n",
                (unsigned long long)req, (double)req / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)limit, (double)limit / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)shortfall, (double)shortfall / (1024.0 * 1024.0 * 1024.0));
        return FH3_E_NOMEM;
    }

    return FH3_OK;
}

/* -------------------------------------------------------------------------
 * Trunk streamer open, close, prefetch, acquire, stats
 * ------------------------------------------------------------------------- */

int fh3_trunk_open(const char *path, uint64_t budget_bytes, Fh3Trunk *t) {
    if (!path || !t) return FH3_ERR;
    memset(t, 0, sizeof(*t));

    /* 1. Open pack metadata */
    int rc = fh3_pack_open(path, &t->pack);
    if (rc != FH3_OK) return rc;

    if (t->pack.num_layers <= 0) {
        fh3_pack_close(&t->pack);
        return FH3_E_INVALID;
    }

    t->num_layers = t->pack.num_layers;
    t->budget_bytes = budget_bytes;

    /* 2. Open direct I/O handle */
    rc = fh3_open_direct(path, &t->direct_file);
    if (rc != FH3_OK) {
        rc = fh3_open_file(path, &t->direct_file);
        if (rc != FH3_OK) {
            fh3_pack_close(&t->pack);
            return rc;
        }
    }

    /* 3. Fixed-point iteration to balance pinned prefix count vs slot size */
    size_t max_layer_size = 0;
    for (int i = 0; i < t->num_layers; i++) {
        if (t->pack.layers[i].size_bytes > max_layer_size) {
            max_layer_size = (size_t)t->pack.layers[i].size_bytes;
        }
    }
    max_layer_size = (max_layer_size + 4095ULL) & ~4095ULL;

    size_t slot = max_layer_size;
    int npin = 0;

    for (int iter = 0; iter < 1000; iter++) {
        uint64_t avail = (budget_bytes > 2ULL * slot) ? (budget_bytes - 2ULL * slot) : 0;
        int n = 0;
        uint64_t sum = 0;
        for (int i = 0; i < t->num_layers; i++) {
            if (sum + t->pack.layers[i].size_bytes <= avail) {
                sum += t->pack.layers[i].size_bytes;
                n++;
            } else {
                break;
            }
        }
        size_t need = 0;
        for (int i = n; i < t->num_layers; i++) {
            if (t->pack.layers[i].size_bytes > need) {
                need = (size_t)t->pack.layers[i].size_bytes;
            }
        }
        size_t aligned_need = (need + 4095ULL) & ~4095ULL;
        if (n == npin && aligned_need == slot) {
            break;
        }
        npin = n;
        slot = aligned_need;
    }

    t->npin = npin;
    t->slot_size = (slot > 0) ? slot : max_layer_size;

    /* 4. Allocate memory for npin pinned layers and read them once upfront */
    if (t->npin > 0) {
        t->pinned_layers = (uint8_t **)calloc((size_t)t->npin, sizeof(uint8_t *));
        if (!t->pinned_layers) {
            fh3_trunk_close(t);
            return FH3_E_NOMEM;
        }
        for (int L = 0; L < t->npin; L++) {
            size_t l_sz = (size_t)t->pack.layers[L].size_bytes;
            t->pinned_layers[L] = (uint8_t *)fh3_alloc_aligned(l_sz);
            if (!t->pinned_layers[L]) {
                fh3_trunk_close(t);
                return FH3_E_NOMEM;
            }
            int64_t rd = fh3_pread(&t->direct_file, t->pinned_layers[L], l_sz, t->pack.layers[L].file_offset);
            if (rd != (int64_t)l_sz) {
                fh3_trunk_close(t);
                return FH3_E_IO;
            }
        }
    }

    /* 5. Allocate 2 4096-aligned ring slots of size slot_size */
    t->ring_buf[0] = (uint8_t *)fh3_alloc_aligned(t->slot_size);
    t->ring_buf[1] = (uint8_t *)fh3_alloc_aligned(t->slot_size);
    if (!t->ring_buf[0] || !t->ring_buf[1]) {
        fh3_trunk_close(t);
        return FH3_E_NOMEM;
    }
    t->ring_layer[0] = -1;
    t->ring_layer[1] = -1;
    t->active_slot = -1;
    t->prefetch_layer = -1;
    t->prefetch_slot = -1;
    t->prefetch_in_flight = false;

    /* 6. Initialize async worker thread */
#if defined(_WIN32)
    InitializeCriticalSection(&t->cs);
    t->start_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    t->done_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    t->worker_stop = false;
    t->thread = CreateThread(NULL, 0, fh3_trunk_worker_win32, t, 0, NULL);
    t->thread_created = (t->thread != NULL);
#else
    pthread_mutex_init(&t->mutex, NULL);
    pthread_cond_init(&t->start_cond, NULL);
    pthread_cond_init(&t->done_cond, NULL);
    t->worker_stop = false;
    t->worker_has_work = false;
    t->worker_done = false;
    if (pthread_create(&t->thread, NULL, fh3_trunk_worker_posix, t) == 0) {
        t->thread_created = true;
    }
#endif

    return FH3_OK;
}

void fh3_trunk_close(Fh3Trunk *t) {
    if (!t) return;

    /* Stop worker thread */
    if (t->thread_created) {
        if (t->prefetch_in_flight) {
#if defined(_WIN32)
            WaitForSingleObject(t->done_event, INFINITE);
#else
            pthread_mutex_lock(&t->mutex);
            while (!t->worker_done) {
                pthread_cond_wait(&t->done_cond, &t->mutex);
            }
            pthread_mutex_unlock(&t->mutex);
#endif
            t->prefetch_in_flight = false;
        }

        t->worker_stop = true;
#if defined(_WIN32)
        SetEvent(t->start_event);
        WaitForSingleObject(t->thread, INFINITE);
        CloseHandle(t->thread);
        CloseHandle(t->start_event);
        CloseHandle(t->done_event);
        DeleteCriticalSection(&t->cs);
#else
        pthread_mutex_lock(&t->mutex);
        pthread_cond_signal(&t->start_cond);
        pthread_mutex_unlock(&t->mutex);
        pthread_join(t->thread, NULL);
        pthread_mutex_destroy(&t->mutex);
        pthread_cond_destroy(&t->start_cond);
        pthread_cond_destroy(&t->done_cond);
#endif
        t->thread_created = false;
    }

    /* Free pinned layers */
    if (t->pinned_layers) {
        for (int i = 0; i < t->npin; i++) {
            if (t->pinned_layers[i]) {
                fh3_free_aligned(t->pinned_layers[i]);
            }
        }
        free(t->pinned_layers);
        t->pinned_layers = NULL;
    }
    t->npin = 0;

    /* Free ring buffers */
    if (t->ring_buf[0]) {
        fh3_free_aligned(t->ring_buf[0]);
        t->ring_buf[0] = NULL;
    }
    if (t->ring_buf[1]) {
        fh3_free_aligned(t->ring_buf[1]);
        t->ring_buf[1] = NULL;
    }

    /* Close direct file */
    fh3_close_file(&t->direct_file);

    /* Close pack */
    fh3_pack_close(&t->pack);
}

void fh3_trunk_prefetch(Fh3Trunk *t, int L) {
    if (!t || L < 0 || L >= t->num_layers) return;
    if (L < t->npin) return; /* Already pinned */

    /* If layer is already in active slot, nothing to prefetch */
    if (t->active_slot >= 0 && t->ring_layer[t->active_slot] == L) {
        return;
    }
    int other_slot = (t->active_slot == 0) ? 1 : 0;
    if (t->active_slot >= 0 && t->ring_layer[other_slot] == L && !t->prefetch_in_flight) {
        return;
    }

    /* If this layer is already being prefetched, do nothing */
    if (t->prefetch_in_flight && t->prefetch_layer == L) {
        return;
    }

    /* If an old prefetch is in flight, wait for it to complete first */
    if (t->prefetch_in_flight) {
#if defined(_WIN32)
        WaitForSingleObject(t->done_event, INFINITE);
#else
        pthread_mutex_lock(&t->mutex);
        while (!t->worker_done) {
            pthread_cond_wait(&t->done_cond, &t->mutex);
        }
        pthread_mutex_unlock(&t->mutex);
#endif
        t->ring_layer[t->prefetch_slot] = t->prefetch_layer;
        t->prefetch_in_flight = false;
    }

    /* Choose idle slot */
    int idle_slot = (t->active_slot == 0) ? 1 : 0;

    /* Start prefetch into idle_slot */
    t->prefetch_layer = L;
    t->prefetch_slot = idle_slot;
    t->prefetch_in_flight = true;
    t->prefetch_start_time = fh3_trunk_time_sec();

#if defined(_WIN32)
    t->worker_req_layer = L;
    t->worker_req_slot = idle_slot;
    SetEvent(t->start_event);
#else
    pthread_mutex_lock(&t->mutex);
    t->worker_req_layer = L;
    t->worker_req_slot = idle_slot;
    t->worker_has_work = true;
    t->worker_done = false;
    pthread_cond_signal(&t->start_cond);
    pthread_mutex_unlock(&t->mutex);
#endif
}

const uint8_t *fh3_trunk_acquire(Fh3Trunk *t, int L) {
    if (!t || L < 0 || L >= t->num_layers) return NULL;

    /* 1. Pinned layer hit */
    if (L < t->npin) {
        t->stats.hits++;
        return t->pinned_layers[L];
    }

    /* 2. Streaming layer */
    /* Check if already resident in active slot */
    if (t->active_slot >= 0 && t->ring_layer[t->active_slot] == L && !t->prefetch_in_flight) {
        t->stats.hits++;
        return t->ring_buf[t->active_slot];
    }

    /* If this layer is currently in flight */
    if (t->prefetch_in_flight && t->prefetch_layer == L) {
        double w0 = fh3_trunk_time_sec();
#if defined(_WIN32)
        WaitForSingleObject(t->done_event, INFINITE);
#else
        pthread_mutex_lock(&t->mutex);
        while (!t->worker_done) {
            pthread_cond_wait(&t->done_cond, &t->mutex);
        }
        pthread_mutex_unlock(&t->mutex);
#endif
        double w1 = fh3_trunk_time_sec();
        double wait_dur = w1 - w0;

        t->prefetch_in_flight = false;
        t->ring_layer[t->prefetch_slot] = L;
        t->active_slot = t->prefetch_slot;

        t->stats.misses++;
        t->stats.bytes_read += t->pack.layers[L].size_bytes;
        t->stats.read_s += t->prefetch_io_time;
        t->stats.wait_s += wait_dur;

        return t->ring_buf[t->active_slot];
    }

    /* If layer was not prefetched, trigger prefetch and wait immediately */
    fh3_trunk_prefetch(t, L);
    double w0 = fh3_trunk_time_sec();
#if defined(_WIN32)
    WaitForSingleObject(t->done_event, INFINITE);
#else
    pthread_mutex_lock(&t->mutex);
    while (!t->worker_done) {
        pthread_cond_wait(&t->done_cond, &t->mutex);
    }
    pthread_mutex_unlock(&t->mutex);
#endif
    double w1 = fh3_trunk_time_sec();
    double wait_dur = w1 - w0;

    t->prefetch_in_flight = false;
    t->ring_layer[t->prefetch_slot] = L;
    t->active_slot = t->prefetch_slot;

    t->stats.misses++;
    t->stats.bytes_read += t->pack.layers[L].size_bytes;
    t->stats.read_s += t->prefetch_io_time;
    t->stats.wait_s += wait_dur;

    return t->ring_buf[t->active_slot];
}

Fh3TrunkStats fh3_trunk_stats(const Fh3Trunk *t) {
    if (!t) {
        Fh3TrunkStats empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return t->stats;
}

Fh3LayerView fh3_trunk_layer(const Fh3Trunk *t, int L, const uint8_t *base) {
    if (!t) {
        Fh3LayerView empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return fh3_pack_layer(&t->pack, L, base);
}

const Fh3Pack *fh3_trunk_pack(const Fh3Trunk *t) {
    return t ? &t->pack : NULL;
}
