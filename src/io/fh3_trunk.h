#ifndef FH3_TRUNK_H
#define FH3_TRUNK_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "fh3/fh3.h"
#include "io/fh3_pack.h"
#include "io/fh3_portable_io.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Pinned prefix + 2-slot ring trunk streamer */
struct Fh3Trunk {
    Fh3Pack pack;                  /* Underlying pack reader with descriptors */
    Fh3File direct_file;           /* Direct / overlapped file handle */
    int num_layers;                /* Total layers in trunk */

    uint64_t budget_bytes;         /* Requested budget */
    int npin;                      /* Pinned layers count */
    size_t slot_size;              /* 4096-aligned ring slot size */

    uint8_t **pinned_layers;       /* [npin] allocated buffers for pinned layers */
    uint8_t *ring_buf[2];          /* 2 ring slots */
    int ring_layer[2];             /* Layer resident in ring slot [0] and [1], or -1 */
    int active_slot;               /* Slot currently active (0 or 1), or -1 */

    /* Prefetch state */
    int prefetch_layer;            /* Layer currently prefetched, or -1 */
    int prefetch_slot;             /* Slot targeted by prefetch */
    bool prefetch_in_flight;       /* Whether background prefetch is running */
    double prefetch_start_time;    /* When prefetch was triggered */
    double prefetch_io_time;       /* Actual I/O duration */

    /* Thread worker synchronization */
#if defined(_WIN32)
    HANDLE thread;
    HANDLE start_event;
    HANDLE done_event;
    CRITICAL_SECTION cs;
#else
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    bool worker_has_work;
    bool worker_done;
#endif
    bool thread_created;
    bool worker_stop;
    int worker_req_layer;
    int worker_req_slot;
    int64_t worker_read_bytes;

    Fh3TrunkStats stats;
};

/* Additional helper functions */
Fh3LayerView fh3_trunk_layer(const Fh3Trunk *t, int L, const uint8_t *base);
const Fh3Pack *fh3_trunk_pack(const Fh3Trunk *t);

#ifdef __cplusplus
}
#endif

#endif /* FH3_TRUNK_H */
