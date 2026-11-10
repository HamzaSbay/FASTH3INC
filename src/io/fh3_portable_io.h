#ifndef FH3_PORTABLE_IO_H
#define FH3_PORTABLE_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>

struct Fh3File {
    HANDLE handle;
    uint64_t size;
    bool is_direct;
};

static inline int fh3_open_direct(const char *path, Fh3File *f) {
    if (!path || !f) return FH3_ERR;
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                        OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
        if (h == INVALID_HANDLE_VALUE) return FH3_E_IO;
        f->is_direct = false;
    } else {
        f->is_direct = true;
    }

    LARGE_INTEGER li;
    if (!GetFileSizeEx(h, &li)) {
        CloseHandle(h);
        return FH3_E_IO;
    }
    f->handle = h;
    f->size = (uint64_t)li.QuadPart;
    return FH3_OK;
}

static inline int fh3_open_file(const char *path, Fh3File *f) {
    if (!path || !f) return FH3_ERR;
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    if (h == INVALID_HANDLE_VALUE) return FH3_E_IO;

    LARGE_INTEGER li;
    if (!GetFileSizeEx(h, &li)) {
        CloseHandle(h);
        return FH3_E_IO;
    }
    f->handle = h;
    f->size = (uint64_t)li.QuadPart;
    f->is_direct = false;
    return FH3_OK;
}

static inline int fh3_open(const char *path, Fh3File *f) {
    return fh3_open_file(path, f);
}

static inline void fh3_close_file(Fh3File *f) {
    if (f && f->handle != NULL && f->handle != INVALID_HANDLE_VALUE) {
        CloseHandle(f->handle);
        f->handle = INVALID_HANDLE_VALUE;
    }
}

static inline void *fh3_alloc_aligned(size_t n) {
    if (n == 0) return NULL;
    return _aligned_malloc(n, 4096);
}

static inline void fh3_free_aligned(void *p) {
    if (p) _aligned_free(p);
}

static inline uint64_t fh3_mem_available(void) {
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return (uint64_t)st.ullAvailPhys;
    }
    return 0;
}

static inline int64_t fh3_pread(Fh3File *f, void *buf, size_t n, uint64_t off) {
    if (!f || f->handle == INVALID_HANDLE_VALUE || !buf) return FH3_ERR;
    if (n == 0) return 0;
    if (off >= f->size) return 0;

    /* If request exceeds remaining file, clamp to remaining bytes */
    if (off + (uint64_t)n > f->size) {
        n = (size_t)(f->size - off);
    }

    if (f->is_direct) {
        /* Direct I/O requires 4096 sector alignment for off, n, and buffer address */
        if (((uintptr_t)buf % 4096 != 0) || (off % 4096 != 0) || (n % 4096 != 0)) {
            uint64_t aligned_off = off & ~4095ULL;
            uint64_t end_off = off + n;
            uint64_t aligned_end = (end_off + 4095ULL) & ~4095ULL;
            size_t aligned_n = (size_t)(aligned_end - aligned_off);

            void *bounce = fh3_alloc_aligned(aligned_n);
            if (!bounce) return FH3_E_NOMEM;

            OVERLAPPED ov_b;
            memset(&ov_b, 0, sizeof(ov_b));
            ov_b.Offset = (DWORD)(aligned_off & 0xFFFFFFFF);
            ov_b.OffsetHigh = (DWORD)((aligned_off >> 32) & 0xFFFFFFFF);

            DWORD b_read = 0;
            BOOL ok = ReadFile(f->handle, bounce, (DWORD)aligned_n, &b_read, &ov_b);
            if (!ok) {
                DWORD err = GetLastError();
                if (err == ERROR_IO_PENDING) {
                    if (!GetOverlappedResult(f->handle, &ov_b, &b_read, TRUE)) {
                        DWORD sub_err = GetLastError();
                        if (sub_err != ERROR_HANDLE_EOF) {
                            fh3_free_aligned(bounce);
                            return FH3_E_IO;
                        }
                    }
                } else if (err == ERROR_HANDLE_EOF) {
                    b_read = 0;
                } else {
                    fh3_free_aligned(bounce);
                    return FH3_E_IO;
                }
            }

            size_t off_in_bounce = (size_t)(off - aligned_off);
            size_t available = (b_read > off_in_bounce) ? (b_read - off_in_bounce) : 0;
            size_t copy_n = (n < available) ? n : available;
            if (copy_n > 0) {
                memcpy(buf, (const uint8_t *)bounce + off_in_bounce, copy_n);
            }
            fh3_free_aligned(bounce);
            return (int64_t)copy_n;
        }
    }

    /* Direct aligned or regular buffered read */
    size_t total_read = 0;
    while (total_read < n) {
        size_t to_read = n - total_read;
        DWORD chunk = (to_read > 0x40000000) ? 0x40000000 : (DWORD)to_read;
        uint64_t cur_off = off + total_read;

        OVERLAPPED ov;
        memset(&ov, 0, sizeof(ov));
        ov.Offset = (DWORD)(cur_off & 0xFFFFFFFF);
        ov.OffsetHigh = (DWORD)((cur_off >> 32) & 0xFFFFFFFF);

        DWORD chunk_read = 0;
        BOOL ok = ReadFile(f->handle, (uint8_t *)buf + total_read, chunk, &chunk_read, &ov);
        if (!ok) {
            DWORD err = GetLastError();
            if (err == ERROR_IO_PENDING) {
                if (!GetOverlappedResult(f->handle, &ov, &chunk_read, TRUE)) {
                    DWORD sub_err = GetLastError();
                    if (sub_err == ERROR_HANDLE_EOF) {
                        total_read += chunk_read;
                        break;
                    }
                    return FH3_E_IO;
                }
            } else if (err == ERROR_HANDLE_EOF) {
                break;
            } else {
                return FH3_E_IO;
            }
        }
        if (chunk_read == 0) break;
        total_read += chunk_read;
    }

    return (int64_t)total_read;
}

#else /* POSIX */

#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

struct Fh3File {
    int fd;
    uint64_t size;
    bool is_direct;
};

static inline int fh3_open_direct(const char *path, Fh3File *f) {
    if (!path || !f) return FH3_ERR;
    int flags = O_RDONLY;
#ifdef O_DIRECT
    flags |= O_DIRECT;
#endif
    int fd = open(path, flags);
    bool direct = true;
#ifdef O_DIRECT
    if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
        fd = open(path, O_RDONLY);
        direct = false;
    }
#endif
    if (fd < 0) return FH3_E_IO;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return FH3_E_IO;
    }
    f->fd = fd;
    f->size = (uint64_t)st.st_size;
    f->is_direct = direct;
    return FH3_OK;
}

static inline int fh3_open_file(const char *path, Fh3File *f) {
    if (!path || !f) return FH3_ERR;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return FH3_E_IO;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return FH3_E_IO;
    }
    f->fd = fd;
    f->size = (uint64_t)st.st_size;
    f->is_direct = false;
    return FH3_OK;
}

static inline int fh3_open(const char *path, Fh3File *f) {
    return fh3_open_file(path, f);
}

static inline void fh3_close_file(Fh3File *f) {
    if (f && f->fd >= 0) {
        close(f->fd);
        f->fd = -1;
    }
}

static inline void *fh3_alloc_aligned(size_t n) {
    if (n == 0) return NULL;
    void *ptr = NULL;
    if (posix_memalign(&ptr, 4096, n) != 0) return NULL;
    return ptr;
}

static inline void fh3_free_aligned(void *p) {
    free(p);
}

static inline uint64_t fh3_mem_available(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %llu kB", (unsigned long long *)&kb) == 1) {
            fclose(fp);
            return kb * 1024ULL;
        }
    }
    fclose(fp);
    return 0;
}

static inline int64_t fh3_pread(Fh3File *f, void *buf, size_t n, uint64_t off) {
    if (!f || f->fd < 0 || !buf) return FH3_ERR;
    size_t total = 0;
    uint8_t *p = (uint8_t *)buf;
    while (total < n) {
        ssize_t ret = pread(f->fd, p + total, n - total, (off_t)(off + total));
        if (ret < 0) {
            if (errno == EINTR) continue;
            return FH3_E_IO;
        }
        if (ret == 0) break;
        total += (size_t)ret;
    }
    return (int64_t)total;
}

#endif /* _WIN32 */

#ifdef __cplusplus
}
#endif

#endif /* FH3_PORTABLE_IO_H */
