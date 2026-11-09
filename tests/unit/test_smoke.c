#include <assert.h>
#include <stdio.h>
#include <omp.h>
#include "fh3/fh3.h"

int main(void) {
    int max_threads = omp_get_max_threads();
    printf("[test_smoke] omp_get_max_threads() = %d\n", max_threads);
    assert(max_threads >= 1);

    size_t ptr_size = sizeof(void *);
    printf("[test_smoke] sizeof(void*) = %zu bytes\n", ptr_size);
    assert(ptr_size == 8);

    const char *ver = fh3_version();
    printf("[test_smoke] fh3_version() = %s\n", ver);
    assert(ver != NULL);

    printf("[test_smoke] PASS\n");
    return 0;
}
