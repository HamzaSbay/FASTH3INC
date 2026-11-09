#include <stdio.h>
#include <stdlib.h>
#include "fh3/fh3.h"

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    printf("FastH3-in-C Engine (v%s)\n", fh3_version());
    printf("A dependency-free C99 engine for FastH3-8-Step-V2 with NVMe streaming.\n");
    return 0;
}
