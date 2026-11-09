#ifndef FH3_H
#define FH3_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations of core engine types */
typedef struct Fh3DitCfg Fh3DitCfg;
typedef struct Fh3TeCfg Fh3TeCfg;
typedef struct Fh3W Fh3W;
typedef struct Fh3Seq Fh3Seq;
typedef struct Fh3Sched Fh3Sched;
typedef struct Fh3VsaCfg Fh3VsaCfg;
typedef struct Fh3File Fh3File;
typedef struct Fh3Gguf Fh3Gguf;
typedef struct Fh3Tensor Fh3Tensor;
typedef struct Fh3Trunk Fh3Trunk;
typedef struct Fh3Tok Fh3Tok;
typedef struct Fh3Te Fh3Te;
typedef struct Fh3VaeA Fh3VaeA;
typedef struct Fh3VaeV Fh3VaeV;
typedef struct Fh3FrameSink Fh3FrameSink;

/* Version info */
#define FH3_VERSION_MAJOR 0
#define FH3_VERSION_MINOR 1
#define FH3_VERSION_PATCH 0
#define FH3_VERSION_STRING "0.1.0"

const char *fh3_version(void);

#ifdef __cplusplus
}
#endif

#endif /* FH3_H */
