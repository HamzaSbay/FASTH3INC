#ifndef FH3_WAV_H
#define FH3_WAV_H

#include <stdint.h>
#include <stddef.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Write float32 planar audio samples to a standard 16-bit PCM RIFF/WAVE file.
 *
 * @param path File system destination path.
 * @param pcm Planar float32 audio buffer of shape [ch, n].
 *            Channel 0: pcm[0 .. n - 1]
 *            Channel 1: pcm[n .. 2 * n - 1]
 * @param ch Number of channels (e.g. 1 for mono, 2 for stereo).
 * @param n Number of sample frames per channel.
 * @param rate Sampling rate in Hz (e.g. 32000).
 * @return FH3_OK (0) on success, or negative FH3 error code on failure.
 */
int fh3_wav_write(const char *path, const float *pcm, int ch, int64_t n, int rate);

#ifdef __cplusplus
}
#endif

#endif /* FH3_WAV_H */
