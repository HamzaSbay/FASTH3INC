#ifndef FH3_VAE_AUDIO_H
#define FH3_VAE_AUDIO_H

#include <stdint.h>
#include <stddef.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fh3VaeA Fh3VaeA;

/**
 * Open BigVGAN audio VAE decoder from packed binary file.
 *
 * @param path Path to packed binary file (e.g. packed/vae_audio_dec.bin).
 * @param out Pointer to store opened decoder handle.
 * @return FH3_OK (0) on success, or negative FH3 error code on failure.
 */
int fh3_vae_audio_open(const char *path, Fh3VaeA **out);

/**
 * Close and release resources of audio VAE decoder.
 *
 * @param m Decoder handle to close.
 */
void fh3_vae_audio_close(Fh3VaeA *m);

/**
 * Decode audio latents into 32 kHz stereo PCM waveform.
 *
 * @param m Decoder handle.
 * @param lat Input latents buffer of shape [2, L, latent_channels] (stereo planar).
 * @param L Number of latent time frames.
 * @param pcm Output buffer of shape [2, L * total_upsample] (stereo planar).
 *            Channel 0: pcm[0 .. L * total_upsample - 1]
 *            Channel 1: pcm[L * total_upsample .. 2 * L * total_upsample - 1]
 * @return FH3_OK (0) on success, or negative FH3 error code on failure.
 */
int fh3_vae_audio_decode(const Fh3VaeA *m, const float *lat, int L, float *pcm);

/**
 * Get total upsample ratio of audio decoder (e.g. 800 for FastH3, 4 for tiny fixture).
 *
 * @param m Decoder handle.
 * @return Upsample ratio, or 0 if invalid.
 */
int fh3_vae_audio_upsample_ratio(const Fh3VaeA *m);

#ifdef __cplusplus
}
#endif

#endif /* FH3_VAE_AUDIO_H */
