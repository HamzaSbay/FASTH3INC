#ifndef FH3_VAE_VIDEO_H
#define FH3_VAE_VIDEO_H

#include <stdint.h>
#include <stddef.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fh3VaeV Fh3VaeV;

/**
 * Open Video ViT VAE decoder from packed binary file.
 *
 * @param path Path to packed binary file (e.g. packed/vae_video_dec.bin).
 * @param out Pointer to store opened decoder handle.
 * @return FH3_OK (0) on success, or negative FH3 error code on failure.
 */
int fh3_vae_video_open(const char *path, Fh3VaeV **out);

/**
 * Close Video ViT VAE decoder and release resources.
 *
 * @param m Decoder handle.
 */
void fh3_vae_video_close(Fh3VaeV *m);

/**
 * Decode video latents into RGB pixel frames streamed to a sink.
 *
 * @param m Decoder handle.
 * @param lat Input latents buffer of shape [latent_channels (24), F, h, w].
 * @param F Number of latent frames.
 * @param h Latent height.
 * @param w Latent width.
 * @param sink Frame sink callback to receive each finished pixel frame.
 * @return FH3_OK (0) on success, or negative FH3 error code on failure.
 */
int fh3_vae_video_decode(const Fh3VaeV *m, const float *lat, int F, int h, int w, Fh3FrameSink *sink);

/**
 * Get spatial patch size of Video VAE (e.g. 16).
 */
int fh3_vae_video_patch_size(const Fh3VaeV *m);

/**
 * Get temporal patch size of Video VAE (e.g. 4).
 */
int fh3_vae_video_patch_size_t(const Fh3VaeV *m);

#ifdef __cplusplus
}
#endif

#endif /* FH3_VAE_VIDEO_H */
