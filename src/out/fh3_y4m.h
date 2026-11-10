#ifndef FH3_Y4M_H
#define FH3_Y4M_H

#include <stdint.h>
#include <stddef.h>
#include "fh3/fh3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fh3Y4m Fh3Y4m;

/**
 * Open/create a Y4M (YUV4MPEG2) video output file with C420jpeg format.
 *
 * @param path Destination file path.
 * @param width Video frame width in pixels (must be > 0 and even).
 * @param height Video frame height in pixels (must be > 0 and even).
 * @param fps_num Frames per second numerator (e.g. 24, 25, 30).
 * @param fps_den Frames per second denominator (e.g. 1).
 * @param out Pointer to store opened Fh3Y4m handle.
 * @return FH3_OK (0) on success, or negative FH3 error code on failure.
 */
int fh3_y4m_open(const char *path, int width, int height, int fps_num, int fps_den, Fh3Y4m **out);

/**
 * Write a single RGB video frame to the Y4M stream.
 * Converts float RGB in [0, 1] to YUV420 with 2x2 box average chroma subsampling.
 *
 * @param y Fh3Y4m handle.
 * @param rgb Planar RGB buffer of shape [3, height, width] with values in [0, 1].
 *            Channel 0 (R): rgb[0 .. width*height - 1]
 *            Channel 1 (G): rgb[width*height .. 2*width*height - 1]
 *            Channel 2 (B): rgb[2*width*height .. 3*width*height - 1]
 * @return FH3_OK (0) on success, or negative FH3 error code on failure.
 */
int fh3_y4m_write_rgb(Fh3Y4m *y, const float *rgb);

/**
 * Close Y4M file and release associated memory.
 *
 * @param y Fh3Y4m handle.
 */
void fh3_y4m_close(Fh3Y4m *y);

/**
 * Create a Fh3FrameSink adapter for an open Fh3Y4m file.
 *
 * @param y Fh3Y4m handle.
 * @return Initialized Fh3FrameSink struct.
 */
Fh3FrameSink fh3_y4m_frame_sink(Fh3Y4m *y);

#ifdef __cplusplus
}
#endif

#endif /* FH3_Y4M_H */
