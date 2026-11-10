#!/usr/bin/env bash
# FastH3-in-C: 01-tiny Example
# Demonstrates running FastH3 end-to-end on synthetic tiny fixtures:
# Phase 1 (Text Encoder) -> Phase 2 (DiT 8-Step) -> Phase 3 (Decoders -> out.y4m + out.wav)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export FASTH3_SKIP_REMOTE_TESTS=1

FH3_BIN="${ROOT_DIR}/build/fh3"
if [[ ! -f "${FH3_BIN}" && -f "${ROOT_DIR}/build/fh3.exe" ]]; then
    FH3_BIN="${ROOT_DIR}/build/fh3.exe"
fi

if [[ ! -f "${FH3_BIN}" ]]; then
    echo "Binary fh3 not found. Building project..."
    cmake -B build
    cmake --build build
fi

OUT_DIR="${ROOT_DIR}/out/01-tiny"
JSON_REPORT="${OUT_DIR}/report.json"

echo "Running FastH3 end-to-end tiny generation..."
"${FH3_BIN}" run \
    --preset tiny \
    --tiny-fixture \
    --prompt "a red fox running through snow, wind sounds" \
    --seed 42 \
    --out "${OUT_DIR}" \
    --json "${JSON_REPORT}"

echo "Success! Playable video at ${OUT_DIR}/out.y4m and stereo audio at ${OUT_DIR}/out.wav"
