#!/bin/bash
# audio.sh <duration_seconds> <out_wav>
DUR=${1:-5}
OUT=${2:-/tmp/input.wav}
# 16kHz mono, 16-bit
arecord -f S16_LE -r 16000 -c 1 -d "$DUR" "$OUT"