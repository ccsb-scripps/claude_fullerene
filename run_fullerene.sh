#!/bin/bash
# Wrapper: run the Fullerene program on an .inp file, emit its Fullerene-3D.xyz.
# Usage: run_fullerene.sh <input.inp> <output.xyz>
FDIR="/Users/olson/Dev/fullerenes/fullerene"
source /Users/olson/miniforge3/etc/profile.d/conda.sh
conda activate fort
INP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT="$2"; [ -z "$OUT" ] && OUT="$(pwd)/out.xyz"
case "$OUT" in /*) : ;; *) OUT="$(pwd)/$OUT" ;; esac
cd "$FDIR"
rm -f Fullerene-3D.xyz
( ulimit -s 65500; ./fullerene < "$INP" > /tmp/fullerene_run.log 2>&1 )
if [ -f Fullerene-3D.xyz ]; then
  cp Fullerene-3D.xyz "$OUT"
  echo "OK -> $OUT ($(head -1 Fullerene-3D.xyz | tr -d ' ') atoms)"
else
  echo "FAILED (see /tmp/fullerene_run.log)"; tail -5 /tmp/fullerene_run.log; exit 1
fi
