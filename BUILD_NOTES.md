# Fullerene program — build & run on Apple Silicon (arm64, macOS 26)

The stock `fullerene` v4.5 code targets Intel Mac with gfortran. Getting it running
on this arm64 machine required a toolchain and three source/flag changes.

## Toolchain
- System had no Fortran compiler. Homebrew (Intel `/usr/local`) fails to build GCC 16;
  MacPorts is broken (OS platform mismatch).
- Installed **miniforge** at `/Users/olson/miniforge3`, conda env **`fort`** with
  conda-forge `gfortran` (GNU 15.2), `gxx`, `make`.
- C++ (`libgraph/*.cc`) compiles with Apple `clang++`; Fortran with conda `gfortran`.

## Build
Use the patched **`fullerene/Makefile.arm64`** (original `Makefile` left untouched):
```
cd fullerene
source /Users/olson/miniforge3/etc/profile.d/conda.sh && conda activate fort
export SDKROOT=$(xcrun --show-sdk-path)
make -f Makefile.arm64 -j4
```
Changes vs original Makefile:
- `CXX=clang++`, drop `-mcmodel=medium` (x86-only), add
  `-fopenmp -std=legacy -fallow-invalid-boz -fallow-argument-mismatch -ffixed-line-length-132`.
- Link `-lc++` instead of `-lstdc++` (clang uses libc++).

Source edits:
- `source/config.f`: `Nmax` 5000 → **1200** (5000² static/stack arrays overflow arm64
  addressing/stack; 1200 covers up to C1200, plenty for our C≤600 targets).
- `source/schlegel.f` lines 767/774: legacy hex color literals `x'rrggbb'` wrapped as
  `int(z'rrggbb')` (BOZ-as-argument rejected by gfortran 15). Schlegel plotting only —
  not used by the coordinate pipeline.

## Run — IMPORTANT
The Fortran main puts large arrays on the stack, so you MUST raise the stack limit
(64 MB, the macOS max) or it crashes with EXC_BAD_ACCESS at `MAIN__`:
```
( ulimit -s 65500; ./fullerene < input.inp > out.log )
```
Stack usage is fixed by `Nmax`, so 64 MB covers all sizes up to C1200.

Use the wrapper: **`./run_fullerene.sh <input.inp> <output.xyz>`** (handles conda env +
ulimit, copies the emitted `Fullerene-3D.xyz` to your path).

## Input/output format
- Input: Fortran namelist. `&General NA=<carbons> /`,
  `&Coord ICart=3, rspi= <12 pentagon indices> /` (RSPI is 1-indexed),
  `&FFChoice Iopt=4 /` (force-field relaxation → regular equal edges).
- Output: `Fullerene-3D.xyz` (standard XYZ), plus analysis to stdout.

## Validation
`input/fullerene_input_0.inp` (C392, rspi 1 8 10 12 14 120 132 157 181 189 191 198)
reproduces the reference `input/cage_0.xyz`: 392 atoms, all degree-3, bond-length
CV 1.58%, spindle axis ~2.7:1.2:1. Confirmed matching.
