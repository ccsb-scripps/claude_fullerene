# Fit cages (physical area-based fits) — inputs for the atomic-capsid build

The exact per-mesh fullerene fits the committed capsid models were built from,
stored so the atomic capsids regenerate **deterministically** without re-running
the stochastic tiling search (`areafit.py`).

Each `<mesh>_best.json` (source-mesh Ångström frame) holds:
`atoms` (fit cage carbons), `bonds`, `pent`/`hex` (the 12 pentagon + hexagon faces),
`peaks` (12 curvature high points / pentamer sites), and `meshV`/`meshF` (the
prepared mesh).

## Which cage each mesh uses

| mesh | cage input for `build_atomic_capsid.py` |
|------|------------------------------------------|
| cage_4, cage_5, cage_6, cage_7, mesh5 | `fit_cages/<mesh>_best.json` (JSON-cage path) |
| cage_3 | relaxed `mesh4a_C440_representative.xyz` **+** `--faces fit_cages/cage_3_best.json` |
| mesh4a | relaxed `mesh4a_C490_representative.xyz` (xyz path) + `mesh4a.stl` — no best.json |

(cage_3 and mesh4a use the *relaxed* uniform cage for clean packing; the other five
use their fit cage directly.)

## Regenerate a capsid

```bash
# JSON-cage meshes (e.g. cage_4):
.venv/bin/python build_atomic_capsid.py fit_cages/cage_4_best.json \
    ~/Downloads/Fullerene/cage_4_process.ply --emit transforms --out-prefix cage_4_C380
# cage_3 (relaxed geometry + authoritative faces):
.venv/bin/python build_atomic_capsid.py mesh4a_C440_representative.xyz \
    ~/Downloads/Fullerene/cage_3_process.ply --faces fit_cages/cage_3_best.json --out-prefix cage_3_C440
```

Source meshes: `~/Downloads/Fullerene/` (`cage_*.ply`, `mesh5.stl`) and `mesh4a.stl`
in this repo.
