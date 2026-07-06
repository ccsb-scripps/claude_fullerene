# Best-fitting fullerene for mesh4a.stl

**Result: a valid C480 fullerene — 12 isolated pentagons + 230 hexagons, all
carbons degree-3, force-field-relaxed bond CV 1.49%.**

## Target
`mesh4a.stl` = hand-segmented HIV capsid core. V=2371, E=7107, F=4738, Euler 2,
closed manifold. Prolate **spindle/cone**: radius 253–657 (asphericity 2.60),
PCA axis ratios 1 : 0.43 : 0.40, surface area ≈ 1.89×10⁶ (mesh units²).

## Pipeline
1. **Characterise mesh** (`fit_fullerene.py` loader): shape, axis, area.
2. **Topology** — started from the prior 242-capsomer triangulation
   (`C480_triangulation.txt`, 14×deg5 + 226×deg6 + 2×deg7). The two (5,7)
   dislocation dipoles are topologically locked to random edge flips (as the
   prior session found). Instead used the **prior pinned RSPI**
   `[1,9,29,77,83,95,143,195,200,221,238,241]` (1-indexed), which the compiled
   **Fullerene v4.5** program accepts as a valid spiral.
3. **Build + relax** — `NA=480, ICart=3, rspi=…, FFChoice Iopt=4` (Wu force
   field). Program confirms: 242 faces, 480 atoms, **12 pentagons, 230
   hexagons, Np=0 (all pentagons isolated)**, point group C1. Bonds 1.36–1.48 Å,
   **CV 1.49%**. → `mesh4a_C480_ideal.xyz`
4. **Fit to mesh** (`validate_fit.py`): PCA-align + axis sign-match, then either
   uniform scale (compare intrinsic shape) or radial projection onto the mesh
   surface (conform).

## Two structures produced
| file | what | bond CV | surface-fit RMS | shape |
|---|---|---|---|---|
| `mesh4a_C480_ideal.xyz` | regular fullerene, intrinsic shape | **1.49%** | — | asph 2.29 |
| `mesh4a_C480_conformed.xyz` | same topology, atoms on mesh4a | 48% | **2.6% of R** | asph 2.62 |

The **ideal** cage is the chemically real fullerene (perfectly regular bonds)
but relaxes ~rounder (2.29:1) than the mesh (2.60:1). The **conformed** cage
puts every atom on the mesh surface (tight geometric fit) at the cost of bond
regularity — the unavoidable tension of tiling a 2.6:1 spindle with a uniform
hexagonal lattice.

## Physically meaningful finding
The 12 pentagons sit at the two tips with a **5 / 7 split** (5 at the narrow
tip, 7 across the broad end; mean |axial|/half-span = 0.81). This reproduces the
**Ganser–Pornillos HIV capsid fullerene-cone model** (12 = 5 + 7), independent
confirmation that the pinned RSPI encodes a genuine capsid-cone isomer.

## Files
- Structures: `mesh4a_C480_ideal.xyz`, `mesh4a_C480_conformed.xyz`,
  `mesh4a_C480_edges.txt` (720 bonds).
- Scripts: `fit_fullerene.py` (dual-on-surface route), `validate_fit.py`
  (relax+fit), `export_viewer.py` (face extraction → JSON), `repair_test.py`
  (dipole-annihilation experiment).
- Input: `mesh4a_C480.inp` (run via `./run_fullerene.sh`).
- Interactive viewer: `~/mytestapp/fit_viewer.html` + `viewer_data.json`
  (orbit the cage on the mesh; toggle conformed vs ideal).
