# RSPIs for meshes

Canonical **Ring Spiral Pentagon Indices (RSPI)** for the best-fitting true
fullerene of each input mesh, produced by the pipeline in the repository root
(`run_all.sh`) and validated with **Fullerene v4.5** (Schwerdtfeger, Wirz, Avery,
*J. Comput. Chem.* 34, 2013).

An RSPI is a compact, canonical description of a fullerene's topology: unwind the
faces along a spiral and list the 1-indexed positions of the 12 pentagons. The 12
indices fully determine the cage's connectivity. All RSPIs below are the
**canonical** (lexicographically minimal) spiral, from Fullerene v4.5 with
`ispsearch=2`; feeding one back via `&Coord ICart=2 rspi=...` rebuilds the exact
isomer (see `fit_built_cage.py` in the repo root).

| mesh | shape | fullerene | PG | IPR | fit | pent_err | RSPI (canonical) |
|------|-------|-----------|----|-----|-----|----------|------|
| mesh4a | spindle | C488 (12 + 234) | C1 | yes | 3.7% | 0.49 | `1 19 23 54 56 148 152 185 215 232 240 245` |
| mesh5 | elongated spindle | C468 (12 + 224) | C1 | yes | 4.1% | 0.40 | `1 25 30 45 64 114 178 220 224 228 230 232` |
| cage_3 | moderately elongated | C492 (12 + 236) | C1 | yes | 4.1% | 0.45 | `1 30 96 104 146 159 161 168 205 211 232 244` |
| cage_4 | moderately elongated | C478 (12 + 229) | C1 | yes | 4.2% | 0.54 | `1 51 53 60 66 71 78 208 219 228 233 239` |
| cage_5 | very elongated cone | C502 (12 + 241) | C1 | **no (Np=2)** | 4.0% | 0.73 | `1 2 8 12 15 194 207 212 218 221 223 244` |
| cage_6 | near-spherical | C474 (12 + 227) | C1 | yes | 3.3% | 0.61 | `1 7 45 94 108 128 151 159 170 190 217 225` |
| cage_7 | near-spherical | C498 (12 + 239) | C1 | yes | 3.5% | ~1.22 | `1 45 49 80 116 141 152 160 166 186 223 236` |

All seven are true fullerenes (exactly 12 pentagons, defect-free). Six are IPR
(isolated pentagons); **cage_5** is not (2 fused pentagon pairs) because its sharp
conical cap clusters pentamers -- physically reasonable.

Two trends are visible:
- **pent_err** (how tightly the pentamers sit on the curvature bowls) tracks how
  strongly the shape localizes curvature: sharp spindles/cones (mesh5, cage_3)
  pin the disclinations best (~0.4-0.5); near-spheres (cage_7) are loosest (~1.2)
  because a sphere barely pins its disclinations.
- **defect-free difficulty**: elongated-but-organic (cage_5) and near-spherical
  (cage_7) shapes are scar-prone -- a clean 12-pentagon tiling is rare and needs
  many multistart draws (cage_5 ~1/480), whereas clean spindles tile easily.

Coarse inputs (< ~1000 verts) are auto-Loop-subdivided before fitting. See the
per-mesh `.txt` files for full provenance.
