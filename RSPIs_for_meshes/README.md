# RSPIs for meshes

Canonical **Ring Spiral Pentagon Indices (RSPI)** for the best-fitting true
fullerene of each input mesh, produced by the pipeline in the repository root
(`run_all.sh`) and validated with **Fullerene v4.5** (Schwerdtfeger, Wirz, Avery,
*J. Comput. Chem.* 34, 2013).

An RSPI is a compact, canonical description of a fullerene's topology: unwind the
faces along a spiral and list the 1-indexed positions of the 12 pentagons. The 12
indices fully determine the cage's connectivity.

| mesh | shape | fullerene | point group | fit | pent_err | RSPI |
|------|-------|-----------|-------------|-----|----------|------|
| mesh4a | spindle | C488 (12 + 234) | C1 | 3.7% | 0.49 | `1 22 26 59 61 157 161 196 209 229 240 245` |
| mesh5 | elongated spindle | C468 (12 + 224) | C1 | 4.1% | 0.40 | `1 26 31 47 69 129 197 218 222 230 232 234` |
| cage_6 | near-spherical | C474 (12 + 227) | C1 | 3.3% | 0.61 | `1 11 43 94 108 128 151 159 167 196 211 220` |
| cage_7 | near-spherical | C498 (12 + 239) | C1 | 3.5% | ~1.22 | `1 45 49 86 104 136 154 169 175 203 211 234` |

All four are defect-free (exactly 12 isolated pentagons). Note the trend in
`pent_err` (how tightly the pentamers sit on the curvature bowls): sharp spindles
(mesh4a, mesh5) localize the disclinations best (~0.4-0.5), while near-spherical
targets (cage_7) are looser (~1.2) because a sphere barely pins its disclinations.
The two coarse cage inputs were auto-Loop-subdivided before fitting. See the
per-mesh `.txt` files for full provenance.
