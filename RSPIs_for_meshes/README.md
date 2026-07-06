# RSPIs for meshes

Canonical **Ring Spiral Pentagon Indices (RSPI)** for the best-fitting true
fullerene of each input mesh, produced by the pipeline in the repository root
(`run_all.sh`) and validated with **Fullerene v4.5** (Schwerdtfeger, Wirz, Avery,
*J. Comput. Chem.* 34, 2013).

An RSPI is a compact, canonical description of a fullerene's topology: unwind the
faces along a spiral and list the 1-indexed positions of the 12 pentagons. The 12
indices fully determine the cage's connectivity.

| mesh | fullerene | point group | fit | pent_err | RSPI |
|------|-----------|-------------|-----|----------|------|
| cage_6 | C474 (12 + 227) | C1 | 3.3% | 0.61 | `1 11 43 94 108 128 151 159 167 196 211 220` |
| cage_7 | C498 (12 + 239) | C1 | 3.5% | ~1.22 | `1 45 49 86 104 136 154 169 175 203 211 234` |

Both are defect-free (exactly 12 isolated pentagons) and derive from
near-spherical inputs that were auto-Loop-subdivided before fitting. See the
per-mesh `.txt` files for full provenance.
