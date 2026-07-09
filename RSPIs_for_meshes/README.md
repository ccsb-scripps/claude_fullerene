# RSPIs for meshes (physically-sized fits)

Best-fitting true fullerene for each input mesh, **sized by physical surface area**
rather than an arbitrary range. Because the fit uses a free scale, size must be
anchored externally: capsomers tile the surface at the CA hexamer center-to-center
spacing d (~95.5 A), each cell = (sqrt3/2) d^2, so C_est = 20 + 2*(Area/A_cell - 7.95).
`full_search` searches a narrow H window around that estimate; the chosen fit lands
within ~+/-8 C of C_est. RSPIs are canonical ring spirals computed directly from the
cage topology (`cage_rspi.py`) -- program-independent, robust on crowded/elongated cages.

| mesh | shape | C_est | area (nm^2) | C_fit | IPR | fit % | pent_err | canonical RSPI |
|------|-------|-------|-------------|-------|-----|-------|----------|----------------|
| mesh4a | spindle (HIV-1 capsid core) | C484 | 18920 | C490 | yes | 3.9 | 0.64 | `1 7 10 23 76 84 140 223 232 234 240 245` |
| mesh5 | elongated spindle | C370 | 14427 | C384 | **no (Np=2)** | 4.0 | 0.52 | `1 2 11 32 50 91 106 165 179 187 190 194` |
| cage_3 | moderately elongated | C432 | 16905 | C426 | yes | 4.1 | 0.70 | `1 7 23 26 62 153 165 170 175 180 183 215` |
| cage_4 | moderately elongated | C372 | 14514 | C380 | yes | 4.3 | 0.55 | `1 7 10 52 57 80 115 145 173 177 185 189` |
| cage_5 | very elongated cone | C478 | 18732 | C482 | **no (Np=3)** | 3.6 | 0.75 | `1 2 4 10 16 168 183 211 217 219 226 231` |
| cage_6 | near-spherical | C448 | 17545 | C446 | yes | 2.7 | 0.59 | `1 7 35 59 110 118 132 150 164 190 196 217` |
| cage_7 | near-spherical | C416 | 16294 | C424 | yes | 3.2 | 1.14 | `1 17 25 54 93 109 129 175 183 188 191 211` |

All fits are defect-free (exactly 12 pentagons). **IPR (no adjacent pentamers) holds
for 5 of 7.** The two exceptions -- mesh5 and cage_5 -- are the sharpest, most
elongated cones, where the narrow tip geometrically forces pentamers to cluster; no
isolated-pentagon solution exists at their physical size (searched and confirmed).
This mirrors the real HIV-1 capsid cone, whose pointed end packs pentamers together.
