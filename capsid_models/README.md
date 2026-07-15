# Pseudo-atomic HIV-1 CA capsid models (instance transforms)

Each mesh's best-fitting true fullerene (stage 0-5 of the pipeline) decorated with
real CA capsomers -- 3H47 hexamer + 3P05 pentamer (stage 6, `build_atomic_capsid.py`).
All capsomers are rigid copies of the same two canonical structures, so each model
is stored as a compact **`<mesh>_C<N>_capsid_transforms.json`** (~40-50 KB: one
`world = R @ canonical_template + t` operator per capsomer) instead of a ~180 MB
coordinate file.

**Coordinate frame: the original cryo-ET mesh frame (Ångström).** Every model
(transforms, assembly mmCIF, and `<mesh>_surface.obj`) is in the SAME frame as the
source segmentation mesh, so loading your original mesh file together with the
expanded capsid (or the assembly) overlays them directly — no alignment needed.
`<mesh>_surface.obj` is that mesh (a convenience copy).

## Rebuild the full atomic model
```bash
python expand_capsid.py capsid_models/<mesh>_C<N>_capsid_transforms.json \
       --outdir . [--cif] [--backbone]
```
This auto-fetches the RCSB templates, regenerates the canonical capsomers, and
writes the full PDB (and optionally mmCIF / an N,CA,C,O backbone with pentamers
flagged). Needs the repo `.venv` (numpy/scipy).

**Assembly mmCIF (committed).** Each mesh also has a ~1.2 MB
`<mesh>_C<N>_capsid_assembly.cif`: one hexamer + one pentamer reference plus a
`pdbx_struct_assembly` / `pdbx_struct_oper_list` transformation list (one operator
per capsomer). Viewers that support assemblies expand it to the full capsid
(Mol*/PDBe "Assembly 1"; ChimeraX `sym #1 assembly 1`; PyMOL `set assembly, 1`).
Regenerate with `expand_capsid.py ... --assembly`.

## Models

| mesh | C | capsomers (hex + pent) | topology | surface fit % | pent_err | ~atoms |
|------|-----|------------------------|----------|---------------|----------|--------|
| mesh4a | 490 | 247 (235 + 12) | IPR | 3.9 | 0.64 | 2.28 M |
| mesh5  | 364 | 184 (172 + 12) | non-IPR (Np=2) | 4.3 | 0.58 | 1.70 M |
| cage_3 | 440 | 222 (210 + 12) | IPR | 4.4 | 0.70 | 2.05 M |
| cage_4 | 380 | 192 (180 + 12) | IPR | 5.1 | 0.55 | 1.77 M |
| cage_5 | 484 | 244 (232 + 12) | non-IPR (Np=4) | 3.7 | 0.61 | 2.26 M |
| cage_6 | 446 | 225 (213 + 12) | non-IPR (Np=1) | 3.9 | 0.62 | 2.08 M |
| cage_7 | 416 | 210 (198 + 12) | IPR | 4.4 | 1.23 | 1.94 M |

Every model: exactly **12 pentamers**, placed at **95.5 Å** hex-hex spacing,
rigid-body de-clashed to **zero inter-capsomer heavy-atom contact < 2.6 A**
(min contact 2.6-2.9 A -- normal van der Waals). C is each mesh's physical
area-based fit (see `../RSPIs_for_meshes/`); it varies a little run to run because
the tiling search is stochastic. cage_6 and mesh5 also have IPR (no-adjacent-
pentamer) alternatives at nearly the same size; the models here use the best
overall fit (minimum pentamer-placement error). cage_3 and mesh4a are built from
the relaxed (uniform-bond) cage; the rest from the mesh-fit cage.

Templates and the large expanded PDB/CIF are .gitignore'd; only these transforms +
surfaces are versioned.
