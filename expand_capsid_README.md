# expand_capsid.py — rebuild a full atomic capsid from its transforms

`expand_capsid.py` takes one compact **transforms** file and reconstructs the full
pseudo-atomic HIV-1 CA capsid (millions of atoms) by placing copies of the two
canonical capsomer structures at the stored per-capsomer positions. The heavy
coordinate files are *not* stored in the repo (each is 100–180 MB, over GitHub's
limit); you regenerate whichever one you need with this script.

---

## Where to find the input

The input files live in **`capsid_models/`** in this repo — one per mesh:

| transforms file (input) | mesh | fullerene | capsomers | ~atoms (full) |
|--------------------------|--------|-----------|-----------|---------------|
| `mesh4a_C490_capsid_transforms.json` | mesh4a | C490 | 247 (235 hex + 12 pent) | 2.28 M |
| `mesh5_C364_capsid_transforms.json`  | mesh5  | C364 | 184 (172 + 12) | 1.70 M |
| `cage_3_C440_capsid_transforms.json` | cage_3 | C440 | 222 (210 + 12) | 2.05 M |
| `cage_4_C380_capsid_transforms.json` | cage_4 | C380 | 192 (180 + 12) | 1.77 M |
| `cage_5_C484_capsid_transforms.json` | cage_5 | C484 | 244 (232 + 12) | 2.26 M |
| `cage_6_C446_capsid_transforms.json` | cage_6 | C446 | 225 (213 + 12) | 2.08 M |
| `cage_7_C416_capsid_transforms.json` | cage_7 | C416 | 210 (198 + 12) | 1.94 M |

Each is ~40–50 KB. `capsid_models/` also contains the matching
`<mesh>_surface.obj` (see "Companion surface" below).

---

## Requirements

- Run with the repo virtual-env: **`.venv/bin/python`** (numpy + scipy; the system
  python has neither).
- **Internet on first run**: the two capsomer templates are downloaded from RCSB —
  `3H47` (CA hexamer) and `3P05` (CA pentamer) — and cached in
  `capsomer_templates/`. Later runs are offline.

---

## Usage

```bash
cd ~/Dev/fullerenes
.venv/bin/python expand_capsid.py capsid_models/<mesh>_C<N>_capsid_transforms.json \
      [--outdir DIR] [--cif] [--backbone] [--templates DIR]
```

Example:
```bash
.venv/bin/python expand_capsid.py capsid_models/cage_4_C380_capsid_transforms.json --backbone
```

Flags:
- `--outdir DIR` — where to write outputs (default: current directory).
- `--cif` — also write mmCIF.
- `--backbone` — also write an N,CA,C,O backbone PDB (pentamers flagged).
- `--templates DIR` — capsomer-template cache dir (default `capsomer_templates/`).

---

## Inputs, in detail

1. **The transforms JSON** (positional argument). Structure:
   - top-level metadata: `frame` (`"Angstrom; world = R @ canonical_template + t"`),
     `hexamer_template`, `pentamer_template`, `spacing_A` (95.5), `azimuth_deg` (30),
     `cage`, `mesh_surface_obj`, `n_capsomers`.
   - `capsomers`: a list, one entry per capsomer:
     `{"id": "H001"|"P001"…, "type": "hex"|"pent", "R": 3×3 rotation, "t": [x,y,z]}`.
     A capsomer's atoms are `world = R · canonical_template_atom + t` (Ångström).
2. **The capsomer templates** — fetched/cached automatically (see Requirements);
   canonicalized to centroid-at-origin, symmetry axis on +z, NTD facing outward.

---

## Outputs

Written to `--outdir`, named from the input prefix (`<mesh>_C<N>`):

- **`<mesh>_C<N>_capsid_atomic.pdb`** — always written. All heavy atoms (~2 M).
  - **hybrid-36** atom serial numbers (the field overflows past 99999, so serials
    ≥100000 are base-36 encoded — read natively by ChimeraX/PyMOL/VMD).
  - `chainID` = subunit within a capsomer (`A`–`F` hexamer, `A`–`E` pentamer).
  - `segID` (columns 73–76) = the capsomer id (`H001`…`H###`, `P001`…`P012`).
- **`<mesh>_C<N>_capsid_atomic.cif`** — with `--cif`. Same model as mmCIF; every
  subunit gets a unique multi-character chain id (`H001A`, `H001B`, …), avoiding
  the PDB single-character chain limit.
- **`<mesh>_C<N>_capsid_backbone.pdb`** — with `--backbone`. N, CA, C, O only
  (about half the atoms). The 12 pentamers are flagged two ways so they are easy
  to see: `segID` `P001`–`P012` **and** `B-factor = 100` (hexamers `B = 0`) —
  e.g. `color byattribute bfactor` in ChimeraX highlights them.

Sizes: the full PDB/CIF are ~100–180 MB. They are deterministic, so treat them as
regenerable artifacts rather than storing them.

---

## Companion surface

`capsid_models/<mesh>_surface.obj` (name is also in the JSON's `mesh_surface_obj`
field) is the original cryo-ET segmentation surface, transformed into the **same
Ångström frame** as the model. Load it together with the expanded PDB/CIF and the
surface wraps the capsomers directly — no extra alignment.

---

## Regenerating a model from scratch (mesh → transforms)

If you want to rebuild the transforms themselves (e.g. for a new mesh) rather than
just expand an existing one, use `build_atomic_capsid.py` (pipeline stage 6); see
the main `README.txt`. `expand_capsid.py` only turns an existing transforms file
back into coordinates.
