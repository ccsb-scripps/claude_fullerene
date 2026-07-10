# How many samples the search evaluates

Fitting is done over a **physically-sized H window** (see `prep_mesh.py` /
`RSPIs_for_meshes/README.md`): the mesh surface area sets `C_est`, and the search
scans `H_est ± 4`. Sampling within that window is layered.

## 1. The window
`H_est ± 4` → **9 hexamer counts** (≈ C_est ± 8 carbons).

## 2. Cheap scan (the bulk of the work)
`K = 6` random multistarts per H → **54 candidate tilings** from Instant Meshes,
each scored cheaply (valence-defect count + pentamer-placement error `pent_err`).
Any tiling within a few defects of a true fullerene is flip-healed
(`heal.py`) to defect-free. For most meshes a defect-free tiling appears in this
scan, so **~54 is the typical number evaluated**.

Observed (area-windowed batch): mesh4a, cage_3, cage_4, cage_6, cage_7 all
completed in the initial 54 draws (25–36 s), yielding 13–44 defect-free candidates.

## 3. Escalation (only if needed)
If the scan finds no defect-free tiling — or, in `areafit.py`, if it has not yet
found an IPR (no-adjacent-pentamer) candidate — it keeps sweeping the same 9-H grid
with fresh random seeds, up to a cap (**150** in `full_search.py`, **200** in
`areafit.py`) with a ~7-minute wall-clock limit. Scar-prone shapes (near-spheres,
sharp cones) reach this stage; e.g. cage_5 needed a little escalation (finished ~55 s).

## 4. Expensive stage (few)
Only a shortlist gets the full **dual → force-field relax → planarize → 7-DOF fit**:
`full_search` relaxes its top ~16 to score them and locks in **1** representative;
`areafit` fully finalizes just the **best fit plus the best IPR fit** (1–2 cages).

## Summary
- **~54 tilings** scored per mesh (typical),
- **up to ~150–200** draws if the shape is hard,
- **only 1–2** cages fully built.

The two knobs are the multistart count `K` (6) and the half-window width (4) — raise
either for denser coverage of a size/isomer plateau at the cost of runtime. For
comparison, the older generic grid `H ∈ [220,250]` step 2 was 16 H × 6 = 96 tilings
per scan; the physical window both narrows and speeds this up.
