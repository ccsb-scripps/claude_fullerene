================================================================================
  Template-free fullerene fitting for segmented cryo-ET meshes
  (HIV-1 capsid core -> best-fitting TRUE fullerene cage)
================================================================================

PROBLEM
-------
Given a low-resolution triangle mesh from segmented cryo-ET data of an HIV-1
capsid core, produce the best-fitting TRUE fullerene: a closed cage of exactly
12 pentamers + N hexamers in which every tile is a rigid, near-regular polygon
(equal edges, equal internal angles). The pentamer (5-fold) positions are
dictated by the mesh curvature; the hexamer count H is chosen from a range; and
the absolute tile size is a FREE parameter, because the cryo-ET magnification is
not known a priori. Fitting is therefore a 7-DOF alignment (rotation +
translation + uniform scale) of a self-consistent, intrinsically-shaped cage
onto the mesh -- NOT a warp of a cage onto the surface.

The method is deliberately TEMPLATE-FREE so it generalises to many segmented
meshes: nothing about a specific known fullerene is assumed. Global optimality
is not claimed; the deliverable is the best structure found under a systematic
search, together with the reported plateau of near-ties.


APPROACH (why it works)
-----------------------
1. Topology is fixed by Gauss-Bonnet. For a genus-0 surface the total Gaussian
   curvature is 4*pi. In a trivalent hexagonal tiling every pentagon is a +60deg
   (pi/3) disclination and every hexagon is intrinsically flat, so a closed
   fullerene ALWAYS has exactly 12 pentagons (Euler): 12 * pi/3 = 4*pi. The cage
   size varies (number of hexagons), the pentagon count never does.

2. Shape is set by WHERE the 12 disclinations sit. Because hexagons carry no
   Gaussian curvature, the gross shape of a regular-tiled fullerene is determined
   by the placement of its 12 pentamers; the hexamers merely tile the flat
   regions between them. We therefore place the 12 pentamers at the mesh's 12
   "bowl" points -- local elliptic maxima where both principal curvatures are
   positive -- and let the tiling fill in around them.

3. The tiling itself comes from field-aligned remeshing. A curvature-aligned
   6-way (hexagonal) orientation field over the mesh, extracted at a chosen
   target face count, yields a triangulation whose vertices are the capsomers;
   its dual is the fullerene cage. Valence-5 vertices are pentamers, valence-6
   are hexamers.

4. Selection + regularisation + fit. Because the remesher is stochastic, we run
   a search over hexamer count H and multiple random starts, score each candidate
   against the bowl targets, dualise + relax the best to rigid regular tiles, and
   align to the mesh with a 7-DOF (orientation-resolved) similarity transform.


ALGORITHM & SOURCES
-------------------
Stage 0  Mesh prep (prep_mesh.py [+ subdivide_mesh.py])
    Read binary STL (per-triangle soup -> welded index) or binary PLY (already
    indexed). Verified via Euler characteristic V-E+F = 2 and zero non-manifold
    edges. COARSE inputs (< ~1000 verts) are auto-Loop-subdivided to >= ~4000
    verts: on a coarse mesh each vertex carries a big lump of curvature, so the
    pi/3 charge partition overshoots (imprecise bowl markers) and the surface is
    faceted; Loop subdivision (subdivide_mesh.py) refines + smooths toward the
    limit surface, fixing both. Fine meshes are left untouched.

Stage 1a Per-vertex curvature (principal_curvature.py)
    Discrete Gaussian curvature K = angle deficit / mixed area; mean curvature H
    from the cotangent Laplacian (signed by the outward normal); principal
    curvatures k1,k2 = H +/- sqrt(H^2 - K); "bowl" = min(k1,k2) (>0 iff elliptic).
    Ref: Meyer, Desbrun, Schroeder, Barr, "Discrete Differential-Geometry
    Operators for Triangulated 2-Manifolds", VisMath III (2003).

Stage 1b Bowl-anchored pentamer placement (charge_partition_bowl.py)
    Partition the surface into 12 geodesic basins each accumulating pi/3 of
    Gaussian curvature (topology-safe "charge quantisation"); place each pentamer
    site at its basin's BOWL PEAK (min(k1,k2)-weighted centroid, saddles clamped
    to 0). Deterministic: identical sites every run. This is the scoring/selection
    target for the search.
    Concept ref (disclinations & crystalline order on curved surfaces):
    D. R. Nelson, "Defects and Geometry in Condensed Matter Physics" (2002);
    Bowick & Giomi, Adv. Phys. 58 (2009) 449.

Stage 2  Field-aligned tiling + search (full_search.py + InstantMeshesBatch)
    For each H in [220,250] and several random starts, run the field-aligned
    remesher at face count 2H+20 (rosy=6, posy=6). Cheap metrics on each raw
    tiling: number of valence defects (!=5,6) and pentamer-placement error
    pent_err = mean over the 12 bowl targets of the distance to the nearest
    valence-5 capsomer, in units of hexamer diameter. Shortlist (gate defects,
    then pent_err), then for each shortlisted candidate build the dual cage,
    relax to regular tiles, and 7-DOF fit; composite rank prefers defect-free,
    then low pent_err, then low Hausdorff. Reports the representative and the
    near-tie band.
    Ref: W. Jakob, M. Tarini, D. Panozzo, O. Sorkine-Hornung, "Instant
    Field-Aligned Meshes", ACM TOG 34(6), SIGGRAPH Asia 2015.
    Binary: instant-meshes/InstantMeshesBatch  (headless batch driver built from
    github.com/wjakob/instant-meshes; native arm64, real oneTBB).

Stage 3  Finalise (finalize_rep.py)
    Dual of the winning tiling -> cage. Harmonic force-field relaxation
    (bond-stretch to a common edge length L0 + angle-bend to 120deg hexagons /
    108deg pentagons; 8000 vectorised steps) makes the tiles rigid and near-
    regular. 7-DOF fit to the mesh WITH ORIENTATION RESOLUTION: because the two
    short principal axes of a spindle are near-degenerate and PCA axes are only
    defined up to sign/reflection, the fit searches the short-axis swap x signs
    and keeps the assignment minimising cage->mesh distance (see NOTES). Reports
    regularity (bond CV, angle deviations) and fit (mean %, Hausdorff95 %).

Stage 4  Roll alignment (roll_align.py)
    The similarity fit leaves the roll about the long axis unconstrained; this
    optimises that 1 DOF to register the cage pentamers onto the bowl sites, then
    writes viewer_data.json + /tmp/render_data.json for visualisation.

Stage 5  Canonical RSPI (optional; run_fullerene.sh + Fullerene v4.5)
    Feeds the relaxed cage to the Fullerene program to obtain the canonical
    ring-spiral pentagon indices (RSPI) and independent topology validation.
    Ref: P. Schwerdtfeger, L. N. Wirz, J. Avery, "Program Fullerene: A software
    package for constructing and analyzing structures of regular fullerenes",
    J. Comput. Chem. 34 (2013) 1508. RSPI: Fowler & Manolopoulos, "An Atlas of
    Fullerenes" (Oxford, 1995).

Background / motivation:
    HIV-1 capsid fullerene-cone model: Ganser, Li, Klishko, Finch, Sundquist,
    "Assembly and Analysis of Conical Models for the HIV-1 Core", Science 283
    (1999) 80; Zhao et al., Nature 497 (2013) 643.
    Why extra 5-7 "scars" appear on curved crystals (and why we reject them for a
    TRUE fullerene): Bausch et al., "Grain Boundary Scars and Spherical
    Crystallography", Science 299 (2003) 1716.


METRICS
-------
  pent_err     mean bowl-target -> nearest pentamer distance, in hexamer diameters
               (hexamer diameter estimated from mesh area assuming ~242 hexamers)
  fit mean     mean cage->mesh nearest-point distance / mean mesh radius, in %
  Hausdorff95  95th percentile of two-sided nearest distances / mean radius, in %
  defects      count of capsomers with valence != 5 or 6 (must be 0 for a TRUE
               fullerene; a nonzero value = a 5-7 scar pair -> REJECT)
  IPR          isolated-pentagon rule: no two pentamers share an atom/edge


HOW TO USE
----------
Dependencies:
  * Python venv (.venv) with numpy + scipy.
  * instant-meshes/InstantMeshesBatch  (prebuilt headless binary; see that dir).
  * (optional) Fullerene v4.5 + the conda "fort" env, driven by run_fullerene.sh,
    for canonical RSPI. Skipped automatically if unavailable.
  * (optional) Blender + Blender-MCP for visualisation.

Single-mesh run (end to end):
    ./run_all.sh mesh4a.stl
  Produces:
    /tmp/mesh4a.npz, /tmp/mesh4a.obj      prepared mesh
    /tmp/curv.npz                         per-vertex curvature
    /tmp/pentamers.npz                    12 bowl-anchored pentamer sites
    /tmp/search_result.json               winning tiling + near-tie band
    viewer_data.json                      final cage (atoms, bonds, faces, fit)
    mesh4a_C<N>_representative.xyz         relaxed cage coordinates (N carbons)
  Typical wall time: ~1-2 min (search ~20-60s, relax/fit a few s).

Variation study (N independent instances):
    NRUNS=10 python ensemble_run.py
  Placement is deterministic; the variation source is the stochastic remesher.
  Writes /tmp/ensemble/run_1..N.json + summary.json with per-run C, H, defects,
  IPR, fit, Hausdorff, pent_err (belly/tip), aspect. Auto-flags scar rejects.

To run on a DIFFERENT mesh: pass its STL to run_all.sh. prep_mesh.py welds and
manifold-checks it; everything downstream is generic. (For non-spindle shapes,
widen the H grid in full_search.py so 2H+20 spans the expected carbon count.)

Pseudo-atomic capsid (mesh -> atomic model):
    python build_atomic_capsid.py mesh4a_C<N>_representative.xyz mesh4a.stl
  Decorates the fitted cage with real CA capsomers (RCSB 3H47 hexamer + 3P05
  pentamer, auto-fetched to ./capsomer_templates/): one per cage face, symmetry
  axis on the local facet normal, subunits rotated toward the inter-capsomer
  2-fold edges. All steric overlap is then removed by tethered rigid-body
  relaxation (Calpha pass then all-heavy-atom pass; capsomers move only ~1-2 A /
  a few degrees). Result: no inter-capsomer heavy-atom contact < 2.6 A at native
  ~95.5 A spacing. The whole model is finally reframed into the SOURCE-MESH frame
  (original cryo-ET coordinates) so the input mesh overlays it directly. Writes,
  in that shared Angstrom frame:
    <prefix>_capsid_atomic.pdb    all-atom (~2.3M atoms; hybrid-36 serials,
                                  segID cols 73-76 = capsomer Hnnn/Pnnn)
    <prefix>_capsid_atomic.cif    same as mmCIF (unique chain id per subunit)
    <prefix>_capsid_backbone.pdb  N,CA,C,O only; pentamers flagged segID P* and
                                  B-factor=100 (color-by-B shows the 12 pentamers)
    <mesh>_surface.obj            the source mesh (same frame; overlays the model)
  Uses the repo .venv (numpy+scipy); needs network on first run for the templates.
  The multi-hundred-MB PDB/CIF are .gitignore'd (>GitHub's 100MB limit) -- they
  regenerate deterministically from this one command.

  Space-saving instance form (--emit transforms): all capsomers are copies of the
  same two canonical capsomers, so instead of ~180MB of coordinates you can keep a
  ~50KB <prefix>_capsid_transforms.json (one rigid operator per capsomer,
  world = R @ canonical_template + t) plus the shared canonical templates. Rebuild
  the full PDB/CIF/backbone on demand:
      python build_atomic_capsid.py <cage>.xyz <mesh> --emit transforms
      python expand_capsid.py <prefix>_capsid_transforms.json [--cif] [--backbone]
  (--emit both, the default, writes the full coordinates AND the transforms.)


RESULTS ON mesh4a (HIV-1 capsid core, spindle, asphericity 2.60)
----------------------------------------------------------------
Over a 10-instance ensemble through the finalised pipeline:
  * fit mean       3.7 - 4.0 %   (mean 3.83, std 0.08)
  * Hausdorff95   ~10 %          (~6.5 % one-sided)
  * pent_err       0.49 - 0.70 hexamer-dia (mean ~0.6); belly comparable to tips
  * shape          aspect ~ 0.50 / 0.47 EVERY run (set by deterministic bowl
                   placement; reproducible independent of the stochastic tiling)
  * size           a bimodal plateau, C ~ 482 - 516 (H 231 - 242); no unique
                   winner -- a resolution-limited band of near-equivalent cages
  * topology       ~85 % of runs are TRUE fullerenes (12 isolated pentagons,
                   defect-free); the rest carry a 5-7 scar pair and are rejected
                   on defects>0.

Shape-dependent difficulty (observed across meshes): pentamer-placement precision
and defect-free yield both scale with how strongly the shape LOCALIZES curvature.
Cones/spindles with sharp tips (mesh4a, mesh5) pin the disclinations tightly
(pent_err ~0.4-0.6) and tile defect-free easily (~85%). Near-SPHERICAL targets
(cage_7, aspect ~0.96/0.85) are the hard case: curvature is nearly uniform so
disclinations barely localize (pent_err ~1.2) AND the field-aligned tiling tends
to nucleate 5-7 grain-boundary scars (Bausch-Bowick regime), making a clean
12-disclination (true) fullerene tiling rare -- it may take many multistart draws
to find one, and the search escalates automatically (full_search.py, cap via argv).
The gross-shape fit stays good (~3.4%) regardless; it is the strict pentagon-only
TOPOLOGY that is hard on a sphere.

Structural ceiling: a regular pentagon+hexagon-only fullerene can represent only
non-negative Gaussian curvature and relaxes slightly rounder than the mesh
(aspect 0.50 vs 0.43); the belly of the capsid carries some negative (saddle)
curvature (~21 % of vertices) that no TRUE fullerene can reproduce. This caps the
SURFACE fit at ~4 %, but does NOT displace the pentamers off the bowls. Matching
the belly exactly would require 5-7 scars (a defected cage), which is outside the
"true fullerene" definition adopted here.


FILE MAP
--------
Working pipeline (in run order):
    prep_mesh.py            0   STL/PLY -> manifold npz/obj (+ auto-subdivide coarse)
    subdivide_mesh.py       0b  Loop subdivision (refine+smooth); used by prep + CLI
    principal_curvature.py  1a  per-vertex K, H, k1, k2, bowl
    charge_partition_bowl.py 1b 12 bowl-anchored pentamer sites (deterministic)
    full_search.py          2   H-scan x multistart search + composite ranking
    finalize_rep.py         3   dual + FF relax + orientation-resolved 7-DOF fit
    roll_align.py           4   close roll gauge freedom; write viewer/render data
    run_fullerene.sh        5   canonical RSPI via Fullerene v4.5 (optional)
    build_atomic_capsid.py  6   decorate cage with CA capsomers (3H47/3P05) ->
                                clash-free pseudo-atomic capsid PDB/mmCIF/backbone
    expand_capsid.py        6b  rebuild the full capsid from a --emit transforms JSON
    ensemble_run.py             N-instance variation study
    run_all.sh                  orchestrates stages 0-5
    instant-meshes/             field-aligned remesher (see build notes there)

Support / earlier exploration (not in the critical path):
    charge_partition.py         equal-charge-CENTROID placement (superseded by
                                the bowl-peak version)
    orient_field.py             trivial-connections 6-RoSy field with prescribed
                                singularities (Crane-Desbrun-Schroeder, SGP 2010);
                                explored, not used by full_search
    heal_final.py               flip-based defect annealer (worked around by
                                defect-free multistart)
    search.py, scan_H.py, geo_fullerene.py, cap_tube*.py, build_*.py,
    field_tile*.py, relax_*.py, scratch_*.py, ...   dead-end / diagnostic scripts

Visualisation note: cages saved by the FINALISED pipeline are correctly oriented
(align to the mesh under identity). Cages saved by pre-fix runs need a short-axis
swap (equivalently, in Blender, scale Y by -1 then rotate X by -90deg, which
composes to (x,y,z)->(x,z,y)) to align -- see the orientation discussion in NOTES.


NOTES
-----
Orientation resolution (important). The 7-DOF fit aligns principal axes and
uniformly scales. Two ambiguities can otherwise corrupt the result silently:
(1) the spindle's two short axes are near-degenerate (0.50 vs 0.47), so their
ORDER can come out swapped versus the mesh; (2) PCA axes are defined only up to
sign, and a fullerene fits its mirror image equally well. finalize_rep.py resolves
this by searching the short-axis swap x sign assignments and keeping the minimum
cage->mesh distance. Skipping this lands on a reflected/rolled fit that scores
almost as well on nearest-distance but is visibly rotated and inflates every
metric (observed: fit 5.7% vs the correct 3.8%; belly pent_err 1.34 vs 0.6).
================================================================================
