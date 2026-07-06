#!/bin/bash
# End-to-end bowl-anchored fullerene fit for a segmented mesh.
# Usage: ./run_all.sh [input.stl]   (default mesh4a.stl)
cd /Users/olson/Dev/fullerenes; source .venv/bin/activate
STL="${1:-mesh4a.stl}"
T0=$(date +%s); s(){ echo "   $1 $(( $(date +%s)-$2 ))s"; }

echo "===== bowl-anchored fullerene fit :: $STL ====="
echo "--- 0  mesh prep (STL -> manifold npz/obj) ---";         t=$(date +%s)
python -u prep_mesh.py "$STL" | { grep -E "welded|wrote" || true; };                        s step0 $t
echo "--- 1a principal curvature ---";                          t=$(date +%s)
python -u principal_curvature.py | { grep -E "verts" || true; };                            s step1a $t
echo "--- 1b bowl-anchored pentamer placement (deterministic) ---"; t=$(date +%s)
python -u charge_partition_bowl.py | { grep -E "charges|sites on bowls|axial" || true; };   s step1b $t
echo "--- 2  field-aligned search (H-scan x multistart, stochastic) ---"; t=$(date +%s)
python -u full_search.py | { grep -E "scanned|REPRESENTATIVE|near-tie|total" || true; };    s step2 $t
echo "--- 3  finalize: dual + FF relax + orientation-resolved fit ---"; t=$(date +%s)
python -u finalize_rep.py | { grep -E "fullerene C|regularity|fit:" || true; };             s step3 $t
echo "--- 4  close roll gauge freedom (register pentamers to bowls) ---"; t=$(date +%s)
python -u roll_align.py | { grep -E "roll opt|rewrote" || true; };                          s step4 $t
echo "--- 5  canonical RSPI via Fullerene v4.5 (optional) ---";  t=$(date +%s)
XYZ=$(ls -t mesh4a_C*_representative.xyz 2>/dev/null | head -1)
if [ -n "$XYZ" ]; then
  python -u - "$XYZ" <<'PY'
import numpy as np, sys
from scipy.spatial import cKDTree
X=[]
for ln in open(sys.argv[1]).read().split('\n')[2:]:
    t=ln.split()
    if len(t)>=4: X.append([float(t[1]),float(t[2]),float(t[3])])
X=np.array(X); dd,_=cKDTree(X).query(X,k=2); L=np.median(dd[:,1]); Xs=(X-X.mean(0))*(1.44/L)
open("/tmp/rspi.inp","w").write("fit\n&General NA=%d iwext=1 /\n&Coord ICart=1 /\n&FFChoice Iopt=4 /\n&FFParameters /\n&Hamilton /\n&Isomers /\n&Graph /\n"%len(X)+"".join("6 %.5f %.5f %.5f\n"%(p[0],p[1],p[2]) for p in Xs))
print("   NA=%d -> /tmp/rspi.inp"%len(X))
PY
  ./run_fullerene.sh /tmp/rspi.inp /tmp/rspi_prog.xyz >/dev/null 2>&1 || echo "   (Fullerene program step skipped)"
  grep -A2 "Ring spiral pentagon positions" /tmp/fullerene_run.log 2>/dev/null | tail -1 || true
fi
s step5 $t
echo "===== TOTAL: $(( $(date +%s)-T0 ))s  ->  viewer_data.json + mesh4a_C*_representative.xyz ====="
