#!/usr/bin/env python3
"""Loop subdivision (refine + smooth) for coarse meshes.

Coarse inputs lump curvature into few vertices, which (a) facets the surface the
cage must hug and (b) makes the equal-charge (pi/3) pentamer partition overshoot,
so the 12 bowl markers are imprecise. Loop subdivision refines each triangle into
4 and moves vertices toward the smooth limit surface, de-faceting it and letting
curvature/charge distribute finely.

Importable: `loop(V,F)` one level; `subdivide_to(V,F,target)` until >= target verts.
CLI (overwrites /tmp/mesh4a.npz + .obj):  python subdivide_mesh.py [levels]
"""
import sys, numpy as np
from collections import defaultdict

def loop(V, F):
    """One level of Loop subdivision. V: (n,3) array; F: list of 3-tuples."""
    nbr = defaultdict(set); ef = defaultdict(list)
    for f in F:
        for i in range(3):
            a, b, o = f[i], f[(i+1)%3], f[(i+2)%3]
            nbr[a].add(b); nbr[a].add(o); ef[(min(a,b),max(a,b))].append(o)
    newV = [v.copy() for v in V]; eidx = {}
    for e, opp in ef.items():
        a, b = e
        p = (3/8*(V[a]+V[b]) + 1/8*(V[opp[0]]+V[opp[1]])) if len(opp) == 2 else 0.5*(V[a]+V[b])
        eidx[e] = len(newV); newV.append(p)
    for i in range(len(V)):                      # reposition originals (Warren weights)
        nb = list(nbr[i]); n = len(nb)
        if n:
            beta = (1.0/n)*(5.0/8 - (3.0/8 + 0.25*np.cos(2*np.pi/n))**2)
            newV[i] = (1-n*beta)*V[i] + beta*sum(V[j] for j in nb)
    newF = []
    for a, b, c in F:
        ab = eidx[(min(a,b),max(a,b))]; bc = eidx[(min(b,c),max(b,c))]; ca = eidx[(min(c,a),max(c,a))]
        newF += [(a,ab,ca), (b,bc,ab), (c,ca,bc), (ab,bc,ca)]
    return np.array(newV), newF

def subdivide_to(V, F, target):
    """Loop-subdivide until vertex count >= target. Returns (V, F, levels)."""
    lv = 0
    while len(V) < target:
        V, F = loop(V, F); lv += 1
    return V, F, lv

if __name__ == "__main__":
    levels = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    m = np.load("/tmp/mesh4a.npz"); V, F = m['V'], [tuple(f) for f in m['F']]
    for _ in range(levels):
        V, F = loop(V, F)
    F = np.array(F)
    ec = defaultdict(int)
    for f in F:
        for i in range(3):
            a, b = f[i], f[(i+1)%3]; ec[(min(a,b),max(a,b))] += 1
    print("Loop x%d -> %d verts, %d faces | Euler=%d nonmanifold=%d"
          % (levels, len(V), len(F), len(V)-len(ec)+len(F), sum(1 for c in ec.values() if c != 2)))
    np.savez("/tmp/mesh4a.npz", V=V, F=F, c=V.mean(0))
    with open("/tmp/mesh4a.obj", "w") as f:
        for p in V: f.write("v %.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
        for t in F: f.write("f %d %d %d\n" % (t[0]+1, t[1]+1, t[2]+1))
    print("wrote higher-res /tmp/mesh4a.npz + /tmp/mesh4a.obj")
