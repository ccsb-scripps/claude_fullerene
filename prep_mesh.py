#!/usr/bin/env python3
"""Step 0 -- mesh preparation.

Read a binary STL of a segmented surface (default: mesh4a.stl), weld the
per-triangle vertex soup into an indexed manifold mesh, and write the two inputs
the rest of the pipeline consumes:
    /tmp/mesh4a.npz   (V, F, c)   -- for the Python stages
    /tmp/mesh4a.obj              -- for Instant Meshes

Usage:  python prep_mesh.py [input.stl]
"""
import sys, struct, numpy as np

fn = sys.argv[1] if len(sys.argv) > 1 else "mesh4a.stl"
buf = open(fn, "rb").read()
ntri = struct.unpack("<I", buf[80:84])[0]
# 50 bytes/triangle: normal(3f) + 3 vertices(3f each) + attrib(2 bytes)
tris = np.frombuffer(buf, dtype=np.dtype([("n","<3f4"),("v","<3,3f4"),("a","<u2")]),
                     count=ntri, offset=84)
P = tris["v"].reshape(-1, 3).astype(np.float64)          # 3*ntri raw vertices

# weld coincident vertices (STL stores them un-indexed)
diag = np.linalg.norm(P.max(0) - P.min(0))
tol = 1e-5 * diag
key = np.round(P / tol).astype(np.int64)
# np.unique returns (unique, index, inverse) in that fixed order
_, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
inv = np.asarray(inv).ravel()                            # numpy 2.0 may return (n,1)
order = np.argsort(first)                                # stable vertex ordering
remap = np.empty(len(first), np.int64); remap[order] = np.arange(len(first))
V = P[first][order]
F = remap[inv].reshape(ntri, 3)

# manifold / Euler check
from collections import defaultdict
ec = defaultdict(int)
for f in F:
    for i in range(3):
        a, b = f[i], f[(i+1)%3]; ec[(min(a,b), max(a,b))] += 1
E = len(ec); nonman = sum(1 for c in ec.values() if c != 2)
print("welded %d raw -> %d verts, %d faces | E=%d Euler(V-E+F)=%d nonmanifold_edges=%d"
      % (len(P), len(V), len(F), E, len(V)-E+len(F), nonman))

c = V.mean(0)
np.savez("/tmp/mesh4a.npz", V=V, F=F, c=c)
with open("/tmp/mesh4a.obj", "w") as f:
    for p in V: f.write("v %.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
    for t in F: f.write("f %d %d %d\n" % (t[0]+1, t[1]+1, t[2]+1))
print("wrote /tmp/mesh4a.npz and /tmp/mesh4a.obj")
