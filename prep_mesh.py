#!/usr/bin/env python3
"""Step 0 -- mesh preparation.

Read a segmented closed surface (binary STL or binary PLY) and write the two
inputs the rest of the pipeline consumes:
    /tmp/mesh4a.npz   (V, F, c)   -- for the Python stages
    /tmp/mesh4a.obj              -- for Instant Meshes
STL is a per-triangle vertex soup, so it is welded into an indexed manifold; PLY
is already indexed and used as-is. Only triangular faces are kept.

Usage:  python prep_mesh.py [input.stl|input.ply]
"""
import sys, struct, numpy as np

fn = sys.argv[1] if len(sys.argv) > 1 else "mesh4a.stl"
buf = open(fn, "rb").read()

def load_stl(buf):
    ntri = struct.unpack("<I", buf[80:84])[0]
    # 50 bytes/triangle: normal(3f) + 3 vertices(3f each) + attrib(2 bytes)
    tris = np.frombuffer(buf, dtype=np.dtype([("n","<3f4"),("v","<3,3f4"),("a","<u2")]),
                         count=ntri, offset=84)
    P = tris["v"].reshape(-1, 3).astype(np.float64)      # 3*ntri raw vertices
    diag = np.linalg.norm(P.max(0) - P.min(0)); tol = 1e-5 * diag
    key = np.round(P / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()                        # numpy 2.0 may return (n,1)
    order = np.argsort(first)
    remap = np.empty(len(first), np.int64); remap[order] = np.arange(len(first))
    V = P[first][order]; F = remap[inv].reshape(ntri, 3)
    print("welded %d raw -> %d verts" % (len(P), len(V)))
    return V, F

def load_ply(buf):
    hpos = buf.index(b"end_header"); he = buf.index(b"\n", hpos) + 1   # handles LF or CRLF
    hdr = buf[:he].decode("ascii", "replace").splitlines()
    fmt = next(l for l in hdr if l.startswith("format"))
    nv = int(next(l for l in hdr if l.startswith("element vertex")).split()[-1])
    nf = int(next(l for l in hdr if l.startswith("element face")).split()[-1])
    pv = 0; inv = False                                  # count vertex-block properties
    for l in hdr:
        if l.startswith("element vertex"): inv = True; continue
        if l.startswith("element "): inv = False
        if inv and l.startswith("property"): pv += 1
    if "ascii" in fmt:
        it = iter(buf[he:].decode("ascii", "replace").split())
        V = np.array([[float(next(it)) for _ in range(pv)][:3] for _ in range(nv)], float)
        F = []
        for _ in range(nf):
            k = int(next(it)); idx = [int(next(it)) for _ in range(k)]
            if k == 3: F.append(idx)
        return V, np.array(F)
    off = he                                             # binary_little_endian
    V = np.frombuffer(buf, dtype="<f4", count=nv*pv, offset=off).reshape(nv, pv)[:, :3].astype(np.float64)
    off += nv*pv*4
    F = []
    for _ in range(nf):                                  # face = uchar count + count*uint32
        k = buf[off]; off += 1
        idx = np.frombuffer(buf, dtype="<u4", count=k, offset=off); off += k*4
        if k == 3: F.append(list(idx))
    return V, np.array(F)

if buf[:3] == b"ply":
    V, F = load_ply(buf)
else:
    V, F = load_stl(buf)

# manifold / Euler check
from collections import defaultdict
ec = defaultdict(int)
for f in F:
    for i in range(3):
        a, b = f[i], f[(i+1)%3]; ec[(min(a,b), max(a,b))] += 1
E = len(ec); nonman = sum(1 for c in ec.values() if c != 2)
print("%d verts, %d faces | E=%d Euler(V-E+F)=%d nonmanifold_edges=%d"
      % (len(V), len(F), E, len(V)-E+len(F), nonman))

# auto-subdivide COARSE inputs: below ~1000 verts the pi/3 charge partition
# overshoots and the surface is faceted, loosening pentamer placement. Loop-refine
# to >= ~4000 verts (fine meshes like mesh4a/mesh5 are already above trigger and
# are left untouched). Override via PREP_MIN_VERTS / PREP_TARGET_VERTS.
import os
trig = int(os.environ.get("PREP_MIN_VERTS", 1000))
tgt  = int(os.environ.get("PREP_TARGET_VERTS", 4000))
if len(V) < trig:
    from subdivide_mesh import subdivide_to
    V, F, lv = subdivide_to(V, [tuple(f) for f in F], tgt)
    F = np.array(F)
    print("coarse input (<%d verts) -> Loop x%d -> %d verts, %d faces" % (trig, lv, len(V), len(F)))

c = V.mean(0)
np.savez("/tmp/mesh4a.npz", V=V, F=F, c=c)
with open("/tmp/mesh4a.obj", "w") as f:
    for p in V: f.write("v %.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
    for t in F: f.write("f %d %d %d\n" % (t[0]+1, t[1]+1, t[2]+1))
print("wrote /tmp/mesh4a.npz and /tmp/mesh4a.obj")
