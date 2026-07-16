#!/usr/bin/env python3
"""Convert OBJ meshes to color-less PLY (ASCII) and binary STL, so viewers like
Mol* let you set color + opacity in the UI (an OBJ's material otherwise fixes them).
Usage: python obj_to_ply_stl.py <mesh.obj> [more.obj ...]  -> writes <mesh>.ply/.stl"""
import sys, struct, numpy as np
def readobj(fn):
    V=[]; F=[]
    for ln in open(fn):
        if ln.startswith("v "): p=ln.split(); V.append([float(p[1]),float(p[2]),float(p[3])])
        elif ln.startswith("f "): F.append([int(t.split("/")[0])-1 for t in ln.split()[1:]])
    return np.array(V,float), F
def tris(F):
    T=[]
    for f in F:
        for i in range(1,len(f)-1): T.append((f[0],f[i],f[i+1]))   # fan-triangulate n-gons
    return T
def write_ply(fn,V,T):
    with open(fn,"w") as o:
        o.write("ply\nformat ascii 1.0\ncomment no color -- set color/opacity in the viewer\n")
        o.write("element vertex %d\nproperty float x\nproperty float y\nproperty float z\n"%len(V))
        o.write("element face %d\nproperty list uchar int vertex_indices\nend_header\n"%len(T))
        for x,y,z in V: o.write("%.4f %.4f %.4f\n"%(x,y,z))
        for a,b,c in T: o.write("3 %d %d %d\n"%(a,b,c))
def write_stl(fn,V,T):
    with open(fn,"wb") as o:
        o.write(b'\0'*80); o.write(struct.pack('<I',len(T)))
        for a,b,c in T:
            p0,p1,p2=V[a],V[b],V[c]; n=np.cross(p1-p0,p2-p0); L=np.linalg.norm(n); n=n/L if L>1e-12 else n*0
            o.write(struct.pack('<3f',*n))
            for p in (p0,p1,p2): o.write(struct.pack('<3f',*p))
            o.write(struct.pack('<H',0))
for obj in sys.argv[1:]:
    V,F=readobj(obj); T=tris(F); base=obj[:-4]
    write_ply(base+".ply",V,T); write_stl(base+".stl",V,T)
    print("%-52s -> .ply + .stl  (%d verts, %d tris)"%(obj.split('/')[-1],len(V),len(T)))
