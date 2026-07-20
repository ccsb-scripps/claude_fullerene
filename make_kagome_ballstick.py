#!/usr/bin/env python3
"""Ball-and-stick Kagome lattice = the MEDIAL graph of a fullerene cage: nodes at
each fullerene edge-midpoint (grey dots / dimer 2-folds), edges = the trihexagonal
(Kagome) connectivity -> triangles (3-fold trimer gaps) + hexagons + 12 pentagons.
Reads capsid_models/<mesh>_C<N>_cage.obj (fullerene polyhedron), writes _kagome_ballstick.obj."""
import numpy as np, glob, os
from collections import defaultdict
def readobj(fn):
    V=[];F=[]
    for ln in open(fn):
        if ln.startswith("v "): p=ln.split(); V.append([float(p[1]),float(p[2]),float(p[3])])
        elif ln.startswith("f "): F.append([int(t.split("/")[0])-1 for t in ln.split()[1:]])
    return np.array(V),F
def ico(sub=1):
    t=(1+5**0.5)/2
    V=np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]],float)
    F=[[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    V=V/np.linalg.norm(V,axis=1,keepdims=True)
    for _ in range(sub):
        mid={};nV=list(V);nF=[]
        def m(a,b):
            k=(min(a,b),max(a,b))
            if k not in mid: p=(V[a]+V[b])/2; mid[k]=len(nV); nV.append(p/np.linalg.norm(p))
            return mid[k]
        for a,b,c in F: ab,bc,ca=m(a,b),m(b,c),m(c,a); nF+=[[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        V=np.array(nV);F=nF
    return V,F
def cyl(p1,p2,r,n=8):
    d=p2-p1;L=np.linalg.norm(d)
    if L<1e-9: return np.zeros((0,3)),[]
    z=d/L;a=np.array([1,0,0.]) if abs(z[0])<0.9 else np.array([0,1,0.]); x=np.cross(a,z);x/=np.linalg.norm(x);y=np.cross(z,x)
    ring=np.array([np.cos(t)*x+np.sin(t)*y for t in np.linspace(0,2*np.pi,n,endpoint=False)])*r
    V=np.vstack([p1+ring,p2+ring]);F=[]
    for i in range(n): j=(i+1)%n;F+=[[i,j,n+j],[i,n+j,n+i]]
    return V,F
ISPH,IFAC=ico(1)
for cf in sorted(glob.glob("capsid_models/*_C*_cage.obj")):
    if "ballstick" in cf or "kagome" in cf: continue
    pre=os.path.basename(cf).replace("_cage.obj",""); V,F=readobj(cf)
    # medial graph: node per fullerene edge (midpoint); kagome edges = consecutive edge-midpoints on each face
    eid={}; nodes=[]
    def eget(a,b):
        k=(min(a,b),max(a,b))
        if k not in eid: eid[k]=len(nodes); nodes.append((V[a]+V[b])/2)
        return eid[k]
    kedges=set()
    for f in F:
        k=len(f); mids=[eget(f[i],f[(i+1)%k]) for i in range(k)]
        for i in range(k):
            a,b=mids[i],mids[(i+1)%k]; kedges.add((min(a,b),max(a,b)))
    nodes=np.array(nodes); kedges=sorted(kedges)
    Lk=np.median([np.linalg.norm(nodes[a]-nodes[b]) for a,b in kedges]); rb=0.22*Lk; rs=0.09*Lk
    aV=[];aF=[];off=0
    for p in nodes:
        aV.append(ISPH*rb+p)
        for a,b,c in IFAC: aF.append((a+off,b+off,c+off))
        off+=len(ISPH)
    for a,b in kedges:
        cv,cfc=cyl(nodes[a],nodes[b],rs); aV.append(cv)
        for fc in cfc: aF.append(tuple(x+off for x in fc))
        off+=len(cv)
    aV=np.vstack(aV); out="capsid_models/%s_kagome_ballstick.obj"%pre
    with open(out,"w") as fo:
        fo.write("# Kagome lattice ball-and-stick (medial graph of fullerene; nodes=edge-midpoints/dimers)\no kagome\n")
        for v in aV: fo.write("v %.3f %.3f %.3f\n"%(v[0],v[1],v[2]))
        for a,b,c in aF: fo.write("f %d %d %d\n"%(a+1,b+1,c+1))
    ntri=sum(1 for f in F if len(f)==5); print("%-12s %d nodes(edges) %d kagome-edges (kedge %.0fA) -> %d verts/%d faces"%(pre,len(nodes),len(kedges),Lk,len(aV),len(aF)))
