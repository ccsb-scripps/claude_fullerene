#!/usr/bin/env python3
"""Ball-and-stick OBJ of each fullerene cage (spheres=carbons, cylinders=bonds),
from the polyhedron capsid_models/<mesh>_C<N>_cage.obj (bonds = its face edges).
Reads better than the faceted polyhedron in surface viewers like Mol*."""
import numpy as np, glob, os
def readobj(fn):
    V=[]; F=[]
    for ln in open(fn):
        if ln.startswith("v "): p=ln.split(); V.append([float(p[1]),float(p[2]),float(p[3])])
        elif ln.startswith("f "): F.append([int(t.split("/")[0])-1 for t in ln.split()[1:]])
    return np.array(V), F
def edges(F):
    E=set()
    for f in F:
        k=len(f)
        for i in range(k): a,b=f[i],f[(i+1)%k]; E.add((min(a,b),max(a,b)))
    return sorted(E)
def ico(subdiv=1):
    t=(1+5**0.5)/2
    V=np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]],float)
    F=[[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    V=V/np.linalg.norm(V,axis=1,keepdims=True)
    for _ in range(subdiv):
        mid={}; nV=list(V); nF=[]
        def m(a,b):
            k=(min(a,b),max(a,b))
            if k not in mid: p=(V[a]+V[b])/2; mid[k]=len(nV); nV.append(p/np.linalg.norm(p))
            return mid[k]
        for a,b,c in F:
            ab,bc,ca=m(a,b),m(b,c),m(c,a); nF+=[[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        V=np.array(nV); F=nF
    return V,F
def cyl(p1,p2,r,n=8):
    d=p2-p1; L=np.linalg.norm(d)
    if L<1e-9: return np.zeros((0,3)),[]
    zc=d/L; a=np.array([1,0,0.]) if abs(zc[0])<0.9 else np.array([0,1,0.])
    x=np.cross(a,zc); x/=np.linalg.norm(x); y=np.cross(zc,x)
    ring=np.array([np.cos(t)*x+np.sin(t)*y for t in np.linspace(0,2*np.pi,n,endpoint=False)])*r
    V=np.vstack([p1+ring,p2+ring]); F=[]
    for i in range(n):
        j=(i+1)%n; F+=[[i,j,n+j],[i,n+j,n+i]]
    return V,F
ISPH,IFAC=ico(1)
for cf in sorted(glob.glob("capsid_models/*_C*_cage.obj")):
    if "ballstick" in cf: continue
    pre=os.path.basename(cf).replace("_cage.obj",""); V,F=readobj(cf); E=edges(F)
    bl=np.median([np.linalg.norm(V[a]-V[b]) for a,b in E]); rb=0.18*bl; rs=0.07*bl
    aV=[]; aF=[]; off=0
    for p in V:
        aV.append(ISPH*rb+p)
        for a,b,c in IFAC: aF.append((a+off,b+off,c+off))
        off+=len(ISPH)
    for a,b in E:
        cv,cfc=cyl(V[a],V[b],rs)
        aV.append(cv)
        for f in cfc: aF.append(tuple(x+off for x in f))
        off+=len(cv)
    aV=np.vstack(aV); out="capsid_models/%s_cage_ballstick.obj"%pre
    with open(out,"w") as fo:
        fo.write("# fullerene cage ball-and-stick: spheres=carbons, cylinders=bonds (Angstrom)\no cage_ballstick\n")
        for v in aV: fo.write("v %.3f %.3f %.3f\n"%(v[0],v[1],v[2]))
        for a,b,c in aF: fo.write("f %d %d %d\n"%(a+1,b+1,c+1))
    print("%-13s %d balls + %d sticks (bond %.0fA, ball %.0fA, stick %.0fA) -> %d verts/%d faces"%(pre,len(V),len(E),bl,rb,rs,len(aV),len(aF)))
