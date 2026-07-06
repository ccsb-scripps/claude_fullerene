#!/usr/bin/env python3
"""Scan number of hexamers H in [220,250]. For each H: place 12+H capsomers on
mesh4a (12 pinned at curvature bowls, rest Lloyd-relaxed), build the intrinsic
(surface-metric) Delaunay triangulation, and score how close it is to a valid
fullerene: valence histogram should be {5:12, 6:H}. Report defects + dual edge CV."""
import numpy as np, sys
from scipy.spatial import ConvexHull, cKDTree
from collections import defaultdict

m=np.load("/tmp/mesh4a.npz"); V,F,cm=m['V'],m['F'],m['c']
b=np.load("/tmp/bowls.npz"); bowls=b['pos']            # 12 x 3 on surface
v0=V[F[:,0]]; e1=V[F[:,1]]-v0; e2=V[F[:,2]]-v0

def raycast(dirs):
    out=np.zeros((len(dirs),3)); eps=1e-6
    for i,dv in enumerate(dirs):
        h=np.cross(dv,e2); a=np.einsum('ij,ij->i',e1,h); mk=np.abs(a)>eps
        f=np.zeros_like(a); f[mk]=1/a[mk]; s=cm-v0
        u=f*np.einsum('ij,ij->i',s,h); q=np.cross(s,e1)
        vv=f*np.einsum('j,ij->i',dv,q); t=f*np.einsum('ij,ij->i',e2,q)
        ok=mk&(u>=-1e-4)&(vv>=-1e-4)&(u+vv<=1+1e-4)&(t>1e-3)
        out[i]=cm+(t[ok].max() if ok.any() else 1.0)*dv
    return out

def farthest_point(N, fixed):
    pts=[p for p in fixed]
    # candidate pool = mesh vertices
    cand=V.copy()
    d=np.min(np.linalg.norm(cand[:,None,:]-np.array(pts)[None,:,:],axis=2),axis=1)
    while len(pts)<N:
        i=int(np.argmax(d)); pts.append(cand[i])
        nd=np.linalg.norm(cand-cand[i],axis=1); d=np.minimum(d,nd)
    return np.array(pts)

def lloyd(caps, npin, iters=18):
    caps=caps.copy()
    for _ in range(iters):
        tree=cKDTree(caps); _,lab=tree.query(V)
        new=caps.copy()
        for j in range(len(caps)):
            sel=V[lab==j]
            if len(sel)>0: new[j]=sel.mean(0)
        new[:npin]=bowls                       # pin bowls
        dirs=new-cm; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
        caps=raycast(dirs); caps[:npin]=bowls
    return caps

def triangulation(caps):
    dirs=caps-cm; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    hull=ConvexHull(dirs)
    return [tuple(s) for s in hull.simplices]

def intrinsic_delaunay(caps,tris,passes=6):
    # edge -> triangles
    def build(tris):
        em=defaultdict(list)
        for ti,t in enumerate(tris):
            for i in range(3):
                a,b=t[i],t[(i+1)%3]; em[(min(a,b),max(a,b))].append(ti)
        return em
    tris=[list(t) for t in tris]
    for _ in range(passes):
        em=build(tris); flips=0
        for e,tt in list(em.items()):
            if len(tt)!=2: continue
            ti,tj=tt; a,b=e
            c=[v for v in tris[ti] if v!=a and v!=b]; d=[v for v in tris[tj] if v!=a and v!=b]
            if not c or not d: continue
            c,d=c[0],d[0]
            # Delaunay: flip if angle(c)+angle(d) > pi  (use 3D distances)
            def ang(p,q,r):  # angle at p in triangle p,q,r
                u=caps[q]-caps[p]; w=caps[r]-caps[p]
                cs=np.dot(u,w)/(np.linalg.norm(u)*np.linalg.norm(w)+1e-12)
                return np.arccos(np.clip(cs,-1,1))
            if ang(c,a,b)+ang(d,a,b) > np.pi+1e-6:
                # avoid duplicate edge
                ok=True
                for t in tris:
                    if c in t and d in t: ok=False;break
                if ok:
                    tris[ti]=[a,c,d]; tris[tj]=[b,c,d]; flips+=1
        if flips==0: break
    return [tuple(t) for t in tris]

def valences(tris,N):
    deg=np.zeros(N,int)
    for t in tris:
        for v in t: deg[v]+=1
    return deg

def dual_edge_cv(caps,tris):
    # dual edges connect adjacent triangle circumcenters(~centroids); use centroids on surface
    cent=np.array([caps[list(t)].mean(0) for t in tris])
    em=defaultdict(list)
    for ti,t in enumerate(tris):
        for i in range(3):
            a,b=t[i],t[(i+1)%3]; em[(min(a,b),max(a,b))].append(ti)
    L=[np.linalg.norm(cent[tt[0]]-cent[tt[1]]) for tt in em.values() if len(tt)==2]
    L=np.array(L); return L.std()/L.mean()*100

print("H   Ccarbons  n5  n6  n7+  n4-  bowls@deg5  dualEdgeCV%")
results=[]
Hlist=[int(x) for x in sys.argv[1:]] or list(range(220,251,5))
for H in Hlist:
    N=12+H
    init=farthest_point(N,list(bowls))
    caps=lloyd(init,12)
    tris=triangulation(caps)
    tris=intrinsic_delaunay(caps,tris)
    deg=valences(tris,N)
    n5=int((deg==5).sum()); n6=int((deg==6).sum()); n7=int((deg>=7).sum()); n4=int((deg<=4).sum())
    bd5=int((deg[:12]==5).sum())
    cv=dual_edge_cv(caps,tris)
    C=2*H+20
    print("%3d  C%-4d    %3d %3d  %3d  %3d   %2d/12       %.1f"%(H,C,n5,n6,n7,n4,bd5,cv))
    results.append((H,n5,n6,n7,n4,bd5,cv,caps,tris))
np.save("/tmp/scan_results.npy",np.array([(r[0],r[1],r[2],r[3],r[4],r[5],r[6]) for r in results]))
# save the cleanest (min defects) candidate
def defects(r): return abs(r[1]-12)+r[3]+r[4]+(12-r[5])
best=min(results,key=defects)
np.savez("/tmp/best_candidate.npz",H=best[0],caps=best[7],tris=np.array(best[8]))
print("\ncleanest H=%d (defect score %d) saved"%(best[0],defects(best)))
