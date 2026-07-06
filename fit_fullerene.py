#!/usr/bin/env python3
"""
Build the best-fitting fullerene for mesh4a.stl.

Pipeline:
  1. Load mesh4a (target surface) and the 242-capsomer triangulation.
  2. Annihilate the two (5,7) dislocation dipoles by guided edge flips,
     protecting the 12 curvature-peak pentagons -> clean {5:12, 6:230}.
  3. Fullerene = DUAL of the triangulation: 480 carbons (one per triangle),
     each placed on the mesh4a surface (ray-cast from centroid).
  4. On-surface regularisation (tangential Laplacian + re-projection) to
     equalise bond lengths while keeping every atom on the mesh.
  5. Validate topology (12 pentagons, all deg-3) and geometry (bond CV,
     surface-fit RMS), export XYZ + edges.
"""
import numpy as np
from collections import defaultdict

TRI_FILE = "/Users/olson/Downloads/C480_triangulation.txt"
MESH_NPZ = "/tmp/mesh4a.npz"
OUT_XYZ  = "/Users/olson/Dev/fullerenes/mesh4a_fullerene_C480.xyz"
OUT_EDG  = "/Users/olson/Dev/fullerenes/mesh4a_fullerene_C480_edges.txt"

# ---------------------------------------------------------------- load
def load_tri(path):
    Vc={}; F=[]
    for ln in open(path):
        if ln.startswith('v '):
            p=ln.split(); Vc[int(p[1])]=[float(p[2]),float(p[3]),float(p[4])]
        elif ln.startswith('f '):
            p=ln.split(); F.append((int(p[1]),int(p[2]),int(p[3])))
    n=max(Vc)+1
    V=np.array([Vc[i] for i in range(n)])
    return V,F

def degrees(F,n):
    d=np.zeros(n,int)
    inc=defaultdict(set)
    for ti,f in enumerate(F):
        for v in f: d[v]+=1; inc[v].add(ti)
    return d,inc

def edge_map(F):
    em=defaultdict(list)
    for ti,f in enumerate(F):
        for i in range(3):
            a,b=f[i],f[(i+1)%3]; em[(min(a,b),max(a,b))].append(ti)
    return em

def validate(F,n,label=""):
    d,_=degrees(F,n)
    em=edge_map(F)
    E=len(em); nonman=sum(1 for t in em.values() if len(t)!=2)
    hist={k:int((d==k).sum()) for k in range(3,10) if (d==k).sum()}
    euler=n-E+len(F)
    charge=int((6-d).sum())
    print(f"[{label}] V={n} E={E} F={len(F)} Euler={euler} nonmanifold={nonman} "
          f"deg={hist} charge(Σ6-deg)={charge}")
    return d,em,hist,nonman,euler

# ---------------------------------------------------------------- flips
def graph_dist(F,n,src):
    adj=defaultdict(set)
    for f in F:
        for i in range(3):
            adj[f[i]].add(f[(i+1)%3]); adj[f[i]].add(f[(i+2)%3])
    import collections
    dist={src:0}; q=collections.deque([src])
    while q:
        u=q.popleft()
        for w in adj[u]:
            if w not in dist: dist[w]=dist[u]+1; q.append(w)
    return dist

def do_flip(F,em,ti,tj,edge):
    a,b=edge
    fi=F[ti]; fj=F[tj]
    c=[v for v in fi if v not in edge][0]
    d=[v for v in fj if v not in edge][0]
    if c==d: return None
    if (min(c,d),max(c,d)) in em and len(em[(min(c,d),max(c,d))])>0: return None
    F[ti]=(a,c,d); F[tj]=(b,c,d)
    return (a,b,c,d)

def energy(d,target):
    return float(((d-target)**2).sum())

def repair(F,n,protect):
    """Anneal edge flips so protected verts -> deg5, all others -> deg6."""
    target=np.full(n,6);
    for v in protect: target[v]=5
    rng=np.random.default_rng(0)
    best=[tuple(f) for f in F]; d,_=degrees(F,n); bestE=energy(d,target)
    for restart in range(40):
        F[:]=[tuple(f) for f in best]
        T=1.5
        for it in range(4000):
            d,_=degrees(F,n); em=edge_map(F)
            curE=energy(d,target)
            if curE==0: break
            # bias proposals toward edges touching a defect
            defect=[v for v in range(n) if d[v]!=target[v]]
            edges=list(em.keys())
            e=edges[rng.integers(len(edges))]
            if rng.random()<0.7 and defect:
                # pick an edge incident to a random defect vertex
                dv=defect[rng.integers(len(defect))]
                inc=[k for k in edges if dv in k]
                if inc: e=inc[rng.integers(len(inc))]
            tt=em[e]
            if len(tt)!=2: continue
            ti,tj=tt
            snap=(tuple(F[ti]),tuple(F[tj]))
            res=do_flip(F,em,ti,tj,e)
            if res is None: continue
            d2,_=degrees(F,n); newE=energy(d2,target)
            if newE<=curE or rng.random()<np.exp(-(newE-curE)/max(T,1e-6)):
                if newE<bestE:
                    bestE=newE; best=[tuple(f) for f in F]
                    if bestE==0: break
            else:
                F[ti],F[tj]=snap[0],snap[1]  # revert
            T*=0.9995
        if bestE==0: break
    F[:]=[tuple(f) for f in best]
    return bestE

# ---------------------------------------------------------------- geometry
def raycast_to_surface(origin, dirs, Vm, Fm):
    """For each unit dir, return surface point along ray from origin (star-shaped)."""
    v0=Vm[Fm[:,0]]; e1=Vm[Fm[:,1]]-v0; e2=Vm[Fm[:,2]]-v0
    out=np.zeros((len(dirs),3))
    eps=1e-6
    for i,dvec in enumerate(dirs):
        h=np.cross(dvec,e2); a=np.einsum('ij,ij->i',e1,h)
        mask=np.abs(a)>eps
        f=np.zeros_like(a); f[mask]=1.0/a[mask]
        s=origin-v0
        u=f*np.einsum('ij,ij->i',s,h)
        q=np.cross(s,e1)
        vv=f*np.einsum('j,ij->i',dvec,q)
        t=f*np.einsum('ij,ij->i',e2,q)
        ok=mask&(u>=-1e-4)&(vv>=-1e-4)&(u+vv<=1+1e-4)&(t>1e-3)
        if ok.any():
            tt=t[ok].max()          # outermost surface crossing
            out[i]=origin+tt*dvec
        else:
            out[i]=origin+dirs[i]*np.linalg.norm(Vm-origin,axis=1).mean()
    return out

def build_dual(F):
    """carbons = triangles; bonds = triangles sharing an edge; faces = around each capsomer."""
    em=edge_map(F)
    bonds=set()
    for tt in em.values():
        if len(tt)==2: bonds.add((min(tt),max(tt)))
    return sorted(bonds)

def main():
    m=np.load(MESH_NPZ); Vm,Fm,cm,axis=m['V'],m['F'],m['c'],m['axis']
    Vt,F=load_tri(TRI_FILE)
    n=len(Vt)
    d,em,hist,nm,eu=validate(F,n,"triangulation as-loaded")

    deg5=[v for v in range(n) if d[v]==5]
    deg7=[v for v in range(n) if d[v]==7]
    # protect 12 deg5 farthest from heptagons; sacrifice the 2 nearest
    score=np.zeros(len(deg5))
    for hi in deg7:
        gd=graph_dist(F,n,hi)
        for k,p in enumerate(deg5): score[k]+=gd.get(p,99)
    order=np.argsort(-score)   # farthest first
    protect=[deg5[i] for i in order[:12]]
    print("deg5:",len(deg5),"deg7:",deg7,"-> protect 12, sacrifice",
          [deg5[i] for i in order[12:]])

    bestE=repair(F,n,protect)
    d,em,hist,nm,eu=validate(F,n,"after repair")
    if bestE!=0:
        print("!! repair residual energy",bestE,"- proceeding with best found")

    # ---- dual geometry on the mesh surface ----
    Ftri=np.array(F)
    cent=Vt[Ftri].mean(1)                 # triangle centroids
    dirs=cent-cm; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    C=raycast_to_surface(cm,dirs,Vm,Fm)   # 480 carbons on surface
    bonds=build_dual(F)
    print("carbons",len(C),"bonds",len(bonds))

    # ---- on-surface regularisation ----
    nadj=defaultdict(list)
    for a,b in bonds: nadj[a].append(b); nadj[b].append(a)
    for _ in range(60):
        newC=C.copy()
        for i in range(len(C)):
            nb=nadj[i]
            if not nb: continue
            newC[i]=C[nb].mean(0)
        dirs=newC-cm; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
        C=raycast_to_surface(cm,dirs,Vm,Fm)

    # ---- validate geometry ----
    bl=np.array([np.linalg.norm(C[a]-C[b]) for a,b in bonds])
    deg=np.zeros(len(C),int)
    for a,b in bonds: deg[a]+=1; deg[b]+=1
    # surface fit: distance of each carbon to nearest mesh vertex (atoms are on surface)
    from scipy.spatial import cKDTree
    tree=cKDTree(Vm); dsurf,_=tree.query(C)
    P=C-cm; u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False); tax=P@vt[0]
    print("\n==== FIT RESULT (C%d) ===="%len(C))
    print("atoms deg3: %d/%d"%(int((deg==3).sum()),len(C)))
    print("bonds: %d  length mean %.2f  CV %.2f%%  (min %.1f max %.1f)"%(
        len(bonds),bl.mean(),bl.std()/bl.mean()*100,bl.min(),bl.max()))
    dd,_=degrees(F,n)
    print("faces: pentagons=%d hexagons=%d (others=%d)"%(
        int((dd==5).sum()),int((dd==6).sum()),int(((dd!=5)&(dd!=6)).sum())))
    print("surface-fit: atom->mesh dist mean %.2f max %.2f (mesh radius ~%.0f => %.2f%% of R)"%(
        dsurf.mean(),dsurf.max(),np.linalg.norm(Vm-cm,axis=1).mean(),
        dsurf.mean()/np.linalg.norm(Vm-cm,axis=1).mean()*100))
    print("shape: axial extent %.0f  PCA ratios"%(tax.max()-tax.min()),(s/s.max()).round(3))

    # ---- export ----
    with open(OUT_XYZ,'w') as f:
        f.write("%d\nC480 fullerene fitted to mesh4a (dual of capsomer triangulation)\n"%len(C))
        for p in C: f.write("C %.4f %.4f %.4f\n"%(p[0],p[1],p[2]))
    with open(OUT_EDG,'w') as f:
        for a,b in bonds: f.write("%d %d\n"%(a,b))
    # pentagon face membership for viz
    np.save('/tmp/fit_carbons.npy',C)
    np.savez('/tmp/fit_result.npz',C=C,bonds=np.array(bonds),tri_deg=dd,Ftri=Ftri)
    print("wrote",OUT_XYZ)

if __name__=="__main__":
    main()
