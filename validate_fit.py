#!/usr/bin/env python3
"""Validate the ideal C480 and fit it to mesh4a; produce a mesh-conformed cage."""
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict

def load_xyz(p):
    L=open(p).read().split('\n'); n=int(L[0].split()[0]); X=[]
    for ln in L[2:2+n]:
        t=ln.split(); X.append([float(t[1]),float(t[2]),float(t[3])])
    return np.array(X)

def bonds_of(X,fac=1.35):
    tree=cKDTree(X); d,_=tree.query(X,k=2); L0=np.median(d[:,1])
    pairs=tree.query_pairs(L0*fac)
    return sorted(pairs),L0

def shape(X):
    c=X.mean(0); P=X-c
    u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False)
    ax=P@vt.T; ext=ax.max(0)-ax.min(0)
    r=np.linalg.norm(P,axis=1)
    return c,vt,ext,r,s

def raycast(origin,dirs,Vm,Fm):
    v0=Vm[Fm[:,0]]; e1=Vm[Fm[:,1]]-v0; e2=Vm[Fm[:,2]]-v0; out=np.zeros((len(dirs),3)); eps=1e-6
    for i,dv in enumerate(dirs):
        h=np.cross(dv,e2); a=np.einsum('ij,ij->i',e1,h); m=np.abs(a)>eps
        f=np.zeros_like(a); f[m]=1/a[m]; s=origin-v0
        u=f*np.einsum('ij,ij->i',s,h); q=np.cross(s,e1)
        vv=f*np.einsum('j,ij->i',dv,q); t=f*np.einsum('ij,ij->i',e2,q)
        ok=m&(u>=-1e-4)&(vv>=-1e-4)&(u+vv<=1+1e-4)&(t>1e-3)
        out[i]=origin+(t[ok].max() if ok.any() else np.linalg.norm(Vm-origin,axis=1).mean())*dv
    return out

# ---- load ----
X=load_xyz("/Users/olson/Dev/fullerenes/mesh4a_C480_ideal.xyz")
m=np.load("/tmp/mesh4a.npz"); Vm,Fm,cm=m['V'],m['F'],m['c']
bonds,L0=bonds_of(X)
deg=np.zeros(len(X),int)
for a,b in bonds: deg[a]+=1; deg[b]+=1
bl=np.array([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
print("IDEAL C480: atoms=%d bonds=%d deg3=%d/%d bond mean=%.4f CV=%.2f%%"%(
    len(X),len(bonds),int((deg==3).sum()),len(X),bl.mean(),bl.std()/bl.mean()*100))

cX,vtX,extX,rX,sX=shape(X)
cM,vtM,extM,rM,sM=shape(Vm)
print("ideal shape: extents",extX.round(2),"asph(max/min r)=%.2f PCA ratios"%(rX.max()/rX.min()),(sX/sX.max()).round(3))
print("mesh  shape: extents",extM.round(0),"asph(max/min r)=%.2f PCA ratios"%(rM.max()/rM.min()),(sM/sM.max()).round(3))

# ---- align ideal -> mesh (PCA axes + per-axis scale), fix reflections ----
Xc=(X-cX)@vtX.T                      # ideal in its PCA frame
Mc=(Vm-cM)@vtM.T                     # mesh in its PCA frame
# match sign of each axis by 3rd-moment (skew) so tips point same way
for k in range(3):
    if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
scale=extM/extX
Xa=Xc*scale                          # anisotropic scale to mesh extents
Xw=Xa@vtM + cM                       # back to mesh world frame
# rigid+uniform-scale fit RMS (before conforming)
uni=extM.mean()/extX.mean()
Xu=(Xc*uni)@vtM+cM
tree=cKDTree(Vm)
d_u,_=tree.query(Xu); d_a,_=tree.query(Xw)
print("\nFIT (align only): uniform-scale surf-dist mean=%.1f max=%.1f | anisotropic mean=%.1f max=%.1f (R~%.0f)"%(
    d_u.mean(),d_u.max(),d_a.mean(),d_a.max(),rM.mean()))

# ---- conform: radially project aligned atoms onto mesh surface ----
dirs=Xw-cm; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
Xs=raycast(cm,dirs,Vm,Fm)
# on-surface spring relaxation: equalize bonds while re-projecting to surface
nadj=defaultdict(list)
for a,b in bonds: nadj[a].append(b); nadj[b].append(a)
for _ in range(120):
    new=Xs.copy()
    for i in range(len(Xs)):
        nb=nadj[i]; new[i]=Xs[nb].mean(0)
    dr=new-cm; dr/=np.linalg.norm(dr,axis=1,keepdims=True)
    Xs=raycast(cm,dr,Vm,Fm)
bl2=np.array([np.linalg.norm(Xs[a]-Xs[b]) for a,b in bonds])
ds,_=tree.query(Xs)
cS,vtS,extS,rS,sS=shape(Xs)
print("CONFORMED C480 (atoms on mesh4a): bond mean=%.1f CV=%.2f%% | surf-dist mean=%.2f max=%.2f (%.2f%% of R)"%(
    bl2.mean(),bl2.std()/bl2.mean()*100,ds.mean(),ds.max(),ds.mean()/rM.mean()*100))
print("conformed shape: asph=%.2f PCA ratios"%(rS.max()/rS.min()),(sS/sS.max()).round(3))

# ---- export ----
def save_xyz(path,P,title):
    with open(path,'w') as f:
        f.write("%d\n%s\n"%(len(P),title))
        for p in P: f.write("C %.4f %.4f %.4f\n"%(p[0],p[1],p[2]))
save_xyz("/Users/olson/Dev/fullerenes/mesh4a_C480_conformed.xyz",Xs,
         "C480 valid fullerene (12 pentagons) conformed to mesh4a surface")
np.savez("/tmp/fit_final.npz",Xs=Xs,Xideal_aligned=Xw,bonds=np.array(bonds),cm=cm)
with open("/Users/olson/Dev/fullerenes/mesh4a_C480_edges.txt","w") as f:
    for a,b in bonds: f.write("%d %d\n"%(a,b))
print("\nwrote mesh4a_C480_conformed.xyz and edges.txt")
