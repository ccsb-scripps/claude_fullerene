#!/usr/bin/env python3
"""Find the 12 best 'bowl' points on mesh4a: local maxima of the minimum
principal curvature with BOTH principal curvatures positive (elliptic domes).
These guide pentamer placement."""
import numpy as np
from collections import defaultdict

m=np.load("/tmp/mesh4a.npz"); V,F,cm=m['V'],m['F'],m['c']
nv=len(V)

# vertex normals (area-weighted), oriented outward
fn=np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]])
VN=np.zeros_like(V)
for i in range(3): np.add.at(VN,F[:,i],fn)
VN/=np.linalg.norm(VN,axis=1,keepdims=True)+1e-12
flip=np.sum(VN*(V-cm),1)<0; VN[flip]*=-1

# adjacency (1-ring) and 2-ring
adj=defaultdict(set)
for f in F:
    for i in range(3):
        adj[f[i]].add(f[(i+1)%3]); adj[f[i]].add(f[(i+2)%3])

k1=np.full(nv,np.nan); k2=np.full(nv,np.nan)
for i in range(nv):
    ring=set(adj[i])
    for j in list(ring): ring|=adj[j]      # 2-ring
    ring.discard(i); nb=list(ring)
    if len(nb)<6: continue
    n=VN[i]
    t=np.array([1.0,0,0]);
    if abs(n[0])>0.9: t=np.array([0,1.0,0])
    e1=t-n*np.dot(t,n); e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
    d=V[nb]-V[i]; x=d@e1; y=d@e2; z=d@n
    # quadric height z = a x^2 + b xy + c y^2 + dx + ey + f
    A=np.column_stack([x*x,x*y,y*y,x,y,np.ones_like(x)])
    coef,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    a,b,c=coef[0],coef[1],coef[2]
    H=np.array([[2*a,b],[b,2*c]])            # Hessian of height
    ev=np.linalg.eigvalsh(H)                 # neighbors dip AWAY from +n on a dome => ev<0
    kk=np.sort(-ev)                          # convex dome -> positive
    k1[i],k2[i]=kk[0],kk[1]                  # k1<=k2, kmin=k1
valid=~np.isnan(k1)
bowl = valid & (k1>0) & (k2>0)               # both principal curvatures positive
print("vertices with both principal curvatures >0 (elliptic domes): %d/%d"%(bowl.sum(),valid.sum()))
print("kmin range on domes: %.5f .. %.5f"%(np.nanmin(k1[bowl]),np.nanmax(k1[bowl])))

# capsomer spacing for a target N (use N~242 as reference; area known)
area=1_891_960
def spacing(N): return np.sqrt(2*area/(N*np.sqrt(3)))
hexdia=spacing(242)
print("hexamer center spacing @N=242: %.1f (mesh units)"%hexdia)

# local maxima of kmin among bowls, well separated (>= 0.8*spacing)
score=np.where(bowl,k1,-1e9)
islocal=np.zeros(nv,bool)
for i in range(nv):
    if not bowl[i]: continue
    if all(score[i]>=score[j] for j in adj[i] if bowl[j]): islocal[i]=True
cand=np.where(islocal)[0]
cand=cand[np.argsort(-score[cand])]
sep=0.8*hexdia
seeds=[]
for idx in cand:
    if all(np.linalg.norm(V[idx]-V[s])>sep for s in seeds): seeds.append(int(idx))
    if len(seeds)==12: break
seeds=np.array(seeds)
print("selected %d bowl seeds (sep>%.0f)"%(len(seeds),sep))

# report positions along principal axis
P=V-cm; u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False); axis=vt[0]
t=(V[seeds]-cm)@axis; span=max(abs(t.min()),abs(t.max()))
order=np.argsort(t)
print("\nbowl  k1(min)   k2(max)   axial(norm)")
for r in order:
    print("  %3d  %.5f  %.5f   %+.2f"%(seeds[r],k1[seeds[r]],k2[seeds[r]],t[r]/span))
print("\naxial split: %d narrow-half / %d broad-half; mean |t|/span=%.2f"%(
    (t<0).sum(),(t>=0).sum(),np.mean(np.abs(t)/span)))
np.savez("/tmp/bowls.npz",seeds=seeds,pos=V[seeds],k1=k1,k2=k2,VN=VN,axis=axis,cm=cm,hexdia=hexdia)
print("saved /tmp/bowls.npz")
