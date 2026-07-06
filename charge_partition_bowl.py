#!/usr/bin/env python3
"""Bowl-anchored placement (revision of charge_partition.py).

Same topology-safe skeleton -- 12 equal pi/3 curvature basins by Gauss-Bonnet --
but the 12 pentamer SITES are placed at each basin's *bowl peak* (the strongest
elliptic point, both principal curvatures positive), not the curvature-mass
centroid.  Saddle curvature is clamped to zero ("flattened") per the requirement
that pentamers sit on genuine bowls.  This re-anchors the loosely placed mid-body
pentamers onto real bowls so the search selects a tiling whose disclinations land
on the shape-defining curvature maxima."""
import numpy as np, heapq
from collections import defaultdict

m=np.load("/tmp/mesh4a.npz"); V,F,cm=m['V'],m['F'],m['c']; nv=len(V)
c=np.load("/tmp/curv.npz"); bowl=c['bowl']            # min principal curvature (>0 == bowl)
kappa=c['Kint']                                       # integrated Gaussian curvature (angle deficit)
bpos=np.clip(bowl,0,None)                             # bowl strength, saddles flattened to 0
print("Σκ = %.4f (=%.2f pentamer-quanta); bowl verts %.0f%%"%(
    kappa.sum(),kappa.sum()/(np.pi/3),100*np.mean(bowl>0)))

# mesh graph for geodesic growth
adj=defaultdict(list)
for f in F:
    for i in range(3):
        a,b=f[i],f[(i+1)%3]; d=np.linalg.norm(V[a]-V[b]); adj[a].append((b,d)); adj[b].append((a,d))

# 12 seeds: strongest bowls, well separated
QUANTUM=np.pi/3
area=sum(0.5*np.linalg.norm(np.cross(V[f[1]]-V[f[0]],V[f[2]]-V[f[0]])) for f in F)
sep=0.6*np.sqrt(area/242)*np.sqrt(2/np.sqrt(3))*2
order=np.argsort(-bpos)
seeds=[]
for idx in order:
    if bpos[idx]<=0: break
    if all(np.linalg.norm(V[idx]-V[s])>sep for s in seeds): seeds.append(int(idx))
    if len(seeds)==12: break
# fall back to sharpest angle-deficit if fewer than 12 strong bowls separate out
if len(seeds)<12:
    for idx in np.argsort(-kappa):
        if all(np.linalg.norm(V[idx]-V[s])>sep for s in seeds): seeds.append(int(idx))
        if len(seeds)==12: break

# charge-quantized (pi/3) multi-source region growing -- unchanged, topology-safe
region=np.full(nv,-1); charge=np.zeros(12); pq=[]
for r,s in enumerate(seeds): heapq.heappush(pq,(0.0,s,r))
while pq:
    d,u,r=heapq.heappop(pq)
    if region[u]!=-1 or charge[r]>=QUANTUM: continue
    region[u]=r; charge[r]+=kappa[u]
    for (w,dw) in adj[u]:
        if region[w]==-1: heapq.heappush(pq,(d+dw,w,r))
if (region==-1).any():
    pq=[]
    for u in range(nv):
        if region[u]!=-1:
            for (w,dw) in adj[u]:
                if region[w]==-1: heapq.heappush(pq,(dw,w,region[u]))
    while pq:
        d,u,r=heapq.heappop(pq)
        if region[u]!=-1: continue
        region[u]=r; charge[r]+=kappa[u]
        for (w,dw) in adj[u]:
            if region[w]==-1: heapq.heappush(pq,(d+dw,w,r))

# sites = bowl-peak of each basin (bowl-strength^2 weighted centroid, snapped to vertex)
sites=np.zeros((12,3))
for r in range(12):
    idx=np.where(region==r)[0]; w=bpos[idx]**2
    if w.sum()<=0: w=np.ones(len(idx))
    cen=(V[idx]*w[:,None]).sum(0)/w.sum()
    sites[r]=V[idx][np.argmin(np.linalg.norm(V[idx]-cen,axis=1))]

P=V-cm; u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False); axis=vt[0]
t=(sites-cm)@axis; span=max(abs(t.min()),abs(t.max()))
print("region charges (π/3 units): mean %.3f std %.3f min %.3f max %.3f"%(
    (charge/QUANTUM).mean(),(charge/QUANTUM).std(),(charge/QUANTUM).min(),(charge/QUANTUM).max()))
print("all 12 sites on bowls: %s"%bool((bowl[[np.argmin(np.linalg.norm(V-x,axis=1)) for x in sites]]>0).all()))
print("axial positions:", np.sort(t/span).round(2).tolist(),"| split %d/%d"%((t<0).sum(),(t>=0).sum()))
np.savez("/tmp/pentamers.npz",sites=sites,region=region,kappa=kappa,charge=charge,axis=axis,cm=cm)
print("saved /tmp/pentamers.npz (bowl-anchored)")
