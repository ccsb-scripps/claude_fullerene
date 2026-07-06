#!/usr/bin/env python3
"""Step 1 of the corrected pipeline: place 12 pentamers by partitioning the
mesh's Gaussian curvature into 12 equal 60-degree (pi/3) quanta.

Discrete Gaussian curvature at a vertex = angular deficit
    kappa_v = 2*pi - sum(incident triangle corner angles).
By Gauss-Bonnet, sum_v kappa_v = 4*pi = 12 * (pi/3) for a genus-0 surface.
We grow 12 geodesic regions from the sharpest points, each accumulating pi/3 of
curvature; the region centroids are the pentamer sites."""
import numpy as np, heapq
from collections import defaultdict

m=np.load("/tmp/mesh4a.npz"); V,F,cm=m['V'],m['F'],m['c']
nv=len(V)

# --- per-vertex Gaussian curvature (angle deficit) ---
kappa=np.full(nv, 2*np.pi)
for f in F:
    for i in range(3):
        a=V[f[i]]; b=V[f[(i+1)%3]]; c=V[f[(i+2)%3]]
        u=b-a; w=c-a
        ang=np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)+1e-12),-1,1))
        kappa[f[i]]-=ang
print("total curvature Σκ = %.4f  (4π = %.4f, i.e. %.2f pentamer-quanta)"%(
    kappa.sum(), 4*np.pi, kappa.sum()/(np.pi/3)))

# --- mesh graph (edge lengths) for geodesic growth ---
adj=defaultdict(list)
for f in F:
    for i in range(3):
        a,b=f[i],f[(i+1)%3]
        d=np.linalg.norm(V[a]-V[b]); adj[a].append((b,d)); adj[b].append((a,d))

# --- 12 seeds: sharpest, well separated ---
QUANTUM=np.pi/3
order=np.argsort(-kappa)
# rough spacing from area & count
area=0.0
for f in F:
    area+=0.5*np.linalg.norm(np.cross(V[f[1]]-V[f[0]],V[f[2]]-V[f[0]]))
sep=0.6*np.sqrt(area/242)*np.sqrt(2/np.sqrt(3))*2  # ~ pentamer spacing scale
seeds=[]
for idx in order:
    if all(np.linalg.norm(V[idx]-V[s])>sep for s in seeds): seeds.append(int(idx))
    if len(seeds)==12: break

# --- charge-quantized multi-source region growing ---
# each vertex assigned to a region; region stops claiming once its charge >= pi/3
region=np.full(nv,-1)
charge=np.zeros(12)
pq=[]  # (geodesic_dist, vertex, region)
for r,s in enumerate(seeds): heapq.heappush(pq,(0.0,s,r))
while pq:
    d,u,r=heapq.heappop(pq)
    if region[u]!=-1: continue
    if charge[r]>=QUANTUM: continue          # this region is full
    region[u]=r; charge[r]+=kappa[u]
    for (w,dw) in adj[u]:
        if region[w]==-1: heapq.heappush(pq,(d+dw,w,r))
# assign any leftover vertices to nearest region by curvature-free growth
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

# --- pentamer sites = curvature-weighted centroids of each region ---
sites=np.zeros((12,3))
for r in range(12):
    mask=region==r; wsum=np.clip(kappa[mask],0,None)
    if wsum.sum()<=0: wsum=np.ones(mask.sum())
    sites[r]=(V[mask]*wsum[:,None]).sum(0)/wsum.sum()
    sites[r]=V[mask][np.argmin(np.linalg.norm(V[mask]-sites[r],axis=1))]  # snap to a vertex

# --- report ---
P=V-cm; u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False); axis=vt[0]
t=(sites-cm)@axis; span=max(abs(t.min()),abs(t.max()))
print("\nregion charges (units of π/3, target 1.00):")
print("  ",(charge/QUANTUM).round(3).tolist())
print("  mean %.3f  std %.3f  min %.3f  max %.3f"%(
    (charge/QUANTUM).mean(),(charge/QUANTUM).std(),(charge/QUANTUM).min(),(charge/QUANTUM).max()))
print("\npentamer axial positions (normalized -1..+1):", np.sort(t/span).round(2).tolist())
print("split %d / %d across mid-plane"%((t<0).sum(),(t>=0).sum()))
np.savez("/tmp/pentamers.npz",sites=sites,region=region,kappa=kappa,charge=charge,axis=axis,cm=cm)
print("\nsaved /tmp/pentamers.npz")
