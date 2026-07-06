#!/usr/bin/env python3
"""Step 1a -- per-vertex curvature for bowl placement.

Computes discrete Gaussian curvature (angle deficit / mixed area), mean curvature
(cotangent Laplacian, signed by the outward normal), the two principal curvatures
k1,k2, and the "bowl" scalar min(k1,k2) (>0 iff the vertex is an elliptic dome).
Writes /tmp/curv.npz (K, bowl, k1, k2, Hs, area, Kint) for charge_partition_bowl.py.

Reference: Meyer, Desbrun, Schroeder, Barr, "Discrete Differential-Geometry
Operators for Triangulated 2-Manifolds", VisMath 2003.
"""
import numpy as np
m = np.load("/tmp/mesh4a.npz"); V, F = m['V'], m['F']; nv = len(V)

Kint = np.full(nv, 2*np.pi)      # integrated Gaussian curvature (angle deficit)
area = np.zeros(nv)              # barycentric vertex area
Hvec = np.zeros((nv, 3))         # cotangent-Laplacian mean-curvature-normal accumulator
def cot(u, w):
    s = np.linalg.norm(np.cross(u, w)); return (u @ w) / max(s, 1e-12)
for f in F:
    i, j, k = f
    for a, b, c in [(i,j,k), (j,k,i), (k,i,j)]:
        u = V[b]-V[a]; w = V[c]-V[a]
        Kint[a] -= np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)+1e-12), -1, 1))
    Atri = 0.5*np.linalg.norm(np.cross(V[j]-V[i], V[k]-V[i]))
    for a in (i, j, k): area[a] += Atri/3
    for a, b, c in [(i,j,k), (j,k,i), (k,i,j)]:
        cw = cot(V[b]-V[a], V[c]-V[a])          # cotangent at a, opposite edge (b,c)
        Hvec[b] += cw*(V[b]-V[c]); Hvec[c] += cw*(V[c]-V[b])

K = Kint/np.maximum(area, 1e-9)                 # pointwise Gaussian curvature
Hn = Hvec/(2*np.maximum(area, 1e-9)[:, None]); Hmag = 0.5*np.linalg.norm(Hn, axis=1)
c0 = V.mean(0); outn = (V-c0)/np.linalg.norm(V-c0, axis=1, keepdims=True)
Hs = Hmag*np.sign(np.einsum('ij,ij->i', Hn, outn))     # mean curvature, + = convex outward
disc = np.maximum(Hs**2 - K, 0.0)
k1 = Hs + np.sqrt(disc); k2 = Hs - np.sqrt(disc)
bowl = np.minimum(k1, k2)                       # >0  <=>  both principal curvatures > 0

print("verts: bowl(both k>0) %.0f%% | saddle(K<0) %.0f%%" % (100*np.mean(bowl>0), 100*np.mean(K<0)))
np.savez("/tmp/curv.npz", K=K, bowl=bowl, k1=k1, k2=k2, Hs=Hs, area=area, Kint=Kint)
print("saved /tmp/curv.npz")
