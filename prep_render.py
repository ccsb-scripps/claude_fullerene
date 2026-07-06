#!/usr/bin/env python3
"""Prep for the final render: extract canonical RSPI + spiral path from the C500
representative, export the cage (OBJ), and write render data (atoms, bonds,
curvature peaks, ordered spiral face-centers) for Blender."""
import numpy as np, json
from collections import defaultdict
D=json.load(open("/Users/olson/mytestapp/viewer_data.json"))
atoms=np.array(D['atoms']); bonds=[tuple(b) for b in D['bonds']]
pent=[list(f) for f in D['pent']]; hexf=[list(f) for f in D['hex']]
faces=pent+hexf; F=len(faces)
peaks=np.load("/tmp/pentamers.npz")['sites']

# face adjacency + cyclic neighbor order (faces already have consistent cyclic vertex order)
edge2f=defaultdict(list)
for fi,f in enumerate(faces):
    k=len(f)
    for i in range(k): edge2f[frozenset((f[i],f[(i+1)%k]))].append(fi)
nbr_cyc=[]
for fi,f in enumerate(faces):
    k=len(f); nb=[]
    for i in range(k):
        o=[x for x in edge2f[frozenset((f[i],f[(i+1)%k]))] if x!=fi]; nb.append(o[0] if o else -1)
    nbr_cyc.append(nb)
# canonical RSPI from the Fullerene program (authoritative)
RSPI=[1,14,24,61,63,113,140,224,234,242,244,247]
best=tuple(RSPI)
# spiral PATH for visualization: geometric helix winding pole-to-pole through the
# capsomers, starting at the pentamer nearest one pole.
centers=np.array([atoms[f].mean(0) for f in faces])
c0=centers.mean(0); P=centers-c0
u,sv,vt=np.linalg.svd(P-P.mean(0),full_matrices=False); axis=vt[0]; e1=vt[1]; e2=vt[2]
t=(centers-c0)@axis; tn=(t-t.min())/(t.max()-t.min()+1e-9)
theta=np.arctan2((centers-c0)@e2,(centers-c0)@e1)%(2*np.pi)
# face adjacency graph (faces share a fullerene edge = two consecutive carbons)
fadj=[[x for x in nb if x>=0] for nb in nbr_cyc]
if np.mean([tn[i] for i in range(len(pent))])>0.5: tn=1-tn
start_face=min(range(len(pent)), key=lambda i: tn[i])   # polar pentamer = #1
# BFS shells from the start pentamer -> spiral outward; wind each shell by azimuth
from collections import deque
dist=[-1]*F; dist[start_face]=0; q=deque([start_face])
while q:
    u=q.popleft()
    for v in fadj[u]:
        if dist[v]<0: dist[v]=dist[u]+1; q.append(v)
# order by (shell, azimuth); alternate azimuth direction per shell so shell hand-offs stay local
order=[]
byshell=defaultdict(list)
for f in range(F): byshell[dist[f]].append(f)
prevth=0.0
for sh in sorted(byshell):
    fs=byshell[sh]
    fs.sort(key=lambda f: (theta[f]-prevth)%(2*np.pi))   # continue winding from where last shell ended
    order+=fs
    if fs: prevth=theta[fs[-1]]
S=order
print("canonical RSPI (from Fullerene program):", RSPI)
print("spiral path: geometric helix, %d faces, start=pentamer face %d"%(len(S),start_face))
centers=np.array([atoms[f].mean(0) for f in faces])
spiral_centers=centers[S].tolist()
start_center=centers[S[0]].tolist()   # pentamer #1

# export cage as OBJ (atoms + faces)
with open("/Users/olson/Dev/fullerenes/mesh4a_C500_cage.obj","w") as f:
    f.write("# C500 representative fullerene for mesh4a (12 pentagons + 240 hexagons)\n")
    for p in atoms: f.write("v %.5f %.5f %.5f\n"%(p[0],p[1],p[2]))
    for face in faces: f.write("f "+" ".join(str(i+1) for i in face)+"\n")
print("exported mesh4a_C500_cage.obj (%d atoms, %d faces)"%(len(atoms),F))

json.dump({"meshV":D['meshV'],"meshF":D['meshF'],"atoms":atoms.tolist(),
           "bonds":[[int(a),int(b)] for a,b in bonds],"peaks":peaks.tolist(),
           "spiral":spiral_centers,"start":start_center,
           "rspi":list(best) if best else []}, open("/tmp/render_data.json","w"))
print("wrote /tmp/render_data.json")
