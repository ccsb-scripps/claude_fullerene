#!/usr/bin/env python3
"""Step 2a: smooth 6-fold (hexagonal) orientation field on mesh4a with its 12
singularities pinned at the charge-partition pentamer sites (trivial-connections
method, Crane et al. 2010).

Levi-Civita holonomy around a vertex = its angle defect Omega_v. We add a
connection adjustment x (per edge) solving a Poisson system so the adjusted
holonomy is 2*pi/6 at the 12 pentamers and 0 everywhere else. Then integrate the
field over faces and verify the singularities.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from collections import defaultdict, deque

m=np.load("/tmp/mesh4a.npz"); V,F,cm=m['V'],m['F'],m['c']
pk=np.load("/tmp/pentamers.npz"); sites=pk['sites']
nv=len(V); nf=len(F)
from scipy.spatial import cKDTree
pin=[int(cKDTree(V).query(s)[1]) for s in sites]      # pentamer vertices

# --- Gaussian curvature (angle defect) ---
Omega=np.full(nv,2*np.pi)
for f in F:
    for i in range(3):
        a=V[f[i]];b=V[f[(i+1)%3]];c=V[f[(i+2)%3]]
        u=b-a;w=c-a
        Omega[f[i]]-=np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)+1e-12),-1,1))

# --- edge list + vertex-edge incidence ---
edges={};
def eid(a,b):
    k=(min(a,b),max(a,b))
    if k not in edges: edges[k]=len(edges)
    return edges[k]
for f in F:
    for i in range(3): eid(f[i],f[(i+1)%3])
ne=len(edges)
elist=[None]*ne
for (a,b),i in edges.items(): elist[i]=(a,b)

# --- desired holonomy b_v = 2*pi*index_v - Omega_v  (index=1/6 at pentamers) ---
index=np.zeros(nv);
for p in pin: index[p]=1.0/6.0
b=2*np.pi*index-Omega
print("Σ index = %.3f (want 2.0);  Σ b = %.2e (want 0)"%(index.sum(),b.sum()))

# --- graph Laplacian L y = b, x_e = y[head]-y[tail] (least-norm adjustment) ---
rows=[];cols=[];vals=[]
for (a,bb) in elist:
    for (i,j) in [(a,a),(bb,bb)]: rows.append(i);cols.append(j);vals.append(1.0)
    rows+=[a,bb];cols+=[bb,a];vals+=[-1.0,-1.0]
L=sp.csr_matrix((vals,(rows,cols)),shape=(nv,nv))
# minimal-norm solution of the consistent rank-deficient system L y = b
y=spl.lsqr(L,b,atol=1e-12,btol=1e-12,iter_lim=20000)[0]
x=np.array([y[a]-y[t] for (t,a) in [(e[0],e[1]) for e in elist]])  # x_e = y[b]-y[a]

# --- per-face tangent basis ---
e1=np.zeros((nf,3)); e2=np.zeros((nf,3)); nrm=np.zeros((nf,3))
for fi,f in enumerate(F):
    a,bv,c=V[f[0]],V[f[1]],V[f[2]]
    u=bv-a; n=np.cross(bv-a,c-a); n/=np.linalg.norm(n)+1e-12
    u=u/(np.linalg.norm(u)+1e-12); v=np.cross(n,u)
    e1[fi]=u; e2[fi]=v; nrm[fi]=n

# --- dual graph: faces sharing an edge, with transport angle ---
e2f=defaultdict(list)
for fi,f in enumerate(F):
    for i in range(3): e2f[(min(f[i],f[(i+1)%3]),max(f[i],f[(i+1)%3]))].append(fi)
def edge_angle_in_face(fi,a,bv):
    d=V[bv]-V[a]; return np.arctan2(d@e2[fi], d@e1[fi])
# adjacency f->(g, transport, edgeid, sign)
dualadj=defaultdict(list)
for (a,bv),fs in e2f.items():
    if len(fs)!=2: continue
    f,g=fs; ei=edges[(a,bv)]
    af=edge_angle_in_face(f,a,bv); ag=edge_angle_in_face(g,a,bv)
    tr=ag-af                                # parallel transport f->g
    dualadj[f].append((g,tr, x[ei])); dualadj[g].append((f,-tr,-x[ei]))

# --- integrate field over faces (BFS spanning tree) ---
phi=np.full(nf,np.nan); phi[0]=0.0; dq=deque([0])
while dq:
    f=dq.popleft()
    for (g,tr,xe) in dualadj[f]:
        if np.isnan(phi[g]):
            phi[g]=phi[f]+tr+xe; dq.append(g)

# --- verify singularities: holonomy around each vertex from integrated field ---
# holonomy_v = sum of (phi[g]-phi[f]-tr) around v's faces, but simplest check:
# recompute per-vertex adjusted holonomy = Omega_v + sum_{e in v} (oriented x)
hol=Omega.copy()
for ei,(a,bv) in enumerate(elist):
    hol[a]+=x[ei]; hol[bv]-=x[ei]           # d0^T x contribution
hol_mod=hol/(2*np.pi)                        # in turns; expect 1/6 at pins, 0 else
sing=np.where(np.abs(hol_mod)>0.02)[0]
print("vertices with nonzero holonomy: %d"%len(sing))
at_pins=sum(1 for p in pin if abs(hol_mod[p]-1/6)<0.03)
print("pentamer vertices with holonomy≈+1/6: %d/12"%at_pins)
others=[v for v in sing if v not in pin]
print("non-pentamer singular vertices: %d  (want 0)"%len(others))
print("holonomy at pentamers (turns):", [round(float(hol_mod[p]),3) for p in pin])

# field direction per face in 3D (for viz)
fld=np.cos(phi)[:,None]*e1 + np.sin(phi)[:,None]*e2
np.savez("/tmp/orient_field.npz", phi=phi, fld=fld, Fcent=V[F].mean(1),
         pin=np.array(pin), sites=sites, x=x)
print("saved /tmp/orient_field.npz")
