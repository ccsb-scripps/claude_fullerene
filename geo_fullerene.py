#!/usr/bin/env python3
"""Template-free fullerene construction from a mesh + its 12 curvature bowls.

Goldberg-Coxeter (geodesic) construction:
  1. 12 bowls -> base icosahedral net (connectivity forced to all-degree-5).
  2. Subdivide each of the 20 triangular patches at frequency nu (class I,
     T=nu^2) into a hexagonal grid; project every vertex onto the mesh surface.
  3. Result = geodesic triangulation: 10T+2 capsomers (12 pentamers at bowls,
     rest hexamers). Its DUAL is the C(20T) fullerene: 12 pentagons + (10T-10)
     hexagons, guaranteed valid by construction (no defect repair needed).
Fully general: works for any genus-0 surface + 12 marked points."""
import numpy as np, sys
from scipy.spatial import ConvexHull
from collections import defaultdict

NU = int(sys.argv[1]) if len(sys.argv)>1 else 5     # frequency; nu=5 -> C500

m=np.load("/tmp/mesh4a.npz"); Vm,Fm,cm=m['V'],m['F'],m['c']
bw=np.load("/tmp/bowls.npz"); bowls=bw['pos']
v0=Vm[Fm[:,0]]; e1=Vm[Fm[:,1]]-v0; e2=Vm[Fm[:,2]]-v0

def raycast1(dv):
    h=np.cross(dv,e2); a=np.einsum('ij,ij->i',e1,h); mk=np.abs(a)>1e-6
    f=np.zeros_like(a); f[mk]=1/a[mk]; s=cm-v0
    u=f*np.einsum('ij,ij->i',s,h); q=np.cross(s,e1)
    vv=f*np.einsum('j,ij->i',dv,q); t=f*np.einsum('ij,ij->i',e2,q)
    ok=mk&(u>=-1e-4)&(vv>=-1e-4)&(u+vv<=1+1e-4)&(t>1e-3)
    return cm+(t[ok].max() if ok.any() else 1.0)*dv
def project(p):
    d=p-cm; d=d/np.linalg.norm(d); return raycast1(d)

# ---- canonical icosahedron connectivity (guaranteed all-degree-5) ----
from scipy.optimize import linear_sum_assignment
def icosahedron():
    z=1/np.sqrt(5); r=2/np.sqrt(5); v=[(0,0,1)]
    for k in range(5): v.append((r*np.cos(np.radians(72*k)),      r*np.sin(np.radians(72*k)),      z))
    for k in range(5): v.append((r*np.cos(np.radians(36+72*k)),   r*np.sin(np.radians(36+72*k)),  -z))
    v.append((0,0,-1))
    faces=[]
    for k in range(5): faces.append((0,1+k,1+(k+1)%5))
    for k in range(5):
        faces.append((1+k,1+(k+1)%5,6+k)); faces.append((1+(k+1)%5,6+(k+1)%5,6+k))
    for k in range(5): faces.append((11,6+(k+1)%5,6+k))
    return np.array(v),faces
ico_v,ico_f=icosahedron()

# spindle axis; rotation mapping icosahedron 5-fold axis (z) onto it
P=Vm-cm; _,_,vt=np.linalg.svd(P-P.mean(0),full_matrices=False); axis=vt[0]
def Rz_to(a):
    z=np.array([0,0,1.0]); a=a/np.linalg.norm(a); vv=np.cross(z,a); s=np.linalg.norm(vv); c=np.dot(z,a)
    if s<1e-9: return np.eye(3) if c>0 else np.diag([1,-1,-1.0])
    vx=np.array([[0,-vv[2],vv[1]],[vv[2],0,-vv[0]],[-vv[1],vv[0],0]])
    return np.eye(3)+vx+vx@vx*((1-c)/s**2)
dirs=bowls-cm; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
best=None
for sgn in (1,-1):
    R0=Rz_to(sgn*axis)
    for az in np.linspace(0,2*np.pi/5,40,endpoint=False):
        ca,sa=np.cos(az),np.sin(az); Rz=np.array([[ca,-sa,0],[sa,ca,0],[0,0,1.0]])
        idir=(R0@Rz@ico_v.T).T
        cost=1-idir@dirs.T                       # 12 ico x 12 bowls
        ri,ci=linear_sum_assignment(cost); tot=cost[ri,ci].sum()
        if best is None or tot<best[0]: best=(tot,ci.copy())
# best[1][ico_vertex] = matched bowl index
assign=best[1]
base_pos=np.zeros((12,3))
for ico_i in range(12): base_pos[ico_i]=bowls[assign[ico_i]]
tris=[list(f) for f in ico_f]
print("base: canonical icosahedron, assignment cost %.3f (each vertex deg 5)"%best[0])

# ---- Goldberg-Coxeter class-I subdivision, project to surface ----
verts=[]; vidx={}
def getv(p):
    s=project(p); key=tuple(np.round(s,1))
    if key not in vidx: vidx[key]=len(verts); verts.append(s)
    return vidx[key]
geo_faces=[]
for (ia,ib,ic) in tris:
    A,B,C=base_pos[ia],base_pos[ib],base_pos[ic]
    g={}
    for i in range(NU+1):
        for j in range(NU+1-i):
            g[(i,j)]=getv(A + (B-A)*i/NU + (C-A)*j/NU)
    for i in range(NU):
        for j in range(NU-i):
            geo_faces.append((g[(i,j)],g[(i+1,j)],g[(i,j+1)]))
            if i+j < NU-1:
                geo_faces.append((g[(i+1,j)],g[(i+1,j+1)],g[(i,j+1)]))
verts=np.array(verts)
# capsomer valences
gdeg=np.zeros(len(verts),int)
for t in geo_faces:
    for v in t: gdeg[v]+=1
T=NU*NU
print("geodesic: capsomers=%d (want %d)  faces=%d (want %d)  deg5=%d deg6=%d others=%d"%(
    len(verts),10*T+2,len(geo_faces),20*T,(gdeg==5).sum(),(gdeg==6).sum(),((gdeg!=5)&(gdeg!=6)).sum()))

# ---- DUAL = carbon fullerene: carbon per geodesic face, bonds share edge ----
cent=np.array([verts[list(t)].mean(0) for t in geo_faces])
carbons=np.array([project(c) for c in cent])
em=defaultdict(list)
for ti,t in enumerate(geo_faces):
    for i in range(3):
        a,b=t[i],t[(i+1)%3]; em[(min(a,b),max(a,b))].append(ti)
bonds=sorted({(min(x),max(x)) for x in em.values() if len(x)==2})
cdeg=np.zeros(len(carbons),int)
for a,b in bonds: cdeg[a]+=1; cdeg[b]+=1
print("fullerene: carbons=%d (want %d)  bonds=%d  deg3=%d/%d"%(
    len(carbons),20*T,len(bonds),(cdeg==3).sum(),len(carbons)))

np.savez("/tmp/geo_result.npz",carbons=carbons,bonds=np.array(bonds),
         capsomers=verts,geo_faces=np.array(geo_faces),gdeg=gdeg,bowls=bowls,cm=cm)
# write coords for the Fullerene program (ICart=1 reads cartesian, builds+relaxes)
with open("/Users/olson/Dev/fullerenes/mesh4a_C%d_geo.xyz"%(20*T),"w") as f:
    f.write("%d\nC%d geodesic fullerene on mesh4a (12 pentagons at curvature bowls)\n"%(len(carbons),20*T))
    for p in carbons: f.write("C %.4f %.4f %.4f\n"%(p[0],p[1],p[2]))
print("wrote mesh4a_C%d_geo.xyz"%(20*T))
