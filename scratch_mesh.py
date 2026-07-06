import numpy as np

def load_ply(path):
    with open(path) as f:
        assert f.readline().strip()=='ply'
        nv=nf=0; line=''
        while True:
            line=f.readline().strip()
            if line.startswith('element vertex'): nv=int(line.split()[-1])
            elif line.startswith('element face'): nf=int(line.split()[-1])
            elif line=='end_header': break
        V=np.array([list(map(float,f.readline().split()[:3])) for _ in range(nv)])
        F=[]
        for _ in range(nf):
            p=f.readline().split()
            k=int(p[0]); F.append([int(x) for x in p[1:1+k]])
    return V,F

V,F=load_ply('clean_0.ply')
print("verts",len(V),"faces",len(F))
tri=all(len(f)==3 for f in F)
print("all triangles?",tri)
Fa=np.array(F)
# edges
from collections import defaultdict
edge_count=defaultdict(int)
for f in F:
    n=len(f)
    for i in range(n):
        a,b=f[i],f[(i+1)%n]
        edge_count[(min(a,b),max(a,b))]+=1
E=len(edge_count)
nonmanifold=sum(1 for c in edge_count.values() if c!=2)
print("E",E,"Euler V-E+F",len(V)-E+len(F),"nonmanifold_edges",nonmanifold)

# shape
c=V.mean(0); P=V-c
# PCA
u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False)
print("PCA singular (axis extents ~):",(s/s.max()).round(3))
r=np.linalg.norm(P,axis=1)
print("radius min/mean/max",r.min().round(1),r.mean().round(1),r.max().round(1),"CV",round(r.std()/r.mean(),3))

# surface area
area=0.0
for f in Fa:
    a,b,cc=V[f[0]],V[f[1]],V[f[2]]
    area+=0.5*np.linalg.norm(np.cross(b-a,cc-a))
# volume (signed tetrahedra from origin at centroid)
vol=0.0
for f in Fa:
    a,b,cc=V[f[0]]-c,V[f[1]]-c,V[f[2]]-c
    vol+=np.dot(a,np.cross(b,cc))/6.0
vol=abs(vol)
print("area %.0f  volume %.0f"%(area,vol))
sph=(np.pi**(1/3)*(6*vol)**(2/3))/area
print("sphericity",round(sph,3))
# axis extents in PCA frame
proj=P@vt.T
ext=proj.max(0)-proj.min(0)
print("axis extents (PCA):",ext.round(0),"ratio",(ext/ext.min()).round(2))
