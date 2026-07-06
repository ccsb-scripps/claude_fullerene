import numpy as np
def load_ply(path):
    with open(path) as f:
        f.readline(); nv=nf=0
        while True:
            line=f.readline().strip()
            if line.startswith('element vertex'): nv=int(line.split()[-1])
            elif line.startswith('element face'): nf=int(line.split()[-1])
            elif line=='end_header': break
        V=np.array([list(map(float,f.readline().split()[:3])) for _ in range(nv)])
        F=np.array([[int(x) for x in f.readline().split()[1:4]] for _ in range(nf)])
    return V,F
V,F=load_ply('clean_0.ply')
fn=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
VN=np.zeros_like(V)
for i in range(3): np.add.at(VN,F[:,i],fn)
VN/=np.linalg.norm(VN,axis=1,keepdims=True)+1e-12
c=V.mean(0)
VN[np.sum(VN*(V-c),1)<0]*=-1
P=V-c; u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False)
np.savez('mesh_clean0.npz',V=V,F=F,VN=VN,centroid=c,axis=vt[0])
print("saved mesh_clean0.npz  V",len(V),"F",len(F))
