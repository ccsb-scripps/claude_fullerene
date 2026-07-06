import numpy as np
from scipy.spatial import cKDTree
def load_xyz(p):
    L=open(p).read().split('\n'); n=int(L[0].split()[0])
    X=[]
    for line in L[2:2+n]:
        t=line.split(); X.append([float(t[1]),float(t[2]),float(t[3])])
    return np.array(X)
def characterize(X,name):
    n=len(X)
    # bonds: nearest ~3 neighbors within 1.3x min spacing
    tree=cKDTree(X); d,_=tree.query(X,k=2); nn=d[:,1]; bond=np.median(nn)
    pairs=tree.query_pairs(bond*1.3); bl=[np.linalg.norm(X[i]-X[j]) for i,j in pairs]
    bl=np.array(bl)
    deg=np.zeros(n,int)
    for i,j in pairs: deg[i]+=1; deg[j]+=1
    c=X.mean(0); r=np.linalg.norm(X-c,axis=1)
    P=X-c; u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False)
    ext=(P@vt.T); ax=ext.max(0)-ext.min(0); ax=ax/ax.min()
    print(f"{name}: N={n} bonds={len(bl)} deg3={np.sum(deg==3)}/{n} "
          f"bondCV={bl.std()/bl.mean():.3%} radiusCV={r.std()/r.mean():.3%} axis={ax.round(2)}")
    return deg
dref=characterize(load_xyz('fullerene/input/cage_0.xyz'),"reference cage_0")
dnew=characterize(load_xyz('fullerene/Fullerene-3D.xyz'),"our Fullerene-3D")
