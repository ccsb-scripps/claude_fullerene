#!/usr/bin/env python3
"""Relax the C238 cage to (near-)regular tiles with a bond+angle force field,
fit to mesh4a (7-DOF: rotation+translation+uniform scale), export for viewer."""
import numpy as np, json
from scipy.spatial import cKDTree
from collections import defaultdict

d=np.load("/tmp/c238.npz",allow_pickle=True)
X=d['carbons'].astype(float); bonds=[tuple(int(x) for x in b) for b in d['bonds']]
pent=[list(map(int,f)) for f in d['pent']]; hexf=[list(map(int,f)) for f in d['hex']]
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']; cmm=m['c']
n=len(X)
nbr=defaultdict(list)
for a,b in bonds: nbr[a].append(b); nbr[b].append(a)
L0=np.median([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
angles=[]
for i in range(n):
    nb=nbr[i]
    for p in range(len(nb)):
        for q in range(p+1,len(nb)): angles.append((nb[p],i,nb[q]))
th0=np.radians(120.0)
def relax(X,iters,kb=1.0,ka=1.0,step=0.05):
    for _ in range(iters):
        Fc=np.zeros_like(X)
        for a,b in bonds:
            e=X[a]-X[b]; L=np.linalg.norm(e)+1e-12; f=kb*(L-L0)*e/L; Fc[a]-=f; Fc[b]+=f
        for (j,i,k) in angles:
            rij=X[j]-X[i]; rik=X[k]-X[i]; lij=np.linalg.norm(rij)+1e-12; lik=np.linalg.norm(rik)+1e-12
            c=np.clip(rij@rik/(lij*lik),-1,1); th=np.arccos(c); ss=max(np.sin(th),1e-6); dE=ka*(th-th0)
            dcj=rik/(lij*lik)-c*rij/lij**2; dck=rij/(lij*lik)-c*rik/lik**2
            gj=dE/ss*dcj; gk=dE/ss*dck; Fc[j]+=gj; Fc[k]+=gk; Fc[i]-=(gj+gk)
        X=X+step*Fc
    return X
X=relax(X,4000)
bl=np.array([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
def fstats(fs,ideal):
    ad=[]
    for f in fs:
        P=X[f];k=len(f)
        for i in range(k):
            a=P[(i-1)%k]-P[i];b=P[(i+1)%k]-P[i]
            ad.append(abs(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))-ideal))
    return np.array(ad)
ha=fstats(hexf,120); pa=fstats(pent,108)
print("C238 relaxed: bonds CV %.2f%%  hex angleDev %.1f  pent angleDev %.1f  hex within5deg %.0f%%"%(
    bl.std()/bl.mean()*100, ha.mean(), pa.mean(), np.mean(ha<5)*100))
# 7-DOF fit to mesh
def pca(P): c=P.mean(0);u,s,vt=np.linalg.svd(P-c,full_matrices=False);return c,vt,s
cX,vtX,sX=pca(X); cM,vtM,sM=pca(Vm)
Xc=(X-cX)@vtX.T; Mc=(Vm-cM)@vtM.T
for k in range(3):
    if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
uni=(Mc.max(0)-Mc.min(0)).mean()/(Xc.max(0)-Xc.min(0)).mean()
Xw=(Xc*uni)@vtM+cM
dS,_=cKDTree(Vm).query(Xw); Rm=np.linalg.norm(Vm-cM,axis=1).mean()
print("fit: relaxed aspect %s (mesh %s)  surf-dist mean %.1f%% max %.1f%% of R"%(
    (sX/sX.max()).round(3),(sM/sM.max()).round(3),dS.mean()/Rm*100,dS.max()/Rm*100))
# export viewer (aligned regular cage + mesh)
out={"meshV":Vm.tolist(),"meshF":Fm.tolist(),"atoms":Xw.tolist(),"atomsIdeal":Xw.tolist(),
     "bonds":[[int(a),int(b)] for a,b in bonds],
     "pent":[[int(i) for i in f] for f in pent],"hex":[[int(i) for i in f] for f in hexf],
     "center":Xw.mean(0).tolist(),"scale":float(np.abs(Xw-Xw.mean(0)).max())}
json.dump(out,open("/Users/olson/mytestapp/viewer_data.json","w"))
with open("/Users/olson/Dev/fullerenes/mesh4a_C238_regular.xyz","w") as f:
    f.write("%d\nC238 regular fullerene (field-aligned tiling of mesh4a)\n"%n)
    for p in X: f.write("C %.4f %.4f %.4f\n"%(p[0],p[1],p[2]))
np.savez("/tmp/c238_final.npz",Xw=Xw,X=X,bonds=np.array(bonds))
print("exported viewer_data.json + mesh4a_C238_regular.xyz")
