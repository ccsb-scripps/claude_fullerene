#!/usr/bin/env python3
"""Polish the best spindle candidate: extra FF relaxation, face extraction,
regularity + fit metrics, and export for the viewer."""
import numpy as np, json
from scipy.spatial import cKDTree
from collections import defaultdict
from cap_tube_sweep import ff_relax

d=np.load("/tmp/best_spindle.npz")
X=d['X'].astype(float); bonds=[tuple(b) for b in d['bonds']]
healed=[tuple(t) for t in d['healed']]; pts=d['pts']; C=int(d['C']); W=int(d['W']); Lt=int(d['Lt'])
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']; cmm=m['c']

X=ff_relax(X,bonds,iters=6000,ka=1.2)          # extra relaxation
bl=np.array([np.linalg.norm(X[a]-X[b]) for a,b in bonds])

# faces = dual of capsomers (triangulation vertices)
inc=defaultdict(list)
for ti,t in enumerate(healed):
    for v in t: inc[v].append(ti)
cm0=X.mean(0)
def order_face(members):
    P=X[members]; c=P.mean(0); nrm=c-cm0; nrm/=np.linalg.norm(nrm)+1e-9
    t=np.array([1.0,0,0]);
    if abs(nrm[0])>0.9: t=np.array([0,1.0,0])
    e1=t-nrm*np.dot(t,nrm); e1/=np.linalg.norm(e1); e2=np.cross(nrm,e1)
    ang=[np.arctan2((P[q]-c)@e2,(P[q]-c)@e1) for q in range(len(members))]
    return [members[q] for q in np.argsort(ang)]
faces=[order_face(inc[v]) for v in inc]
pent=[f for f in faces if len(f)==5]; hexf=[f for f in faces if len(f)==6]

def fstats(fs,ideal):
    ad=[]
    for f in fs:
        P=X[f];k=len(f)
        for i in range(k):
            a=P[(i-1)%k]-P[i];b=P[(i+1)%k]-P[i]
            ad.append(abs(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))-ideal))
    return np.array(ad)
ha=fstats(hexf,120); pa=fstats(pent,108)
print("C%d spindle (W%d,Lt%d): pentagons=%d hexagons=%d"%(C,W,Lt,len(pent),len(hexf)))
print("bonds CV %.2f%%  hex angleDev %.1f  pent angleDev %.1f  hex within5deg %.0f%%"%(
    bl.std()/bl.mean()*100, ha.mean(), pa.mean(), np.mean(ha<5)*100))

# fit
def pca(P): c=P.mean(0);u,s,vt=np.linalg.svd(P-c,full_matrices=False);return c,vt,s
cX,vtX,sX=pca(X); cM,vtM,sM=pca(Vm)
Xc=(X-cX)@vtX.T; Mc=(Vm-cM)@vtM.T
for k in range(3):
    if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
uni=(Mc.max(0)-Mc.min(0)).mean()/(Xc.max(0)-Xc.min(0)).mean()
Xw=(Xc*uni)@vtM+cM
dS,_=cKDTree(Vm).query(Xw); Rm=np.linalg.norm(Vm-cM,axis=1).mean()
print("relaxed aspect %s (mesh %s)  surface fit mean %.1f%% max %.1f%% of R"%(
    (sX/sX.max()).round(3),(sM/sM.max()).round(3),dS.mean()/Rm*100,dS.max()/Rm*100))

# export viewer (project onto mesh for display) + xyz
v0=Vm[Fm[:,0]]; e1=Vm[Fm[:,1]]-v0; e2=Vm[Fm[:,2]]-v0
def project(p):
    dv=p-cmm; dv/=np.linalg.norm(dv); h=np.cross(dv,e2); a=np.einsum('ij,ij->i',e1,h); mk=np.abs(a)>1e-6
    f=np.zeros_like(a); f[mk]=1/a[mk]; s=cmm-v0
    u=f*np.einsum('ij,ij->i',s,h); q=np.cross(s,e1); vv=f*np.einsum('j,ij->i',dv,q); t=f*np.einsum('ij,ij->i',e2,q)
    ok=mk&(u>=-1e-4)&(vv>=-1e-4)&(u+vv<=1+1e-4)&(t>1e-3); return cmm+(t[ok].max() if ok.any() else 1.0)*dv
Xs=np.array([project(p) for p in Xw])
out={"meshV":Vm.tolist(),"meshF":Fm.tolist(),"atoms":Xs.tolist(),"atomsIdeal":Xw.tolist(),
     "bonds":[[int(a),int(b)] for a,b in bonds],
     "pent":[[int(i) for i in f] for f in pent],"hex":[[int(i) for i in f] for f in hexf],
     "center":Xs.mean(0).tolist(),"scale":float(np.abs(Xs-Xs.mean(0)).max())}
json.dump(out,open("/Users/olson/mytestapp/viewer_data.json","w"))
with open("/Users/olson/Dev/fullerenes/mesh4a_C%d_spindle.xyz"%C,"w") as f:
    f.write("%d\nC%d spindle fullerene fitted to mesh4a (aspect-matched capped tube)\n"%(C,C))
    for p in X: f.write("C %.4f %.4f %.4f\n"%(p[0],p[1],p[2]))
print("exported viewer_data.json + mesh4a_C%d_spindle.xyz"%C)
