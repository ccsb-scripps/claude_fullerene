#!/usr/bin/env python3
"""Relax the geodesic C500 to equilateral (spring embedder on the exact graph),
measure regularity + fit to mesh4a, and export for the viewer."""
import numpy as np, json
from scipy.spatial import cKDTree
from collections import defaultdict

d=np.load("/tmp/geo_result.npz")
C=d['carbons'].astype(float); bonds=[tuple(b) for b in d['bonds']]
geo_faces=[tuple(t) for t in d['geo_faces']]; gdeg=d['gdeg']; cm=d['cm']
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; cmm=m['c']
bw=np.load("/tmp/bowls.npz"); bowls=bw['pos']; hexdia=float(bw['hexdia'])
n=len(C)
nadj=defaultdict(list)
for a,b in bonds: nadj[a].append(b); nadj[b].append(a)

# fullerene faces = dual of geodesic vertices (capsomers)
inc=defaultdict(list)
for ti,f in enumerate(geo_faces):
    for v in f: inc[v].append(ti)
def order_face(v,members):
    P=C[members]; c=P.mean(0); nrm=c-cm; nrm/=np.linalg.norm(nrm)
    t=np.array([1.0,0,0]);
    if abs(nrm[0])>0.9: t=np.array([0,1.0,0])
    e1=t-nrm*np.dot(t,nrm); e1/=np.linalg.norm(e1); e2=np.cross(nrm,e1)
    ang=[np.arctan2((C[mi]-c)@e2,(C[mi]-c)@e1) for mi in members]
    return [members[i] for i in np.argsort(ang)]
faces=[order_face(v,inc[v]) for v in range(len(gdeg))]
pent=[f for f in faces if len(f)==5]; hexf=[f for f in faces if len(f)==6]

# ---- spring relaxation to equilateral (free / intrinsic shape) ----
L0=np.median([np.linalg.norm(C[a]-C[b]) for a,b in bonds])
X=C.copy()
for it in range(800):
    new=X.copy()
    for i in range(n):
        acc=np.zeros(3)
        for j in nadj[i]:
            dv=X[i]-X[j]; L=np.linalg.norm(dv)+1e-9
            acc+=X[j]+dv/L*L0
        new[i]=0.6*X[i]+0.4*(acc/len(nadj[i]))
    X=new
bl=np.array([np.linalg.norm(X[a]-X[b]) for a,b in bonds])

def face_stats(faces,ideal):
    ed=[];ad=[];pl=[]
    for f in faces:
        P=X[f];k=len(f)
        e=[np.linalg.norm(P[i]-P[(i+1)%k]) for i in range(k)]
        ed.append((max(e)-min(e))/np.mean(e))
        for i in range(k):
            a=P[(i-1)%k]-P[i];b=P[(i+1)%k]-P[i]
            ad.append(abs(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))-ideal))
        c=P.mean(0);_,_,vt=np.linalg.svd(P-c);pl.append(np.max(np.abs((P-c)@vt[2]))/np.mean(e))
    return np.array(ed),np.array(ad),np.array(pl)
pe,pa,pp=face_stats(pent,108); he,ha,hp=face_stats(hexf,120)
print("=== equilateral C500 (geodesic, template-free) ===")
print("bonds: mean %.3f CV %.2f%% (min %.3f max %.3f)"%(bl.mean(),bl.std()/bl.mean()*100,bl.min(),bl.max()))
print("pentagons: edge-spread %.1f%% angleDev(108) mean %.1f max %.1f nonplanar %.1f%%"%(pe.mean()*100,pa.mean(),pa.max(),pp.mean()*100))
print("hexagons:  edge-spread %.1f%% angleDev(120) mean %.1f max %.1f nonplanar %.1f%%"%(he.mean()*100,ha.mean(),ha.max(),hp.mean()*100))
print("hexagons within 5deg of 120: %.0f%%"%(np.mean(ha<5)*100))

# ---- fit relaxed cage to mesh (align + uniform scale) ----
def pca(P): c=P.mean(0);u,s,vt=np.linalg.svd(P-c,full_matrices=False);return c,vt,s
cX,vtX,sX=pca(X); cM,vtM,sM=pca(Vm)
Xc=(X-cX)@vtX.T; Mc=(Vm-cM)@vtM.T
for k in range(3):
    if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
extX=Xc.max(0)-Xc.min(0); extM=Mc.max(0)-Mc.min(0)
uni=extM.mean()/extX.mean()
Xw=(Xc*uni)@vtM+cM
d_surf,_=cKDTree(Vm).query(Xw)
Rm=np.linalg.norm(Vm-cM,axis=1).mean()
print("\n=== fit to mesh4a (equilateral, uniform scale) ===")
print("relaxed aspect (PCA ratios):",(sX/sX.max()).round(3),"  mesh:",(sM/sM.max()).round(3))
print("surface distance: mean %.0f max %.0f (mean %.1f%% of R=%.0f)"%(d_surf.mean(),d_surf.max(),d_surf.mean()/Rm*100,Rm))

# ---- export (aligned-to-mesh, on-surface projection for tight viewer) ----
# also snap a copy onto the surface for the "conformed" view
dirs=Xw-cmm; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
# reuse mesh raycast
Fm=m['F']; v0=Vm[Fm[:,0]]; e1=Vm[Fm[:,1]]-v0; e2=Vm[Fm[:,2]]-v0
def project(p):
    dv=p-cmm; dv/=np.linalg.norm(dv)
    h=np.cross(dv,e2); a=np.einsum('ij,ij->i',e1,h); mk=np.abs(a)>1e-6
    f=np.zeros_like(a); f[mk]=1/a[mk]; s=cmm-v0
    u=f*np.einsum('ij,ij->i',s,h); q=np.cross(s,e1)
    vv=f*np.einsum('j,ij->i',dv,q); t=f*np.einsum('ij,ij->i',e2,q)
    ok=mk&(u>=-1e-4)&(vv>=-1e-4)&(u+vv<=1+1e-4)&(t>1e-3)
    return cmm+(t[ok].max() if ok.any() else 1.0)*dv
Xs=np.array([project(p) for p in Xw])
out={"meshV":Vm.tolist(),"meshF":Fm.tolist(),
     "atoms":Xs.tolist(),"atomsIdeal":Xw.tolist(),
     "bonds":[[int(a),int(b)] for a,b in bonds],
     "pent":[[int(i) for i in f] for f in pent],"hex":[[int(i) for i in f] for f in hexf],
     "center":Xs.mean(0).tolist(),"scale":float(np.abs(Xs-Xs.mean(0)).max())}
json.dump(out,open("/Users/olson/mytestapp/viewer_data.json","w"))
with open("/Users/olson/Dev/fullerenes/mesh4a_C500_relaxed.xyz","w") as f:
    f.write("%d\nC500 equilateral geodesic fullerene (12 pentagons at curvature bowls)\n"%n)
    for p in X: f.write("C %.4f %.4f %.4f\n"%(p[0],p[1],p[2]))
print("\nexported viewer_data.json + mesh4a_C500_relaxed.xyz")
