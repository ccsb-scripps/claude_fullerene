#!/usr/bin/env python3
"""Proper fullerene force-field relaxation (bond-stretch + angle-bend to 120 sp2)
on the exact C500 graph, starting from the non-overlapping on-surface geodesic.
Then measure regularity + fit to mesh4a and export."""
import numpy as np, json
from scipy.spatial import cKDTree
from collections import defaultdict

d=np.load("/tmp/geo_result.npz")
X=d['carbons'].astype(float); bonds=[tuple(b) for b in d['bonds']]
geo_faces=[tuple(t) for t in d['geo_faces']]; gdeg=d['gdeg']; cm=d['cm']
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']; cmm=m['c']
bw=np.load("/tmp/bowls.npz"); bowls=bw['pos']
n=len(X)
nbr=defaultdict(list)
for a,b in bonds: nbr[a].append(b); nbr[b].append(a)
L0=np.median([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
# angle triples at each vertex
angles=[]
for i in range(n):
    nb=nbr[i]
    for p in range(len(nb)):
        for q in range(p+1,len(nb)):
            angles.append((nb[p],i,nb[q]))
th0=np.radians(120.0)

def forces(X,kb=1.0,ka=1.5):
    Fc=np.zeros_like(X)
    for a,b in bonds:
        e=X[a]-X[b]; L=np.linalg.norm(e)+1e-12; f=kb*(L-L0)*e/L
        Fc[a]-=f; Fc[b]+=f
    for (j,i,k) in angles:
        rij=X[j]-X[i]; rik=X[k]-X[i]
        lij=np.linalg.norm(rij)+1e-12; lik=np.linalg.norm(rik)+1e-12
        c=np.clip(np.dot(rij,rik)/(lij*lik),-1,1); th=np.arccos(c); s=max(np.sin(th),1e-6)
        dEdth=ka*(th-th0)
        dcj=rik/(lij*lik)-c*rij/lij**2
        dck=rij/(lij*lik)-c*rik/lik**2
        gj=dEdth/s*dcj; gk=dEdth/s*dck
        Fc[j]+=gj; Fc[k]+=gk; Fc[i]-=(gj+gk)
    return Fc

step=0.05
for it in range(6000):
    Fc=forces(X); X=X+step*Fc
    if it%1500==0:
        bl=np.array([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
        print("it%4d bondCV %.2f%%  Fmax %.4f"%(it,bl.std()/bl.mean()*100,np.abs(Fc).max()))

# ---- fullerene faces (dual of geodesic vertices) ----
inc=defaultdict(list)
for ti,f in enumerate(geo_faces):
    for v in f: inc[v].append(ti)
def order_face(members):
    P=X[members]; c=P.mean(0); nrm=c-X.mean(0); nrm/=np.linalg.norm(nrm)+1e-9
    t=np.array([1.0,0,0]);
    if abs(nrm[0])>0.9: t=np.array([0,1.0,0])
    e1=t-nrm*np.dot(t,nrm); e1/=np.linalg.norm(e1); e2=np.cross(nrm,e1)
    ang=[np.arctan2((P[q]-c)@e2,(P[q]-c)@e1) for q in range(len(members))]
    return [members[q] for q in np.argsort(ang)]
faces=[order_face(inc[v]) for v in range(len(gdeg))]
pent=[f for f in faces if len(f)==5]; hexf=[f for f in faces if len(f)==6]

def fstats(fs,ideal):
    ed=[];ad=[];pl=[]
    for f in fs:
        P=X[f];k=len(f); e=[np.linalg.norm(P[i]-P[(i+1)%k]) for i in range(k)]
        ed.append((max(e)-min(e))/np.mean(e))
        for i in range(k):
            a=P[(i-1)%k]-P[i];b=P[(i+1)%k]-P[i]
            ad.append(abs(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))-ideal))
        c=P.mean(0);_,_,vt=np.linalg.svd(P-c);pl.append(np.max(np.abs((P-c)@vt[2]))/np.mean(e))
    return np.array(ed),np.array(ad),np.array(pl)
bl=np.array([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
pe,pa,pp=fstats(pent,108); he,ha,hp=fstats(hexf,120)
print("\n=== equilateral+equiangular C500 (FF relaxed, template-free) ===")
print("bonds mean %.3f CV %.2f%% (min %.3f max %.3f)"%(bl.mean(),bl.std()/bl.mean()*100,bl.min(),bl.max()))
print("pentagons: edge-spread %.1f%% angleDev(108) mean %.1f max %.1f nonplanar %.1f%%"%(pe.mean()*100,pa.mean(),pa.max(),pp.mean()*100))
print("hexagons:  edge-spread %.1f%% angleDev(120) mean %.1f max %.1f nonplanar %.1f%%"%(he.mean()*100,ha.mean(),ha.max(),hp.mean()*100))
print("hexagons within 5deg of 120: %.0f%%"%(np.mean(ha<5)*100))

# ---- fit to mesh ----
def pca(P): c=P.mean(0);u,s,vt=np.linalg.svd(P-c,full_matrices=False);return c,vt,s
cX,vtX,sX=pca(X); cM,vtM,sM=pca(Vm)
Xc=(X-cX)@vtX.T; Mc=(Vm-cM)@vtM.T
for k in range(3):
    if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
uni=(Mc.max(0)-Mc.min(0)).mean()/(Xc.max(0)-Xc.min(0)).mean()
Xw=(Xc*uni)@vtM+cM
dS,_=cKDTree(Vm).query(Xw); Rm=np.linalg.norm(Vm-cM,axis=1).mean()
print("\n=== fit to mesh4a (equilateral, uniform scale) ===")
print("relaxed aspect",(sX/sX.max()).round(3),"mesh",(sM/sM.max()).round(3))
print("surface dist mean %.0f max %.0f (%.1f%% of R)"%(dS.mean(),dS.max(),dS.mean()/Rm*100))

# ---- export viewer + xyz ----
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
with open("/Users/olson/Dev/fullerenes/mesh4a_C500_relaxed.xyz","w") as f:
    f.write("%d\nC500 equilateral geodesic fullerene, FF relaxed (12 pentagons at bowls)\n"%n)
    for p in X: f.write("C %.4f %.4f %.4f\n"%(p[0],p[1],p[2]))
print("\nexported viewer_data.json + mesh4a_C500_relaxed.xyz")
