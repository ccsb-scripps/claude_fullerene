#!/usr/bin/env python3
"""Systematic fit-landscape scan over H (hexamer count). For each H: run Instant
Meshes at the corresponding face count, build the dual cage, relax to regular
tiles, and score the fit to mesh4a. (Healing to exactly-12 is only needed for the
final chosen H, so the landscape itself is cheap.) Reports fit vs H + timings."""
import numpy as np, subprocess, time, sys
from scipy.spatial import cKDTree
from collections import defaultdict

IMBIN="/Users/olson/Dev/fullerenes/instant-meshes/InstantMeshesBatch"
MESH="/tmp/mesh4a.obj"
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; cM,vtM,sM=(lambda P:(P.mean(0),np.linalg.svd(P-P.mean(0),full_matrices=False)[2],np.linalg.svd(P-P.mean(0),full_matrices=False)[1]))(Vm)
Rm=np.linalg.norm(Vm-cM,axis=1).mean(); Mc=(Vm-cM)@vtM.T; tree=cKDTree(Vm)
pk=np.load("/tmp/pentamers.npz"); PEAKS=pk['sites']
_Fm=m['F']; _ar=0.5*np.linalg.norm(np.cross(Vm[_Fm[:,1]]-Vm[_Fm[:,0]],Vm[_Fm[:,2]]-Vm[_Fm[:,0]]),axis=1).sum()
HEXDIA=float(np.sqrt(2*_ar/(242*np.sqrt(3))))   # reference hexamer center spacing (~95)

def run_im(faces,out):
    subprocess.run([IMBIN,MESH,"-o",out,"-r","6","-p","6","-f",str(faces)],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def load_obj(fn):
    V=[];T=[]
    for ln in open(fn):
        p=ln.split()
        if not p: continue
        if p[0]=='v': V.append([float(x) for x in p[1:4]])
        elif p[0]=='f': T.append([int(x.split('/')[0])-1 for x in p[1:]])
    return np.array(V),[tuple(t) for t in T]

def process(V,tris):
    n=len(V); deg=np.zeros(n,int)
    for t in tris:
        for v in t: deg[v]+=1
    defects=int((deg<5).sum()+(deg>6).sum())
    # pentamer placement error: each curvature peak -> nearest deg-5 capsomer (on mesh), in hexamer-diameters
    d5=V[np.where(deg==5)[0]]
    if len(d5): pe=float(np.linalg.norm(PEAKS[:,None,:]-d5[None,:,:],axis=2).min(axis=1).mean())/HEXDIA
    else: pe=float('nan')
    e2t=defaultdict(list)
    for ti,t in enumerate(tris):
        for i in range(3):
            a,b=t[i],t[(i+1)%3]; e2t[(min(a,b),max(a,b))].append(ti)
    bonds=np.array(sorted({(min(x),max(x)) for x in e2t.values() if len(x)==2}))
    X=np.array([V[list(t)].mean(0) for t in tris])
    # angles
    nbr=defaultdict(list)
    for a,b in bonds: nbr[a].append(b); nbr[b].append(a)
    angs=[]
    for i in range(len(X)):
        nb=nbr[i]
        for p_ in range(len(nb)):
            for q in range(p_+1,len(nb)): angs.append((nb[p_],i,nb[q]))
    A=np.array(angs); th0=np.radians(120.0); L0=np.median(np.linalg.norm(X[bonds[:,0]]-X[bonds[:,1]],axis=1))
    for it in range(1500):
        Fc=np.zeros_like(X)
        e=X[bonds[:,0]]-X[bonds[:,1]]; L=np.linalg.norm(e,axis=1,keepdims=True); f=(L-L0)/L*e
        np.add.at(Fc,bonds[:,0],-f); np.add.at(Fc,bonds[:,1],f)
        rij=X[A[:,0]]-X[A[:,1]]; rik=X[A[:,2]]-X[A[:,1]]
        lij=np.linalg.norm(rij,axis=1,keepdims=True); lik=np.linalg.norm(rik,axis=1,keepdims=True)
        c=np.clip(np.sum(rij*rik,1,keepdims=True)/(lij*lik),-1,1); th=np.arccos(c); ss=np.maximum(np.sin(th),1e-6)
        dE=(th-th0); dcj=rik/(lij*lik)-c*rij/lij**2; dck=rij/(lij*lik)-c*rik/lik**2
        gj=dE/ss*dcj; gk=dE/ss*dck
        np.add.at(Fc,A[:,0],gj); np.add.at(Fc,A[:,2],gk); np.add.at(Fc,A[:,1],-(gj+gk))
        X=X+0.05*Fc
    # 7-DOF fit
    cX=X.mean(0); u,s,vt=np.linalg.svd(X-cX,full_matrices=False); Xc=(X-cX)@vt.T
    for k in range(3):
        if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
    uni=(Mc.max(0)-Mc.min(0)).mean()/(Xc.max(0)-Xc.min(0)).mean()
    Xw=(Xc*uni)@vtM+cM
    dS,_=tree.query(Xw)              # cage -> mesh
    dM,_=cKDTree(Xw).query(Vm)       # mesh -> cage
    haus=max(np.percentile(dS,95),np.percentile(dM,95))/Rm*100   # robust two-sided Hausdorff (95pct)
    return dict(C=len(X),defects=defects,fit_mean=dS.mean()/Rm*100,fit_max=dS.max()/Rm*100,
                aspect=(s/s.max()).round(3),pent_err=pe,haus=haus)

Hgrid=[int(x) for x in sys.argv[1:]] or [220,226,232,238,244,250]
K=1
print("H   -f   C     defc  fit_mean%  Hausdorff%  pent_err(hexdia)")
t0=time.time(); rows=[]
for H in Hgrid:
    faces=2*H+20; best=None
    for k in range(K):
        run_im(faces,"/tmp/s.obj"); V,tris=load_obj("/tmp/s.obj")
        r=process(V,tris)
        if best is None or (r['pent_err'],r['haus'])<(best['pent_err'],best['haus']): best=r
    rows.append((H,best))
    print("%3d %4d C%-4d  %2d    %5.1f      %5.1f       %.3f"%(
        H,faces,best['C'],best['defects'],best['fit_mean'],best['haus'],best['pent_err']))
print("\ntotal %.0fs for %d H x %d starts"%(time.time()-t0,len(Hgrid),K))
print("\nspread across H:  fit_mean %.1f-%.1f%%   Hausdorff %.1f-%.1f%%   pent_err %.3f-%.3f"%(
    min(r[1]['fit_mean'] for r in rows),max(r[1]['fit_mean'] for r in rows),
    min(r[1]['haus'] for r in rows),max(r[1]['haus'] for r in rows),
    min(r[1]['pent_err'] for r in rows),max(r[1]['pent_err'] for r in rows)))
b_rms=min(rows,key=lambda r:r[1]['fit_mean']); b_h=min(rows,key=lambda r:r[1]['haus']); b_p=min(rows,key=lambda r:r[1]['pent_err'])
print("best by fit_mean: H=%d ; best by Hausdorff: H=%d ; best by pent_err: H=%d"%(b_rms[0],b_h[0],b_p[0]))
