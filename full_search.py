#!/usr/bin/env python3
"""Full systematic search with composite scoring.
 1) scan H x multistart, cheap metrics (defects, pentamer-placement error);
 2) relax+fit the shortlist (Hausdorff, fit_mean, regularity);
 3) composite rank: pent_err primary, Hausdorff secondary, defect-free preferred;
 4) lock in representative (heal if needed), report the near-tie band."""
import numpy as np, subprocess, time, sys, json
from scipy.spatial import cKDTree
from collections import defaultdict
IMBIN="/Users/olson/Dev/fullerenes/instant-meshes/InstantMeshesBatch"; MESH="/tmp/mesh4a.obj"
# guard: kill any stale InstantMeshesBatch from a previous/crashed run -- leftover
# procs contend for oneTBB threads and can silently corrupt this run's tilings.
subprocess.run(["pkill","-9","-f","InstantMeshesBatch"],stderr=subprocess.DEVNULL); time.sleep(0.3)
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']
cM=Vm.mean(0); u0,s0,vtM=np.linalg.svd(Vm-cM,full_matrices=False); Mc=(Vm-cM)@vtM.T
Rm=np.linalg.norm(Vm-cM,axis=1).mean(); tree=cKDTree(Vm)
pk=np.load("/tmp/pentamers.npz"); PEAKS=pk['sites']
ar=0.5*np.linalg.norm(np.cross(Vm[Fm[:,1]]-Vm[Fm[:,0]],Vm[Fm[:,2]]-Vm[Fm[:,0]]),axis=1).sum()
HEXDIA=float(np.sqrt(2*ar/(242*np.sqrt(3))))

def run_im(faces,out):
    for _ in range(4):   # retry with a timeout guard: IM occasionally hangs on a degenerate field
        try:
            subprocess.run([IMBIN,MESH,"-o",out,"-r","6","-p","6","-f",str(faces)],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=6)
            return True
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill","-f","InstantMeshesBatch"],stderr=subprocess.DEVNULL); continue
    return False
def load_obj(fn):
    V=[];T=[]
    for ln in open(fn):
        p=ln.split()
        if not p: continue
        if p[0]=='v': V.append([float(x) for x in p[1:4]])
        elif p[0]=='f': T.append([int(x.split('/')[0])-1 for x in p[1:]])
    return np.array(V),[tuple(t) for t in T]
def cheap(V,tris):
    n=len(V); deg=np.zeros(n,int)
    for t in tris:
        for v in t: deg[v]+=1
    d5=V[np.where(deg==5)[0]]
    pe=float(np.linalg.norm(PEAKS[:,None,:]-d5[None,:,:],axis=2).min(axis=1).mean())/HEXDIA if len(d5) else 9.9
    return int((deg<5).sum()+(deg>6).sum()), pe
def dualcage(V,tris):
    e2t=defaultdict(list)
    for ti,t in enumerate(tris):
        for i in range(3):
            a,b=t[i],t[(i+1)%3]; e2t[(min(a,b),max(a,b))].append(ti)
    bonds=np.array(sorted({(min(x),max(x)) for x in e2t.values() if len(x)==2}))
    X=np.array([V[list(t)].mean(0) for t in tris])
    nbr=defaultdict(list)
    for a,b in bonds: nbr[a].append(b); nbr[b].append(a)
    angs=[]
    for i in range(len(X)):
        nb=nbr[i]
        for p_ in range(len(nb)):
            for q in range(p_+1,len(nb)): angs.append((nb[p_],i,nb[q]))
    return X,bonds,np.array(angs),e2t
def relax(X,bonds,A,iters):
    th0=np.radians(120.); L0=np.median(np.linalg.norm(X[bonds[:,0]]-X[bonds[:,1]],axis=1))
    for it in range(iters):
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
    return X
def fitmetrics(X):
    cX=X.mean(0); u,s,vt=np.linalg.svd(X-cX,full_matrices=False); Xc=(X-cX)@vt.T
    for k in range(3):
        if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
    uni=(Mc.max(0)-Mc.min(0)).mean()/(Xc.max(0)-Xc.min(0)).mean()
    Xw=(Xc*uni)@vtM+cM
    dS,_=tree.query(Xw); dM,_=cKDTree(Xw).query(Vm)
    return dS.mean()/Rm*100, max(np.percentile(dS,95),np.percentile(dM,95))/Rm*100, Xw

t0=time.time()
Hgrid=list(range(220,251,2)); K=6
cands=[]   # (H,defects,pent_err,objfile)
ci=0
for H in Hgrid:
    faces=2*H+20
    for k in range(K):
        of="/tmp/cand_%d.obj"%ci; run_im(faces,of); V,tris=load_obj(of)
        dd,pe=cheap(V,tris); cands.append([H,dd,pe,of]); ci+=1
print("scanned %d candidates in %.0fs"%(len(cands),time.time()-t0),flush=True)
# guarantee a TRUE-fullerene (defect-free) candidate exists: the remesher is
# stochastic, so an unlucky scan may contain none. Escalate multistart -- extra
# random draws on the H values that came closest -- until one appears (capped).
tries=0
while not any(c[1]==0 for c in cands) and tries<int(sys.argv[1] if len(sys.argv)>1 else 150):
    for H in Hgrid:                      # sweep the full grid with fresh random seeds
        of="/tmp/cand_%d.obj"%ci; run_im(2*H+20,of); V,tris=load_obj(of)
        dd,pe=cheap(V,tris); cands.append([H,dd,pe,of]); ci+=1; tries+=1
        if dd==0: break
    if any(c[1]==0 for c in cands): break
if tries: print("top-up multistart: +%d draws, defect-free found=%s"%(
    tries,any(c[1]==0 for c in cands)),flush=True)
# shortlist: prefer valid/low-defect + good pent_err
cands.sort(key=lambda c:(min(c[1],3), c[2]))     # defects capped at 3 as gate, then pent_err
short=cands[:16]
print("shortlist (H,defects,pent_err): %s"%[(c[0],c[1],round(c[2],3)) for c in short[:8]],flush=True)
scored=[]
for H,dd,pe,of in short:
    V,tris=load_obj(of); X,bonds,A,e2t=dualcage(V,tris); X=relax(X,bonds,A,2000)
    fm,hz,_=fitmetrics(X)
    scored.append(dict(H=H,defects=dd,pent_err=pe,fit_mean=fm,haus=hz,obj=of))
# composite ranking. A TRUE fullerene is required, so defect-free is a HARD
# preference: if any defect-free candidate exists, choose among those only. The
# fit_mean<8 gate (computed from the search's rough, orientation-unresolved fit --
# badly inflated for near-spherical shapes) must never drop every defect-free
# option, so it only applies within the chosen pool. Rank pent_err then Hausdorff
# (pent_err is orientation-independent; the authoritative fit is done in finalize).
df=[s for s in scored if s['defects']==0]
if df:
    pool=df
else:
    pool=[s for s in scored if s['fit_mean']<8.0] or scored
valid=sorted(pool,key=lambda s:(round(s['pent_err'],3), round(s['haus'],2)))
print("\n=== composite ranking (top, %s) ==="%("defect-free pool" if df else "no defect-free found"))
print(" H   def  pent_err  Hausdorff%  fit_mean%")
for s in valid[:8]:
    print("%3d   %2d   %.3f      %5.1f      %5.1f"%(s['H'],s['defects'],s['pent_err'],s['haus'],s['fit_mean']))
rep=valid[0]
# near-tie band: within 0.05 hexdia pent_err and 1% Hausdorff of representative
band=[s for s in valid if abs(s['pent_err']-rep['pent_err'])<0.05 and abs(s['haus']-rep['haus'])<1.5]
print("\nREPRESENTATIVE: H=%d defects=%d pent_err=%.3f Hausdorff=%.1f%% fit_mean=%.1f%%"%(
    rep['H'],rep['defects'],rep['pent_err'],rep['haus'],rep['fit_mean']))
print("near-tie band: %d structures within (dpent<0.05, dHaus<1.5%%) spanning H=%s"%(
    len(band),sorted(set(s['H'] for s in band))))
print("total search time %.0fs"%(time.time()-t0),flush=True)
json.dump({'representative':{k:rep[k] for k in ['H','defects','pent_err','haus','fit_mean','obj']},
           'band_H':sorted(set(s['H'] for s in band)),
           'all':[{k:s[k] for k in ['H','defects','pent_err','haus','fit_mean']} for s in scored]},
          open("/tmp/search_result.json","w"))
print("saved /tmp/search_result.json ; representative tiling = %s"%rep['obj'])
