#!/usr/bin/env python3
"""Sweep capped-tube (spindle) fullerenes over (W,Lt); for each: heal to a true
fullerene (12 pentagons), build the dual carbon cage, FF-relax to equilateral,
and score fit to mesh4a. Report the best."""
import numpy as np, sys, json
from scipy.spatial import cKDTree
from collections import defaultdict
from cap_tube import build_spindle, analyze

# ---------- heal 5-7 defects by edge flips (few defects -> easy) ----------
class Tri:
    def __init__(self,F,n):
        self.n=n; self.F=[list(f) for f in F]; self.deg=np.zeros(n,int); self.em=defaultdict(set)
        for ti,f in enumerate(self.F):
            for v in f: self.deg[v]+=1
            for i in range(3):
                a,b=f[i],f[(i+1)%3]; self.em[(min(a,b),max(a,b))].add(ti)
    def energy(self): return int((self.deg<5).sum()+(self.deg>6).sum())  # #vertices outside {5,6}
    def hist(self): return {k:int((self.deg==k).sum()) for k in range(3,11) if (self.deg==k).sum()}
    def can(self,e):
        tt=self.em[e]
        if len(tt)!=2: return None
        ti,tj=tuple(tt); a,b=e
        c=[v for v in self.F[ti] if v!=a and v!=b]; dd=[v for v in self.F[tj] if v!=a and v!=b]
        if not c or not dd: return None
        c,dd=c[0],dd[0]
        if c==dd or len(self.em[(min(c,dd),max(c,dd))])>0: return None
        return ti,tj,a,b,c,dd
    def flip(self,e):
        r=self.can(e)
        if not r: return None
        ti,tj,a,b,c,dd=r
        for x,y in [(a,b),(a,c),(b,c)]: self.em[(min(x,y),max(x,y))].discard(ti)
        for x,y in [(a,b),(a,dd),(b,dd)]: self.em[(min(x,y),max(x,y))].discard(tj)
        self.F[ti]=[a,c,dd]; self.F[tj]=[b,c,dd]
        for x,y in [(a,c),(a,dd),(c,dd)]: self.em[(min(x,y),max(x,y))].add(ti)
        for x,y in [(b,c),(b,dd),(c,dd)]: self.em[(min(x,y),max(x,y))].add(tj)
        self.deg[a]-=1; self.deg[b]-=1; self.deg[c]+=1; self.deg[dd]+=1
        return (a,b,c,dd)

def heal(tris,n,seed=0):
    rng=np.random.default_rng(seed)
    best=[list(t) for t in tris]; T=Tri(tris,n); bestE=T.energy()
    for rs in range(25):
        T=Tri([t[:] for t in best],n); temp=2.0
        for it in range(8000):
            E=T.energy()
            if E<bestE: bestE=E; best=[t[:] for t in T.F]
            if bestE<=0: break
            defs=np.where((T.deg<5)|(T.deg>6))[0]      # only truly-bad vertices
            if len(defs)==0: break
            dv=int(defs[rng.integers(len(defs))])
            inc=[e for e in T.em if dv in e and len(T.em[e])==2]
            if not inc: continue
            e=inc[rng.integers(len(inc))]; b0=E; r=T.flip(e)
            if not r: continue
            if not (T.energy()<=b0 or rng.random()<np.exp(-(T.energy()-b0)/max(temp,1e-6))):
                T.flip((min(r[2],r[3]),max(r[2],r[3])))
            temp*=0.999
        if bestE<=0: break
    return best,bestE

# ---------- dual carbon cage + FF relax ----------
def dual(tris,pts):
    cent=np.array([pts[list(t)].mean(0) for t in tris])
    em=defaultdict(list)
    for ti,t in enumerate(tris):
        for i in range(3):
            a,b=t[i],t[(i+1)%3]; em[(min(a,b),max(a,b))].append(ti)
    bonds=sorted({(min(x),max(x)) for x in em.values() if len(x)==2})
    return cent,bonds

def ff_relax(X,bonds,iters=2500,kb=1.0,ka=1.5,step=0.05):
    n=len(X); nbr=defaultdict(list)
    for a,b in bonds: nbr[a].append(b); nbr[b].append(a)
    L0=np.median([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
    angles=[]
    for i in range(n):
        nb=nbr[i]
        for p in range(len(nb)):
            for q in range(p+1,len(nb)): angles.append((nb[p],i,nb[q]))
    th0=np.radians(120.0)
    for it in range(iters):
        Fc=np.zeros_like(X)
        for a,b in bonds:
            e=X[a]-X[b]; L=np.linalg.norm(e)+1e-12; f=kb*(L-L0)*e/L; Fc[a]-=f; Fc[b]+=f
        for (j,i,k) in angles:
            rij=X[j]-X[i]; rik=X[k]-X[i]; lij=np.linalg.norm(rij)+1e-12; lik=np.linalg.norm(rik)+1e-12
            c=np.clip(rij@rik/(lij*lik),-1,1); th=np.arccos(c); ssin=max(np.sin(th),1e-6); dE=ka*(th-th0)
            dcj=rik/(lij*lik)-c*rij/lij**2; dck=rij/(lij*lik)-c*rik/lik**2
            gj=dE/ssin*dcj; gk=dE/ssin*dck; Fc[j]+=gj; Fc[k]+=gk; Fc[i]-=(gj+gk)
        X=X+step*Fc
    return X

def fit_metrics(X,Vm):
    def pca(P): c=P.mean(0);u,s,vt=np.linalg.svd(P-c,full_matrices=False);return c,vt,s
    cX,vtX,sX=pca(X); cM,vtM,sM=pca(Vm)
    Xc=(X-cX)@vtX.T; Mc=(Vm-cM)@vtM.T
    for k in range(3):
        if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
    uni=(Mc.max(0)-Mc.min(0)).mean()/(Xc.max(0)-Xc.min(0)).mean()
    Xw=(Xc*uni)@vtM+cM
    dS,_=cKDTree(Vm).query(Xw); Rm=np.linalg.norm(Vm-cM,axis=1).mean()
    return (sX/sX.max()), dS.mean()/Rm*100, dS.max()/Rm*100, Xw

def evaluate(W,Lt,Vm,verbose=False):
    pts,tris,_=build_spindle(W,Lt); n=len(pts)
    tris=[tuple(sorted(t)) for t in set(tuple(sorted(t)) for t in tris)]
    healed,E=heal(tris,n)
    if E>0: return None
    X0,bonds=dual(healed,pts)
    X=ff_relax(X0,bonds)
    bl=np.array([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
    asp,fmean,fmax,Xw=fit_metrics(X,Vm)
    C=len(X)
    if verbose:
        print("  C%d bondCV %.2f%% relaxed-aspect %.2f fit(mean/max %%R) %.1f/%.1f"%(
            C,bl.std()/bl.mean()*100,asp[0]/asp[-1],fmean,fmax))
    return dict(W=W,Lt=Lt,C=C,bondCV=bl.std()/bl.mean()*100,aspect=asp[0]/asp[-1],
                fit_mean=fmean,fit_max=fmax,X=X,bonds=bonds,Xw=Xw,healed=healed,pts=pts)

if __name__=="__main__":
    m=np.load("/tmp/mesh4a.npz"); Vm=m['V']
    # candidate (W,Lt) with C in [460,520], pre-filtered by geometric aspect near 2.6
    cand=[]
    for W in range(12,20):
        for Lt in range(1,16):
            pts,tris,_=build_spindle(W,Lt); C=2*(len(pts)-2)
            if 460<=C<=520:
                c=pts.mean(0);u,s,vt=np.linalg.svd(pts-c,full_matrices=False)
                ext=(pts-c)@vt.T;L=ext.max(0)-ext.min(0)
                cand.append((abs(L.max()/L.min()-2.6),W,Lt,C))
    cand.sort(); cand=cand[:8]
    print("evaluating %d candidates..."%len(cand))
    results=[]
    for _,W,Lt,C in cand:
        print("W=%d Lt=%d (C~%d):"%(W,Lt,C)); r=evaluate(W,Lt,Vm,verbose=True)
        if r: results.append(r)
    results.sort(key=lambda r:r['fit_mean'])
    print("\n=== BEST FIT ===")
    for r in results[:5]:
        print("C%d (W%d,Lt%d): fit mean %.1f%% max %.1f%% aspect %.2f bondCV %.2f%%"%(
            r['C'],r['W'],r['Lt'],r['fit_mean'],r['fit_max'],r['aspect'],r['bondCV']))
    b=results[0]
    np.savez("/tmp/best_spindle.npz",X=b['X'],Xw=b['Xw'],bonds=np.array(b['bonds']),
             healed=np.array(b['healed']),pts=b['pts'],W=b['W'],Lt=b['Lt'],C=b['C'])
    print("saved best C%d"%b['C'])
