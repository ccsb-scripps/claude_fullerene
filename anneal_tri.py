#!/usr/bin/env python3
"""Anneal the subdivided-CVT triangulation toward a valid fullerene dual:
minimise sum|6-deg| (=12 iff exactly 12 pentagons, 0 heptagons)."""
import numpy as np, time
from scipy.spatial import cKDTree
from collections import defaultdict

d=np.load("/tmp/caps_subdiv.npz"); caps=d['caps']; Vs=d['Vs']; Fs=d['Fs']
N=len(caps)
_,lab=cKDTree(caps).query(Vs)
tris=set()
for f in Fs:
    l=(lab[f[0]],lab[f[1]],lab[f[2]])
    if l[0]!=l[1] and l[1]!=l[2] and l[0]!=l[2]: tris.add(tuple(sorted(l)))
tris=[list(t) for t in tris]

class Tri:
    def __init__(self,F,n):
        self.n=n; self.F=[list(f) for f in F]; self.deg=np.zeros(n,int)
        self.em=defaultdict(set)
        for ti,f in enumerate(self.F):
            for v in f: self.deg[v]+=1
            for i in range(3):
                a,b=f[i],f[(i+1)%3]; self.em[(min(a,b),max(a,b))].add(ti)
    def energy(self): return int(np.abs(6-self.deg).sum())
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

T=Tri(tris,N); print("start hist",T.hist(),"energy",T.energy(),"(target 12)")
rng=np.random.default_rng(1)
best=[f[:] for f in T.F]; bestE=T.energy()
t0=time.time()
for rs in range(30):
    T=Tri([f[:] for f in best],N); temp=3.0
    for it in range(30000):
        curE=T.energy()
        if curE<bestE: bestE=curE; best=[f[:] for f in T.F]
        if bestE<=12: break
        defs=np.where(T.deg!=6)[0]
        dv=int(defs[rng.integers(len(defs))])
        inc=[e for e in T.em if dv in e and len(T.em[e])==2]
        if not inc: continue
        e=inc[rng.integers(len(inc))]
        before=curE; r=T.flip(e)
        if not r: continue
        after=T.energy()
        if not (after<=before or rng.random()<np.exp(-(after-before)/max(temp,1e-6))):
            T.flip((min(r[2],r[3]),max(r[2],r[3])))  # revert
        temp*=0.9998
    if bestE<=12: break
    if rs%5==0: print("restart %d bestE %d (%.0fs)"%(rs,bestE,time.time()-t0))
Tf=Tri(best,N)
print("FINAL hist",Tf.hist(),"energy",bestE,"time %.0fs"%(time.time()-t0))
if bestE<=12:
    np.savez("/tmp/tri_clean.npz",tris=np.array(best),caps=caps)
    print("CLEAN fullerene dual reached -> /tmp/tri_clean.npz")
else:
    print("residual defects:",bestE-12)
