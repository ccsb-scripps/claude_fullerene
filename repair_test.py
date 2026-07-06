#!/usr/bin/env python3
"""Isolated, fast test of dipole annihilation on the C480 triangulation."""
import numpy as np, time
from collections import defaultdict, deque

def load_tri(path):
    Vc={}; F=[]
    for ln in open(path):
        if ln.startswith('v '):
            p=ln.split(); Vc[int(p[1])]=[float(p[2]),float(p[3]),float(p[4])]
        elif ln.startswith('f '):
            p=ln.split(); F.append([int(p[1]),int(p[2]),int(p[3])])
    n=max(Vc)+1; V=np.array([Vc[i] for i in range(n)])
    return V,F,n

class Tri:
    """Maintains triangles + incremental degrees + edge->tri map for fast flips."""
    def __init__(self,F,n):
        self.n=n; self.F=[list(f) for f in F]
        self.deg=np.zeros(n,int)
        self.em=defaultdict(set)
        for ti,f in enumerate(self.F):
            for v in f: self.deg[v]+=1
            for i in range(3):
                a,b=f[i],f[(i+1)%3]; self.em[(min(a,b),max(a,b))].add(ti)
    def energy(self):
        return int(np.abs(6-self.deg).sum())
    def hist(self):
        return {k:int((self.deg==k).sum()) for k in range(3,10) if (self.deg==k).sum()}
    def can_flip(self,e):
        tt=self.em[e]
        if len(tt)!=2: return None
        ti,tj=tuple(tt)
        a,b=e
        c=[v for v in self.F[ti] if v!=a and v!=b][0]
        d=[v for v in self.F[tj] if v!=a and v!=b][0]
        if c==d: return None
        if len(self.em[(min(c,d),max(c,d))])>0: return None
        return ti,tj,a,b,c,d
    def flip(self,e):
        r=self.can_flip(e)
        if r is None: return None
        ti,tj,a,b,c,d=r
        # remove old edges/tri membership
        for (x,y) in [(a,b),(a,c),(b,c)]:
            self.em[(min(x,y),max(x,y))].discard(ti)
        for (x,y) in [(a,b),(a,d),(b,d)]:
            self.em[(min(x,y),max(x,y))].discard(tj)
        self.F[ti]=[a,c,d]; self.F[tj]=[b,c,d]
        for (x,y) in [(a,c),(a,d),(c,d)]:
            self.em[(min(x,y),max(x,y))].add(ti)
        for (x,y) in [(b,c),(b,d),(c,d)]:
            self.em[(min(x,y),max(x,y))].add(tj)
        self.deg[a]-=1; self.deg[b]-=1; self.deg[c]+=1; self.deg[d]+=1
        return (a,b,c,d)

def repair(F,n,seed=0,restarts=60,iters=6000):
    rng=np.random.default_rng(seed)
    base=Tri(F,n)
    bestF=[f[:] for f in base.F]; bestE=base.energy()
    for rs in range(restarts):
        T=Tri([f[:] for f in bestF],n)  # restart from best
        temp=2.5
        for it in range(iters):
            curE=T.energy()
            if curE<=12:
                if curE<bestE: bestE=curE; bestF=[f[:] for f in T.F]
                if bestE<=12: break
            # propose an edge incident to a defect vertex
            defects=np.where(T.deg!=6)[0]
            if len(defects)==0: break
            dv=int(defects[rng.integers(len(defects))])
            inc=[e for e in T.em if dv in e and len(T.em[e])==2]
            if not inc: continue
            e=inc[rng.integers(len(inc))]
            snapF=[T.F[list(T.em[e])[0]][:],T.F[list(T.em[e])[1]][:]]
            before=curE
            r=T.flip(e)
            if r is None: continue
            after=T.energy()
            if after<=before or rng.random()<np.exp(-(after-before)/max(temp,1e-6)):
                if after<bestE: bestE=after; bestF=[f[:] for f in T.F]
            else:
                T.flip((min(r[2],r[3]),max(r[2],r[3])))  # reverse: flip new edge (c,d) back
            temp*=0.9997
        if bestE<=12: break
    return bestF,bestE

if __name__=="__main__":
    V,F,n=load_tri("/Users/olson/Downloads/C480_triangulation.txt")
    T0=Tri(F,n); print("start hist",T0.hist(),"energy",T0.energy())
    t=time.time()
    bestF,bestE=repair(F,n)
    Tf=Tri(bestF,n)
    print("repaired hist",Tf.hist(),"energy",bestE,"time %.1fs"%(time.time()-t))
    # save if clean
    if bestE<=12 and Tf.hist().get(5)==12 and 7 not in Tf.hist():
        with open("/tmp/clean_tri.txt","w") as f:
            for i in range(n): f.write("v %d %.4f %.4f %.4f\n"%(i,V[i,0],V[i,1],V[i,2]))
            for tri in bestF: f.write("f %d %d %d\n"%(tri[0],tri[1],tri[2]))
        print("SAVED clean triangulation -> /tmp/clean_tri.txt")
    else:
        print("NOT clean yet")
