#!/usr/bin/env python3
"""Reusable flip-healing for near-defect-free capsomer triangulations.

Instant Meshes often returns a tiling that is one or a few 5-7 scars away from a
true fullerene (12 pentagons, no other defects). heal() annihilates those defects
by edge flips (greedy best-flip passes + a short simulated-annealing fallback that
glides defects together), returning a defect-free triangulation when it can.
"""
import numpy as np, time
from collections import defaultdict

class Tri:
    def __init__(s, F, n):
        s.F=[list(t) for t in F]; s.deg=np.zeros(n,int); s.em=defaultdict(set); s.ve=defaultdict(set)
        for ti,f in enumerate(s.F):
            for v in f: s.deg[v]+=1
            for i in range(3):
                a,b=f[i],f[(i+1)%3]; k=(min(a,b),max(a,b)); s.em[k].add(ti); s.ve[a].add(k); s.ve[b].add(k)
    def nd(s): return int((s.deg<5).sum()+(s.deg>6).sum())
    def flip(s, e):
        tt=s.em[e]
        if len(tt)!=2: return None
        ti,tj=tuple(tt); a,b=e
        c=[v for v in s.F[ti] if v!=a and v!=b]; dd=[v for v in s.F[tj] if v!=a and v!=b]
        if not c or not dd: return None
        c,dd=c[0],dd[0]; cd=(min(c,dd),max(c,dd))
        if c==dd or len(s.em[cd])>0: return None
        for x,y in [(a,b),(a,c),(b,c)]: s.em[(min(x,y),max(x,y))].discard(ti)
        for x,y in [(a,b),(a,dd),(b,dd)]: s.em[(min(x,y),max(x,y))].discard(tj)
        s.F[ti]=[a,c,dd]; s.F[tj]=[b,c,dd]
        for x,y in [(a,c),(a,dd),(c,dd)]: s.em[(min(x,y),max(x,y))].add(ti)
        for x,y in [(b,c),(b,dd),(c,dd)]: s.em[(min(x,y),max(x,y))].add(tj)
        s.ve[a].discard(e); s.ve[b].discard(e); s.ve[c].add(cd); s.ve[dd].add(cd)
        s.deg[a]-=1; s.deg[b]-=1; s.deg[c]+=1; s.deg[dd]+=1; return (a,b,c,dd)

def _greedy(tri):
    while True:
        bd=tri.nd(); be=None
        for e in list(tri.em):
            if len(tri.em[e])!=2: continue
            r=tri.flip(e)
            if r is None: continue
            if tri.nd()<bd: bd=tri.nd(); be=e
            tri.flip((min(r[2],r[3]),max(r[2],r[3])))     # undo (test only)
        if be is None: return
        tri.flip(be)

def heal(V, tris, budget=3.0, seed=3):
    """Return (healed_tris, num_defects). num_defects==0 means success."""
    n=len(V)
    edges={(min(t[i],t[(i+1)%3]),max(t[i],t[(i+1)%3])) for t in tris for i in range(3)}
    sp=np.median([np.linalg.norm(V[a]-V[b]) for a,b in edges])
    def attr(deg):
        hp=np.where(deg>6)[0]; pt=np.where(deg<6)[0]
        if len(hp)==0 or len(pt)==0: return 0.0
        return float(np.linalg.norm(V[hp][:,None,:]-V[pt][None,:,:],axis=2).min(1).sum())/sp
    rng=np.random.default_rng(seed)
    tri=Tri(tris,n); best=[t[:] for t in tri.F]; bestnd=tri.nd(); t0=time.time()
    while bestnd>0 and time.time()-t0<budget:
        tri=Tri([t[:] for t in best],n); _greedy(tri)
        if tri.nd()<bestnd: bestnd=tri.nd(); best=[t[:] for t in tri.F]
        temp=8.0; stuck=0
        while tri.nd()>0 and time.time()-t0<budget:
            for _ in range(120):
                defs=np.where((tri.deg<5)|(tri.deg>6))[0]
                if len(defs)==0: break
                dv=int(defs[rng.integers(len(defs))]); inc=[e for e in tri.ve[dv] if len(tri.em[e])==2]
                if not inc: continue
                e=inc[rng.integers(len(inc))]; cur=1000*tri.nd()+attr(tri.deg); r=tri.flip(e)
                if not r: continue
                new=1000*tri.nd()+attr(tri.deg)
                if not (new<=cur or rng.random()<np.exp(-(new-cur)/max(temp,1e-6))): tri.flip((min(r[2],r[3]),max(r[2],r[3])))
                temp*=0.9985
            _greedy(tri)
            if tri.nd()<bestnd: bestnd=tri.nd(); best=[t[:] for t in tri.F]; stuck=0
            else: stuck+=1
            if stuck>25: break
    return [tuple(t) for t in best], bestnd
