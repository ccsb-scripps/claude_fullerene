#!/usr/bin/env python3
"""Capped-tube (spindle) fullerene constructor.
Capsomer centers on latitude rings of a spherocylinder; correct annulus
triangulation between consecutive rings. If all vertices are degree 5/6, Euler
guarantees exactly 12 pentagons. Fullerene = dual.
Params: W = capsomers around tube (width); Lt = tube rings (length)."""
import numpy as np, sys
from collections import defaultdict

def ring_sizes(W, Lt):
    caps=list(range(6, W+1))            # 6,7,...,W  (gentle +1 growth)
    return [1] + caps + [W]*Lt + caps[::-1] + [1]

def build_spindle(W, Lt, s=1.0):
    row=s*np.sqrt(3)/2
    R=W*s/(2*np.pi)                      # tube radius
    caps=list(range(6,W+1))             # ring sizes 6..W (gentle +1)
    def rad(n): return n*s/(2*np.pi)
    def zoff(n): return np.sqrt(max(R*R-rad(n)**2,0.0))   # dome height for size-n ring
    ztube=Lt*row
    rings=[(-R,0.0,1)]                                     # bottom apex
    for n in caps: rings.append((-zoff(n), rad(n), n))     # bottom cap 6..W climbs to z=0
    for j in range(1,Lt+1): rings.append((j*row, R, W))    # tube
    for n in caps[::-1]: rings.append((ztube+(R-zoff(n)), rad(n), n))  # top cap W..6
    rings.append((ztube+R,0.0,1))                          # top apex
    pts=[]; ringidx=[]
    for ri,(z,rad,n) in enumerate(rings):
        off=0.5*(ri%2); idx=[]
        for i in range(n):
            a=2*np.pi*(i+off)/n
            idx.append(len(pts)); pts.append((rad*np.cos(a),rad*np.sin(a),z))
        ringidx.append(idx)
    pts=np.array(pts)
    tris=[]
    def band(low,up):
        a=len(low); b=len(up); i=j=0; out=[]
        for _ in range(a+b):
            if i>=a: adv='u'
            elif j>=b: adv='l'
            else:
                dl=np.linalg.norm(pts[low[(i+1)%a]]-pts[up[j%b]])
                du=np.linalg.norm(pts[up[(j+1)%b]]-pts[low[i%a]])
                adv='l' if dl<=du else 'u'
            if adv=='l': out.append((low[i%a],low[(i+1)%a],up[j%b])); i+=1
            else:        out.append((low[i%a],up[(j+1)%b],up[j%b])); j+=1
        return out
    for ri in range(len(ringidx)-1):
        low,up=ringidx[ri],ringidx[ri+1]
        if len(low)==1:
            c=low[0]
            for i in range(len(up)): tris.append((c,up[i],up[(i+1)%len(up)]))
        elif len(up)==1:
            c=up[0]
            for i in range(len(low)): tris.append((c,low[(i+1)%len(low)],low[i]))
        else:
            tris+=band(low,up)
    return pts,tris,ringidx

def analyze(pts,tris):
    T=set(tuple(sorted(t)) for t in tris if len(set(t))==3)
    deg=np.zeros(len(pts),int)
    for t in T:
        for v in t: deg[v]+=1
    return deg,T

if __name__=="__main__":
    W=int(sys.argv[1]) if len(sys.argv)>1 else 12
    Lt=int(sys.argv[2]) if len(sys.argv)>2 else 6
    pts,tris,ringidx=build_spindle(W,Lt)
    deg,T=analyze(pts,tris)
    N=len(pts)
    hist={k:int((deg==k).sum()) for k in range(2,10) if (deg==k).sum()}
    C=2*(N-2)                          # carbons in dual fullerene
    print("W=%d Lt=%d capsomers=%d  C=%d  sizes=%s"%(W,Lt,N,C,sizes:=[len(r) for r in ringidx]))
    print("  tris=%d valence=%s  netcharge Σ(6-deg)=%d  (want 12, all deg in {5,6})"%(
        len(T),hist,int((6-deg).sum())))
