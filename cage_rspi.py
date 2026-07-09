#!/usr/bin/env python3
"""Canonical ring-spiral pentagon indices (RSPI) computed directly from a cage's
topology -- no coordinates fed to the Fullerene program (whose distance-based bond
reader over-counts on crowded/elongated cages). Detects faces via a rotation
system, builds face adjacency, winds spirals from every pentagon start/direction,
and returns the lexicographically smallest valid 12-pentagon spiral.

Usage: python cage_rspi.py <cage.xyz>
"""
import sys, numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict

def load_xyz(fn):
    P=[]
    for ln in open(fn).read().split('\n')[2:]:
        t=ln.split()
        if len(t)>=4: P.append([float(t[1]),float(t[2]),float(t[3])])
    return np.array(P)

def faces_of(X):
    n=len(X); tr=cKDTree(X); d,_=tr.query(X,k=4); L=np.median(d[:,1:4])
    pairs=tr.query_pairs(1.35*L); adj=[[] for _ in range(n)]
    for a,b in pairs: adj[a].append(b); adj[b].append(a)
    ctr=X.mean(0); oadj=[]
    for i in range(n):
        nrm=X[i]-ctr; nrm/=np.linalg.norm(nrm)+1e-9
        t=np.array([1.,0,0]) if abs(nrm[0])<0.9 else np.array([0,1.,0])
        e1=t-(t@nrm)*nrm; e1/=np.linalg.norm(e1); e2=np.cross(nrm,e1)
        nb=adj[i]; ang=[np.arctan2((X[j]-X[i])@e2,(X[j]-X[i])@e1) for j in nb]
        oadj.append([nb[k] for k in np.argsort(ang)])
    def trace(step):
        seen=set(); F=[]
        for u in range(n):
            for v in adj[u]:
                if (u,v) in seen: continue
                f=[]; a,b=u,v
                for _ in range(9):
                    seen.add((a,b)); f.append(a); nb=oadj[b]; a,b=b,nb[(nb.index(a)+step)%len(nb)]
                    if (a,b)==(u,v): break
                F.append(f)
        return F
    F=trace(1)
    if sum(len(f)==5 for f in F)!=12: F=trace(-1)
    return [f for f in F if len(f) in (5,6)]

def orient(faces):
    """flood-fill a consistent orientation so adjacent faces traverse the shared
    edge in opposite directions (required by the spiral winder)."""
    ue=defaultdict(list)
    for fi,f in enumerate(faces):
        k=len(f)
        for i in range(k): ue[frozenset((f[i],f[(i+1)%k]))].append(fi)
    de=lambda f:{(f[i],f[(i+1)%len(f)]) for i in range(len(f))}
    out=[None]*len(faces); out[0]=list(faces[0]); done=[False]*len(faces); done[0]=True
    from collections import deque
    q=deque([0])
    while q:
        fi=q.popleft()
        for (a,b) in de(out[fi]):
            for fj in ue[frozenset((a,b))]:
                if fj==fi or done[fj]: continue
                g=list(faces[fj]); out[fj]=g if (b,a) in de(g) else g[::-1]
                done[fj]=True; q.append(fj)
    return out

def canonical_rspi(faces):
    faces=orient(faces)
    F=len(faces)
    edge2f=defaultdict(list)
    for fi,f in enumerate(faces):
        k=len(f)
        for i in range(k): edge2f[frozenset((f[i],f[(i+1)%k]))].append(fi)
    nbr=[]
    for fi,f in enumerate(faces):
        k=len(f); row=[]
        for i in range(k):
            o=[x for x in edge2f[frozenset((f[i],f[(i+1)%k]))] if x!=fi]; row.append(o[0] if o else -1)
        nbr.append(row)
    def wind(s,first,direction):
        vis=[False]*F; nx=nbr[s][first]
        if nx<0: return None
        S=[s,nx]; vis[s]=vis[nx]=True; prev,cur=s,nx
        while len(S)<F:
            cyc=nbr[cur]; k=len(cyc)
            try: i=cyc.index(prev)
            except ValueError: return None
            nxt=None
            for step in range(1,k+1):
                cand=cyc[(i+direction*step)%k]
                if cand>=0 and not vis[cand]: nxt=cand; break
            if nxt is None: return None
            S.append(nxt); vis[nxt]=True; prev,cur=cur,nxt
        return S
    cands=[]
    for s in range(F):
        if len(faces[s])!=5: continue
        for first in range(len(nbr[s])):
            for direction in (1,-1):
                S=wind(s,first,direction)
                if S and len(set(S))==F:
                    pos=tuple(i+1 for i,fi in enumerate(S) if len(faces[fi])==5)
                    if len(pos)==12: cands.append(pos)
    return min(cands) if cands else None

if __name__=="__main__":
    faces=faces_of(load_xyz(sys.argv[1]))
    npent=sum(len(f)==5 for f in faces)
    print("faces=%d (pentagons=%d hexagons=%d)"%(len(faces),npent,sum(len(f)==6 for f in faces)))
    r=canonical_rspi(faces)
    print("canonical RSPI:", " ".join(map(str,r)) if r else "NONE (no valid spiral)")
