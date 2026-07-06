#!/usr/bin/env python3
"""Extract a ring-spiral (RSPI) from the valid C500 fullerene graph, so the
Fullerene program can rebuild+relax it from pure topology (ICart=3)."""
import numpy as np
from collections import defaultdict

d=np.load("/tmp/geo_result.npz")
C=d['carbons']; geo_faces=[tuple(t) for t in d['geo_faces']]; gdeg=d['gdeg']; cm=C.mean(0)
# fullerene faces = dual of geodesic vertices, oriented CCW about outward normal
inc=defaultdict(list)
for ti,f in enumerate(geo_faces):
    for v in f: inc[v].append(ti)
def order_face(members):
    P=C[members]; c=P.mean(0); nrm=c-cm; nrm/=np.linalg.norm(nrm)+1e-9
    t=np.array([1.0,0,0]);
    if abs(nrm[0])>0.9: t=np.array([0,1.0,0])
    e1=t-nrm*np.dot(t,nrm); e1/=np.linalg.norm(e1); e2=np.cross(nrm,e1)
    ang=[np.arctan2((P[q]-c)@e2,(P[q]-c)@e1) for q in range(len(members))]
    return [members[q] for q in np.argsort(ang)]
faces=[order_face(inc[v]) for v in range(len(gdeg))]
F=len(faces); sizes=[len(f) for f in faces]
print("faces=%d pentagons=%d hexagons=%d"%(F,sizes.count(5),sizes.count(6)))

# face adjacency via shared carbon-pair (bond); cyclic neighbor order per face
edge2f=defaultdict(list)
for fi,f in enumerate(faces):
    k=len(f)
    for i in range(k):
        edge2f[frozenset((f[i],f[(i+1)%k]))].append(fi)
nbr_cyc=[]
for fi,f in enumerate(faces):
    k=len(f); nb=[]
    for i in range(k):
        e=frozenset((f[i],f[(i+1)%k])); pair=edge2f[e]
        other=[x for x in pair if x!=fi]
        nb.append(other[0] if other else -1)
    nbr_cyc.append(nb)

def wind(s, first, direction):
    visited=[False]*F; S=[s]; visited[s]=True
    cur=s; prev=None
    # first move
    nxt=nbr_cyc[s][first];
    if nxt<0 or visited[nxt]: return None
    S.append(nxt); visited[nxt]=True; prev=s; cur=nxt
    while len(S)<F:
        cyc=nbr_cyc[cur]; k=len(cyc)
        try: i=cyc.index(prev)
        except ValueError: return None
        nxt=None
        for step in range(1,k+1):
            cand=cyc[(i+direction*step)%k]
            if cand>=0 and not visited[cand]: nxt=cand; break
        if nxt is None: return None
        S.append(nxt); visited[nxt]=True; prev=cur; cur=nxt
    return S

rspi=None
import itertools
for s in range(F):
    if len(faces[s])!=5: continue          # start on a pentagon (canonical-ish)
    for first in range(len(nbr_cyc[s])):
        for direction in (1,-1):
            S=wind(s,first,direction)
            if S and len(S)==F and len(set(S))==F:
                pos=[i+1 for i,fi in enumerate(S) if len(faces[fi])==5]
                if len(pos)==12: rspi=pos; break
        if rspi: break
    if rspi: break
print("RSPI found:" , rspi if rspi else "NONE")
if rspi:
    with open("/Users/olson/Dev/fullerenes/mesh4a_C500_rspi.inp","w") as f:
        f.write("C500 mesh4a geodesic - rebuild+relax from ring spiral\n")
        f.write("&General NA=500 iwext=1 /\n")
        f.write("&Coord ICart=3, rspi=%s /\n"%(",".join(map(str,rspi))))
        f.write("&FFChoice Iopt=4 /\n&FFParameters /\n&Hamilton /\n&Isomers /\n&Graph /\n")
    print("wrote mesh4a_C500_rspi.inp")
