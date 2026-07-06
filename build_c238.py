#!/usr/bin/env python3
"""Build the clean C238 fullerene from the Instant Meshes field-aligned tiling:
heal -> dual carbon cage -> ring spiral (RSPI) for the Fullerene program, and
save the on-mesh dual geometry for the viewer."""
import numpy as np, json
from collections import defaultdict

# ---- load IM capsomer triangulation (verts=capsomers on mesh4a, tris=adjacency)
V=[];T=[]
for ln in open("/tmp/im_out.obj"):
    p=ln.split()
    if not p: continue
    if p[0]=='v': V.append([float(x) for x in p[1:4]])
    elif p[0]=='f': T.append([int(x.split('/')[0])-1 for x in p[1:]])
V=np.array(V); n=len(V)

# ---- heal to {5:12,6:rest} ----
class Tri:
    def __init__(s,F):
        s.F=[list(t) for t in F]; s.deg=np.zeros(n,int); s.em=defaultdict(set)
        for ti,f in enumerate(s.F):
            for v in f: s.deg[v]+=1
            for i in range(3):
                a,b=f[i],f[(i+1)%3]; s.em[(min(a,b),max(a,b))].add(ti)
    def E(s): return int((s.deg<5).sum()+(s.deg>6).sum())
    def flip(s,e):
        tt=s.em[e]
        if len(tt)!=2: return None
        ti,tj=tuple(tt); a,b=e
        c=[v for v in s.F[ti] if v!=a and v!=b]; dd=[v for v in s.F[tj] if v!=a and v!=b]
        if not c or not dd: return None
        c,dd=c[0],dd[0]
        if c==dd or len(s.em[(min(c,dd),max(c,dd))])>0: return None
        for x,y in [(a,b),(a,c),(b,c)]: s.em[(min(x,y),max(x,y))].discard(ti)
        for x,y in [(a,b),(a,dd),(b,dd)]: s.em[(min(x,y),max(x,y))].discard(tj)
        s.F[ti]=[a,c,dd]; s.F[tj]=[b,c,dd]
        for x,y in [(a,c),(a,dd),(c,dd)]: s.em[(min(x,y),max(x,y))].add(ti)
        for x,y in [(b,c),(b,dd),(c,dd)]: s.em[(min(x,y),max(x,y))].add(tj)
        s.deg[a]-=1;s.deg[b]-=1;s.deg[c]+=1;s.deg[dd]+=1; return (a,b,c,dd)
rng=np.random.default_rng(0); best=[list(t) for t in T]; tri=Tri(T); bestE=tri.E()
for rs in range(150):
    tri=Tri([t[:] for t in best]); temp=1.5
    for it in range(9000):
        if tri.E()<bestE: bestE=tri.E(); best=[t[:] for t in tri.F]
        if bestE==0: break
        defs=np.where((tri.deg<5)|(tri.deg>6))[0]
        if len(defs)==0: break
        dv=int(defs[rng.integers(len(defs))]); inc=[e for e in tri.em if dv in e and len(tri.em[e])==2]
        if not inc: continue
        e=inc[rng.integers(len(inc))]; b0=tri.E(); r=tri.flip(e)
        if not r: continue
        if not (tri.E()<=b0 or rng.random()<np.exp(-(tri.E()-b0)/max(temp,1e-6))): tri.flip((min(r[2],r[3]),max(r[2],r[3])))
        temp*=0.999
    if bestE==0: break
tris=[tuple(t) for t in best]; d=Tri(tris).deg
print("healed: %d capsomers, valence %s"%(n,{k:int((d==k).sum()) for k in range(3,9) if (d==k).sum()}))

# ---- dual carbon cage: carbon per triangle; bonds share edge; faces = fan around capsomer
e2t=defaultdict(list)
for ti,t in enumerate(tris):
    for i in range(3):
        a,b=t[i],t[(i+1)%3]; e2t[(min(a,b),max(a,b))].append(ti)
bonds=sorted({(min(x),max(x)) for x in e2t.values() if len(x)==2})
carbons=np.array([V[list(t)].mean(0) for t in tris])       # on mesh (approx)
inc=defaultdict(list)
for ti,t in enumerate(tris):
    for v in t: inc[v].append(ti)
def order_face(v):
    cur=inc[v][0]; face=[cur]; w=[x for x in tris[cur] if x!=v][0]
    for _ in range(len(inc[v])+2):
        nb=[t for t in e2t[(min(v,w),max(v,w))] if t!=cur]
        if not nb: break
        nx=nb[0]
        if nx==face[0]: break
        face.append(nx); ow=[x for x in tris[nx] if x!=v and x!=w]
        if not ow: break
        w=ow[0]; cur=nx
    return face
faces=[order_face(v) for v in range(n)]
pent=[f for f in faces if len(f)==5]; hexf=[f for f in faces if len(f)==6]
print("fullerene: carbons=%d bonds=%d faces=%d (pent %d hex %d)"%(len(carbons),len(bonds),len(faces),len(pent),len(hexf)))

# ---- ring spiral (RSPI) from the face adjacency ----
f_edges=defaultdict(list)
for fi,f in enumerate(faces):
    k=len(f)
    for i in range(k): f_edges[frozenset((f[i],f[(i+1)%k]))].append(fi)
nbr_cyc=[]
for fi,f in enumerate(faces):
    k=len(f); nb=[]
    for i in range(k):
        o=[x for x in f_edges[frozenset((f[i],f[(i+1)%k]))] if x!=fi]; nb.append(o[0] if o else -1)
    nbr_cyc.append(nb)
def wind(s,first,dr):
    vis=[False]*len(faces); S=[s]; vis[s]=True
    nx=nbr_cyc[s][first]
    if nx<0 or vis[nx]: return None
    S.append(nx); vis[nx]=True; prev=s; cur=nx
    while len(S)<len(faces):
        cyc=nbr_cyc[cur]; k=len(cyc)
        try: i=cyc.index(prev)
        except ValueError: return None
        nn=None
        for st in range(1,k+1):
            cand=cyc[(i+dr*st)%k]
            if cand>=0 and not vis[cand]: nn=cand; break
        if nn is None: return None
        S.append(nn); vis[nn]=True; prev=cur; cur=nn
    return S
rspi=None
for s in range(len(faces)):
    if len(faces[s])!=5: continue
    for first in range(len(nbr_cyc[s])):
        for dr in (1,-1):
            Sq=wind(s,first,dr)
            if Sq and len(Sq)==len(faces) and len(set(Sq))==len(faces):
                pos=[i+1 for i,fi in enumerate(Sq) if len(faces[fi])==5]
                if len(pos)==12: rspi=pos; break
        if rspi: break
    if rspi: break
NA=2*(n-2)
print("RSPI (C%d):"%NA, rspi)
np.savez("/tmp/c238.npz",carbons=carbons,bonds=np.array(bonds),
         pent=np.array([f for f in pent],dtype=object),hex=np.array([f for f in hexf],dtype=object),
         tris=np.array(tris),caps=V)
if rspi:
    with open("/Users/olson/Dev/fullerenes/mesh4a_C%d.inp"%NA,"w") as f:
        f.write("C%d from mesh4a field-aligned tiling\n"%NA)
        f.write("&General NA=%d iwext=1 /\n"%NA)
        f.write("&Coord ICart=3, rspi=%s /\n"%(",".join(map(str,rspi))))
        f.write("&FFChoice Iopt=4 /\n&FFParameters /\n&Hamilton /\n&Isomers /\n&Graph /\n")
    print("wrote mesh4a_C%d.inp"%NA)
