#!/usr/bin/env python3
"""Robust field-guided tiling on a SUBDIVIDED (uniform) mesh.
Subdivide -> trivial-connections orientation field (singularities pinned at the
12 pentamers) -> hardened geodesic advancing front -> capsomer graph."""
import numpy as np, sys, scipy.sparse as sp, scipy.sparse.linalg as spl
from collections import defaultdict, deque
from scipy.spatial import cKDTree

H=int(sys.argv[1]) if len(sys.argv)>1 else 230
N=12+H
m=np.load("/tmp/mesh4a.npz"); V0=m['V']; F0=m['F']; cm=m['c']
pk=np.load("/tmp/pentamers.npz"); sites=pk['sites']

def subdivide(V,F):
    V=[tuple(x) for x in V]; idx={}
    def mid(a,b):
        k=(min(a,b),max(a,b))
        if k not in idx: idx[k]=len(V); V.append(tuple((np.array(V[a])+np.array(V[b]))/2))
        return idx[k]
    F2=[]
    for f in F:
        a,b,c=int(f[0]),int(f[1]),int(f[2]); ab=mid(a,b);bc=mid(b,c);ca=mid(c,a)
        F2+=[[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
    return np.array(V),np.array(F2)
V,F=subdivide(V0,F0)                       # ~19k faces, uniform
nv=len(V); nf=len(F)
area=sum(0.5*np.linalg.norm(np.cross(V0[f[1]]-V0[f[0]],V0[f[2]]-V0[f[0]])) for f in F0)
s=np.sqrt(2*area/(N*np.sqrt(3)))
el=np.mean([np.linalg.norm(V[f[0]]-V[f[1]]) for f in F])
print("H=%d N=%d s=%.1f  subdiv mesh: V=%d F=%d edge~%.1f (s=%.1f edges)"%(H,N,s,nv,nf,el,s/el))
pin=[int(cKDTree(V).query(x)[1]) for x in sites]

# --- trivial-connections field ---
Omega=np.full(nv,2*np.pi)
for f in F:
    for i in range(3):
        a=V[f[i]];b=V[f[(i+1)%3]];c=V[f[(i+2)%3]];u=b-a;w=c-a
        Omega[f[i]]-=np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)+1e-12),-1,1))
edges={}
def eid(a,b):
    k=(min(a,b),max(a,b))
    if k not in edges: edges[k]=len(edges)
    return edges[k]
for f in F:
    for i in range(3): eid(f[i],f[(i+1)%3])
elist=[None]*len(edges)
for k,i in edges.items(): elist[i]=k
index=np.zeros(nv)
for p in pin: index[p]=1/6
b=2*np.pi*index-Omega
r=[];c=[];v=[]
for (a,bb) in elist:
    r+=[a,bb,a,bb]; c+=[a,bb,bb,a]; v+=[1.,1.,-1.,-1.]
L=sp.csr_matrix((v,(r,c)),shape=(nv,nv))
y=spl.lsqr(L,b,atol=1e-11,btol=1e-11,iter_lim=30000)[0]
x=np.array([y[bb]-y[a] for (a,bb) in elist])

# per-face frame + transport
e1=np.zeros((nf,3));e2=np.zeros((nf,3));O=np.zeros((nf,3))
for fi,f in enumerate(F):
    a,bv,cc=V[f[0]],V[f[1]],V[f[2]];u=bv-a;n=np.cross(bv-a,cc-a);n/=np.linalg.norm(n)+1e-12
    u/=np.linalg.norm(u)+1e-12;e1[fi]=u;e2[fi]=np.cross(n,u);O[fi]=a
e2f=defaultdict(list)
for fi,f in enumerate(F):
    for i in range(3): e2f[(min(f[i],f[(i+1)%3]),max(f[i],f[(i+1)%3]))].append(fi)
def eang(fi,a,bv): d=V[bv]-V[a]; return np.arctan2(d@e2[fi],d@e1[fi])
trans={}
for (a,bv),fs in e2f.items():
    if len(fs)!=2: continue
    f,g=fs; ei=edges[(a,bv)]; af=eang(f,a,bv); ag=eang(g,a,bv)
    trans[(f,g)]=(ag-af)+x[ei]; trans[(g,f)]=(af-ag)-x[ei]

def to2d(fi,P): d=P-O[fi]; return np.array([d@e1[fi],d@e2[fi]])
def to3d(fi,p): return O[fi]+p[0]*e1[fi]+p[1]*e2[fi]
cent2d={}
def walk(f,P,theta,dist):
    skip=-1
    for _ in range(600):
        tv=[to2d(f,V[F[f][k]]) for k in range(3)]
        p=to2d(f,P); d=np.array([np.cos(theta),np.sin(theta)])
        best=None
        for k in range(3):
            if k==skip: continue
            A=tv[k];B=tv[(k+1)%3];ab=B-A
            den=-d[0]*ab[1]+d[1]*ab[0]
            if abs(den)<1e-12: continue
            rhs=A-p
            t=(-rhs[0]*ab[1]+rhs[1]*ab[0])/den; u=(d[0]*rhs[1]-d[1]*rhs[0])/den
            if t>1e-9 and -1e-6<=u<=1+1e-6 and (best is None or t<best[0]): best=(t,k,p+t*d)
        if best is None:
            return f,to3d(f,p+dist*d),theta
        t,k,q=best
        if t>=dist: return f,to3d(f,p+dist*d),theta
        a=F[f][k];bv=F[f][(k+1)%3];key=(min(a,bv),max(a,bv))
        nb=[g for g in e2f[key] if g!=f]
        q3=to3d(f,q)
        if not nb: return f,q3,theta
        g=nb[0]; theta=theta+trans[(f,g)]
        # nudge into g, and skip the edge we came through
        gc=(V[F[g][0]]+V[F[g][1]]+V[F[g][2]])/3
        P=q3+1e-3*(gc-q3); dist-=t; f=g
        # find which local edge of g is the shared one, to skip it next
        skip=-1
        for kk in range(3):
            if {F[g][kk],F[g][(kk+1)%3]}=={a,bv}: skip=kk;break
    return f,P,theta

# --- advancing front ---
caps=[]; cappos=[]; adj=defaultdict(set)
def add(P,f,th): caps.append((P,f,th)); cappos.append(P); return len(caps)-1
seedf=int(np.argmin(np.linalg.norm(V[F].mean(1)-cm,axis=1)))
add(V[F[seedf]].mean(0),seedf,0.0)
merge_r=0.5*s
front=deque([0]); guard=0
while front and guard<20000:
    guard+=1; ci=front.popleft(); P,f,th=caps[ci]
    tree=cKDTree(np.array(cappos))
    for k in range(6):
        gf,Q,th2=walk(f,P,th+k*np.pi/3,s)
        dd,idx=tree.query(Q,k=1)
        if dd<merge_r:
            j=int(idx)
            if j!=ci: adj[ci].add(j); adj[j].add(ci)
        else:
            j=add(Q,gf,th2-k*np.pi/3); adj[ci].add(j); adj[j].add(ci); front.append(j)
deg=np.array([len(adj[i]) for i in range(len(caps))])
hist={k:int((deg==k).sum()) for k in range(2,10) if (deg==k).sum()}
print("advancing front: capsomers=%d valence=%s (want ~{5:12,6:%d})"%(len(caps),hist,N-12))
np.savez("/tmp/field_tiling.npz",pos=np.array(cappos),
         adj=np.array([[i,j] for i in adj for j in adj[i] if i<j]),deg=deg,pin=np.array(pin),sites=sites)
print("saved")
