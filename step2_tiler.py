#!/usr/bin/env python3
"""Step 2: build a clean 12-pentagon capsomer topology on the mesh, with the 12
disclinations pinned at the charge-partition pentamer sites.

  place capsomers (12 pinned pentamers + H hexamers) at uniform spacing
  -> intrinsic (surface-Voronoi) triangulation
  -> anneal edge flips to a valid fullerene with pentamers AT the pinned sites.
"""
import numpy as np, sys, heapq, time
from scipy.spatial import cKDTree
from collections import defaultdict

H = int(sys.argv[1]) if len(sys.argv)>1 else 230
N = 12 + H

m=np.load("/tmp/mesh4a.npz"); V0,F0,cm=m['V'],m['F'],m['c']
pk=np.load("/tmp/pentamers.npz"); sites=pk['sites']

def subdivide(V,F):
    V=[tuple(x) for x in V]; idx={}
    def mid(a,b):
        k=(min(a,b),max(a,b))
        if k not in idx:
            idx[k]=len(V); V.append(tuple((np.array(V[a])+np.array(V[b]))/2))
        return idx[k]
    F2=[]
    for f in F:
        a,b,c=int(f[0]),int(f[1]),int(f[2]); ab=mid(a,b);bc=mid(b,c);ca=mid(c,a)
        F2+=[[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
    return np.array(V),np.array(F2)

Vs,Fs=subdivide(V0,F0); Vs,Fs=subdivide(Vs,Fs)     # ~75k faces: smooth Voronoi
area=0.0
for f in F0: area+=0.5*np.linalg.norm(np.cross(V0[f[1]]-V0[f[0]],V0[f[2]]-V0[f[0]]))
spacing=np.sqrt(2*area/(N*np.sqrt(3)))
print("H=%d N=%d spacing=%.1f meshV=%d"%(H,N,spacing,len(Vs)))

# ---- placement: pin 12 pentamers, farthest-point add hexamers, Lloyd ----
tree=cKDTree(Vs)
pin=[int(tree.query(s)[1]) for s in sites]          # nearest subdivided vertex to each site
caps=[Vs[i] for i in pin]
d=np.min(np.linalg.norm(Vs[:,None]-np.array(caps)[None],axis=2),axis=1)
while len(caps)<N:
    i=int(np.argmax(d)); caps.append(Vs[i]); d=np.minimum(d,np.linalg.norm(Vs-Vs[i],axis=1))
caps=np.array(caps)
def snap(P):                                         # keep capsomers on the surface (nearest mesh vertex)
    _,idx=tree.query(P); return Vs[idx]
for _ in range(40):                                  # Lloyd, pentamers pinned
    _,lab=cKDTree(caps).query(Vs); new=caps.copy()
    for j in range(N):
        sel=Vs[lab==j]
        if len(sel): new[j]=sel.mean(0)
    new[:12]=Vs[pin]; caps=snap(new); caps[:12]=Vs[pin]

# ---- intrinsic triangulation from surface-Voronoi (mesh faces w/ 3 labels) ----
_,lab=cKDTree(caps).query(Vs)
tris=set()
for f in Fs:
    l=(lab[f[0]],lab[f[1]],lab[f[2]])
    if l[0]!=l[1] and l[1]!=l[2] and l[0]!=l[2]: tris.add(tuple(sorted(l)))
tris=[list(t) for t in tris]
deg=np.zeros(N,int)
for t in tris:
    for v in t: deg[v]+=1
print("after CVT+Voronoi: tris=%d deg5=%d deg6=%d deg7+=%d deg<=4=%d  pinned@deg5=%d/12"%(
    len(tris),(deg==5).sum(),(deg==6).sum(),(deg>=7).sum(),(deg<=4).sum(),(deg[:12]==5).sum()))

# ---- disclination-aware annealer ----
class Tri:
    def __init__(self,F,n):
        self.n=n; self.F=[list(f) for f in F]; self.deg=np.zeros(n,int); self.em=defaultdict(set)
        for ti,f in enumerate(self.F):
            for v in f: self.deg[v]+=1
            for i in range(3):
                a,b=f[i],f[(i+1)%3]; self.em[(min(a,b),max(a,b))].add(ti)
    def energy(self):
        bad=int((self.deg<5).sum()+(self.deg>6).sum())          # structural
        mis=int((self.deg[:12]!=5).sum())+int((self.deg[12:]==5).sum())  # disclination placement
        return bad*3+mis
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

def anneal(tris,n,seconds=90):
    rng=np.random.default_rng(0); best=[list(t) for t in tris]; T=Tri(tris,n); bestE=T.energy()
    t0=time.time()
    while time.time()-t0<seconds and bestE>0:
        T=Tri([t[:] for t in best],n); temp=3.0
        for it in range(20000):
            E=T.energy()
            if E<bestE: bestE=E; best=[t[:] for t in T.F]
            if bestE==0: break
            targets=np.where((T.deg<5)|(T.deg>6))[0]
            mis=np.where((T.deg[:12]!=5))[0]
            pool=list(targets)+list(mis)+list(np.where(T.deg[12:]==5)[0]+12)
            if not pool: break
            dv=int(pool[rng.integers(len(pool))])
            inc=[e for e in T.em if dv in e and len(T.em[e])==2]
            if not inc: continue
            e=inc[rng.integers(len(inc))]; b0=E; r=T.flip(e)
            if not r: continue
            if not (T.energy()<=b0 or rng.random()<np.exp(-(T.energy()-b0)/max(temp,1e-6))):
                T.flip((min(r[2],r[3]),max(r[2],r[3])))
            temp*=0.9997
        if bestE==0: break
    return best,bestE

healed,E=anneal(tris,N,seconds=120)
Tf=Tri(healed,N)
print("after anneal: energy=%d  deg5=%d deg6=%d deg7+=%d deg<=4=%d  pinned@deg5=%d/12"%(
    E,(Tf.deg==5).sum(),(Tf.deg==6).sum(),(Tf.deg>=7).sum(),(Tf.deg<=4).sum(),(Tf.deg[:12]==5).sum()))
if E==0:
    np.savez("/tmp/topo_H%d.npz"%H,tris=np.array(healed),caps=caps,pin=np.array(pin),N=N,H=H)
    print("CLEAN fullerene topology saved -> /tmp/topo_H%d.npz"%H)
else:
    print("residual energy",E)
