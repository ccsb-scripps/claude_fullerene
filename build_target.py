#!/usr/bin/env python3
"""Heal an Instant Meshes tiling to a clean fullerene, build the dual cage,
relax to near-regular tiles (VECTORIZED FF), fit to mesh4a, export. Efficient."""
import sys, numpy as np, json, time
from scipy.spatial import cKDTree
from collections import defaultdict

IM = sys.argv[1] if len(sys.argv)>1 else "/tmp/im_480.obj"
V=[];T=[]
for ln in open(IM):
    p=ln.split()
    if not p: continue
    if p[0]=='v': V.append([float(x) for x in p[1:4]])
    elif p[0]=='f': T.append([int(x.split('/')[0])-1 for x in p[1:]])
V=np.array(V); n=len(V)

class Tri:
    def __init__(s,F):
        s.F=[list(t) for t in F]; s.deg=np.zeros(n,int); s.em=defaultdict(set); s.ve=defaultdict(set)
        for ti,f in enumerate(s.F):
            for v in f: s.deg[v]+=1
            for i in range(3):
                a,b=f[i],f[(i+1)%3]; k=(min(a,b),max(a,b)); s.em[k].add(ti); s.ve[a].add(k); s.ve[b].add(k)
    def E(s): return int((s.deg<5).sum()+(s.deg>6).sum())
    def flip(s,e):
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
        s.deg[a]-=1;s.deg[b]-=1;s.deg[c]+=1;s.deg[dd]+=1; return (a,b,c,dd)

t0=time.time(); rng=np.random.default_rng(0); best=[list(t) for t in T]; bestE=Tri(T).E()
sp=np.median([np.linalg.norm(V[a]-V[b]) for (a,b) in {(min(t[i],t[(i+1)%3]),max(t[i],t[(i+1)%3])) for t in T for i in range(3)}])
def attr(deg):
    hp=np.where(deg>6)[0]; pt=np.where(deg<6)[0]
    if len(hp)==0 or len(pt)==0: return 0.0
    D=np.linalg.norm(V[hp][:,None,:]-V[pt][None,:,:],axis=2)
    return float(D.min(axis=1).sum())/sp
def full(deg): return 1000*(int((deg<5).sum())+int((deg>6).sum())) + attr(deg)
print("initial defects:",bestE,flush=True)
for rs in range(80):
    tri=Tri([t[:] for t in best]); temp=6.0
    for it in range(25000):
        nbad=tri.E()
        if nbad<bestE: bestE=nbad; best=[t[:] for t in tri.F]
        if bestE==0: break
        defs=np.where((tri.deg<5)|(tri.deg>6))[0]
        if len(defs)==0: break
        cur=full(tri.deg)
        dv=int(defs[rng.integers(len(defs))]); inc=[e for e in tri.ve[dv] if len(tri.em[e])==2]
        if not inc: continue
        e=inc[rng.integers(len(inc))]; r=tri.flip(e)
        if not r: continue
        if not (full(tri.deg)<=cur or rng.random()<np.exp(-(full(tri.deg)-cur)/max(temp,1e-6))): tri.flip((min(r[2],r[3]),max(r[2],r[3])))
        temp*=0.9997
    if bestE==0: break
tris=[tuple(t) for t in best]; d=Tri(tris).deg
print("healed: %d capsomers valence %s (E=%d, %.0fs)"%(n,{k:int((d==k).sum()) for k in range(3,9) if (d==k).sum()},bestE,time.time()-t0),flush=True)

e2t=defaultdict(list)
for ti,t in enumerate(tris):
    for i in range(3):
        a,b=t[i],t[(i+1)%3]; e2t[(min(a,b),max(a,b))].append(ti)
bonds=sorted({(min(x),max(x)) for x in e2t.values() if len(x)==2})
X=np.array([V[list(t)].mean(0) for t in tris])
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
NA=len(X)
print("fullerene C%d: bonds=%d pent=%d hex=%d"%(NA,len(bonds),len(pent),len(hexf)),flush=True)

B=np.array(bonds)
angs=[]
nbr=defaultdict(list)
for a,b in bonds: nbr[a].append(b); nbr[b].append(a)
for i in range(NA):
    nb=nbr[i]
    for p in range(len(nb)):
        for q in range(p+1,len(nb)): angs.append((nb[p],i,nb[q]))
A=np.array(angs); th0=np.radians(120.0)
L0=np.median(np.linalg.norm(X[B[:,0]]-X[B[:,1]],axis=1))
for it in range(6000):
    Fc=np.zeros_like(X)
    e=X[B[:,0]]-X[B[:,1]]; L=np.linalg.norm(e,axis=1,keepdims=True)
    f=(L-L0)/L*e; np.add.at(Fc,B[:,0],-f); np.add.at(Fc,B[:,1],f)
    rij=X[A[:,0]]-X[A[:,1]]; rik=X[A[:,2]]-X[A[:,1]]
    lij=np.linalg.norm(rij,axis=1,keepdims=True); lik=np.linalg.norm(rik,axis=1,keepdims=True)
    c=np.clip(np.sum(rij*rik,1,keepdims=True)/(lij*lik),-1,1); th=np.arccos(c); ss=np.maximum(np.sin(th),1e-6)
    dE=(th-th0)
    dcj=rik/(lij*lik)-c*rij/lij**2; dck=rij/(lij*lik)-c*rik/lik**2
    gj=dE/ss*dcj; gk=dE/ss*dck
    np.add.at(Fc,A[:,0],gj); np.add.at(Fc,A[:,2],gk); np.add.at(Fc,A[:,1],-(gj+gk))
    X=X+0.05*Fc
bl=np.linalg.norm(X[B[:,0]]-X[B[:,1]],axis=1)
def adev(fs,ideal):
    r=[]
    for ff in fs:
        P=X[ff];k=len(ff)
        for i in range(k):
            a=P[(i-1)%k]-P[i];b=P[(i+1)%k]-P[i]
            r.append(abs(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))-ideal))
    return np.array(r)
ha=adev(hexf,120)
print("relaxed: bond CV %.2f%%  hex angleDev %.1f  within5deg %.0f%% (%.0fs)"%(bl.std()/bl.mean()*100,ha.mean(),np.mean(ha<5)*100,time.time()-t0),flush=True)

m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']
def pca(P): c=P.mean(0);u,s,vt=np.linalg.svd(P-c,full_matrices=False);return c,vt,s
cX,vtX,sX=pca(X); cM,vtM,sM=pca(Vm)
Xc=(X-cX)@vtX.T; Mc=(Vm-cM)@vtM.T
for k in range(3):
    if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
uni=(Mc.max(0)-Mc.min(0)).mean()/(Xc.max(0)-Xc.min(0)).mean()
Xw=(Xc*uni)@vtM+cM
dS,_=cKDTree(Vm).query(Xw); Rm=np.linalg.norm(Vm-cM,axis=1).mean()
print("fit: aspect %s (mesh %s) surf-dist mean %.1f%% max %.1f%%"%((sX/sX.max()).round(3),(sM/sM.max()).round(3),dS.mean()/Rm*100,dS.max()/Rm*100),flush=True)
out={"meshV":Vm.tolist(),"meshF":Fm.tolist(),"atoms":Xw.tolist(),"atomsIdeal":Xw.tolist(),
     "bonds":[[int(a),int(b)] for a,b in bonds],
     "pent":[[int(i) for i in f] for f in pent],"hex":[[int(i) for i in f] for f in hexf],
     "center":Xw.mean(0).tolist(),"scale":float(np.abs(Xw-Xw.mean(0)).max())}
json.dump(out,open("/Users/olson/mytestapp/viewer_data.json","w"))
with open("/Users/olson/Dev/fullerenes/mesh4a_C%d_regular.xyz"%NA,"w") as f:
    f.write("%d\nC%d regular fullerene fitted to mesh4a\n"%(NA,NA))
    for p in X: f.write("C %.4f %.4f %.4f\n"%(p[0],p[1],p[2]))
print("exported C%d"%NA,flush=True)
