#!/usr/bin/env python3
"""Build the target-size cage from the raw IM tiling (defects kept & marked),
FF-relax, fit to mesh4a, export for the viewer."""
import numpy as np, json
from scipy.spatial import cKDTree
from collections import defaultdict
IM="/tmp/im_best.obj"
V=[];T=[]
for ln in open(IM):
    p=ln.split()
    if not p: continue
    if p[0]=='v': V.append([float(x) for x in p[1:4]])
    elif p[0]=='f': T.append([int(x.split('/')[0])-1 for x in p[1:]])
V=np.array(V); n=len(V); tris=[tuple(t) for t in T]
deg=np.zeros(n,int)
for t in tris:
    for v in t: deg[v]+=1
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
defect=[f for f in faces if len(f) not in (5,6)]
NA=len(X)
print("C%d cage: pent=%d hex=%d defect=%d"%(NA,len(pent),len(hexf),len(defect)))
# vectorized FF relax
B=np.array(bonds); nbr=defaultdict(list)
for a,b in bonds: nbr[a].append(b); nbr[b].append(a)
angs=[]
for i in range(NA):
    nb=nbr[i]
    for p_ in range(len(nb)):
        for q in range(p_+1,len(nb)): angs.append((nb[p_],i,nb[q]))
A=np.array(angs); th0=np.radians(120.0)
L0=np.median(np.linalg.norm(X[B[:,0]]-X[B[:,1]],axis=1))
for it in range(6000):
    Fc=np.zeros_like(X)
    e=X[B[:,0]]-X[B[:,1]]; L=np.linalg.norm(e,axis=1,keepdims=True)
    f=(L-L0)/L*e; np.add.at(Fc,B[:,0],-f); np.add.at(Fc,B[:,1],f)
    rij=X[A[:,0]]-X[A[:,1]]; rik=X[A[:,2]]-X[A[:,1]]
    lij=np.linalg.norm(rij,axis=1,keepdims=True); lik=np.linalg.norm(rik,axis=1,keepdims=True)
    c=np.clip(np.sum(rij*rik,1,keepdims=True)/(lij*lik),-1,1); th=np.arccos(c); ss=np.maximum(np.sin(th),1e-6)
    dE=(th-th0); dcj=rik/(lij*lik)-c*rij/lij**2; dck=rij/(lij*lik)-c*rik/lik**2
    gj=dE/ss*dcj; gk=dE/ss*dck
    np.add.at(Fc,A[:,0],gj); np.add.at(Fc,A[:,2],gk); np.add.at(Fc,A[:,1],-(gj+gk))
    X=X+0.05*Fc
bl=np.linalg.norm(X[B[:,0]]-X[B[:,1]],axis=1)
print("relaxed bond CV %.2f%%"%(bl.std()/bl.mean()*100))
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']
def pca(P): c=P.mean(0);u,s,vt=np.linalg.svd(P-c,full_matrices=False);return c,vt,s
cX,vtX,sX=pca(X); cM,vtM,sM=pca(Vm)
Xc=(X-cX)@vtX.T; Mc=(Vm-cM)@vtM.T
for k in range(3):
    if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): Xc[:,k]*=-1
uni=(Mc.max(0)-Mc.min(0)).mean()/(Xc.max(0)-Xc.min(0)).mean()
Xw=(Xc*uni)@vtM+cM
dS,_=cKDTree(Vm).query(Xw); Rm=np.linalg.norm(Vm-cM,axis=1).mean()
print("fit surf-dist mean %.1f%% max %.1f%% ; aspect %s vs mesh %s"%(dS.mean()/Rm*100,dS.max()/Rm*100,(sX/sX.max()).round(3),(sM/sM.max()).round(3)))
out={"meshV":Vm.tolist(),"meshF":Fm.tolist(),"atoms":Xw.tolist(),"atomsIdeal":Xw.tolist(),
     "bonds":[[int(a),int(b)] for a,b in bonds],
     "pent":[[int(i) for i in f] for f in pent],"hex":[[int(i) for i in f] for f in hexf],
     "defect":[[int(i) for i in f] for f in defect],
     "center":Xw.mean(0).tolist(),"scale":float(np.abs(Xw-Xw.mean(0)).max())}
json.dump(out,open("/Users/olson/mytestapp/viewer_data.json","w"))
print("exported C%d (with %d defect faces marked)"%(NA,len(defect)))
