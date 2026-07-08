#!/usr/bin/env python3
"""Register each self-consistency result's procedure cage onto the (cage_4) mesh,
carrying its paired RSPI cage through the SAME transform, so the Blender overlay
can show mesh + curvature markers + both cages together. Writes /tmp/sc/registered.json.
"""
import json, numpy as np
from scipy.spatial import cKDTree

def topology(X):
    n=len(X); tr=cKDTree(X); d,_=tr.query(X,k=4); L=np.median(d[:,1:4])
    pairs=sorted(tr.query_pairs(1.35*L)); adj=[[] for _ in range(n)]
    for a,b in pairs: adj[a].append(b); adj[b].append(a)
    ctr=X.mean(0); oadj=[]
    for i in range(n):
        nrm=X[i]-ctr; nrm/=np.linalg.norm(nrm)+1e-9
        t=np.array([1.,0,0]) if abs(nrm[0])<0.9 else np.array([0,1.,0])
        e1=t-(t@nrm)*nrm; e1/=np.linalg.norm(e1); e2=np.cross(nrm,e1)
        nb=adj[i]; ang=[np.arctan2((X[j]-X[i])@e2,(X[j]-X[i])@e1) for j in nb]
        oadj.append([nb[k] for k in np.argsort(ang)])
    def tr_faces(step):
        seen=set(); faces=[]
        for u in range(n):
            for v in adj[u]:
                if (u,v) in seen: continue
                f=[]; a,b=u,v
                for _ in range(9):
                    seen.add((a,b)); f.append(a); nb=oadj[b]; a,b=b,nb[(nb.index(a)+step)%len(nb)]
                    if (a,b)==(u,v): break
                faces.append(f)
        return faces
    faces=tr_faces(1)
    if sum(len(f)==5 for f in faces)!=12: faces=tr_faces(-1)
    return pairs,[f for f in faces if len(f)==5]

m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']
pk=np.load("/tmp/pentamers.npz"); sites=pk['sites']; cm=pk['cm']; axis=pk['axis']/np.linalg.norm(pk['axis'])
cM=Vm.mean(0); _,_,vtM=np.linalg.svd(Vm-cM,full_matrices=False); Mc=(Vm-cM)@vtM.T; mtree=cKDTree(Vm)
ar=0.5*np.linalg.norm(np.cross(Vm[Fm[:,1]]-Vm[Fm[:,0]],Vm[Fm[:,2]]-Vm[Fm[:,0]]),axis=1).sum()
HEX=float(np.sqrt(2*ar/(242*np.sqrt(3))))
tmp=np.array([1.,0,0]) if abs(axis[0])<0.9 else np.array([0,1.,0])
e1=tmp-(tmp@axis)*axis; e1/=np.linalg.norm(e1); e2=np.cross(axis,e1)

R=[r for r in json.load(open("/tmp/sc/results.json")) if r.get('disc') is not None]
R.sort(key=lambda r:r['disc'])
out=[]
for k,r in enumerate(R):
    proc=np.array(r['proc']); rspi=np.array(r['rspi_cage'])
    bonds,pent=topology(proc)
    # stage 1: PCA + uniform scale + orientation search (min surface dist)
    cP=proc.mean(0); _,_,vt=np.linalg.svd(proc-cP,full_matrices=False)
    best=None
    for swap in ([0,1,2],[0,2,1]):
        for s in [(a,b,c) for a in(1,-1) for b in(1,-1) for c in(1,-1)]:
            Xt=((proc-cP)@vt.T)[:,swap]*np.array(s)
            uni=(Mc.max(0)-Mc.min(0)).mean()/(Xt.max(0)-Xt.min(0)).mean()
            Xw=(Xt*uni)@vtM+cM; sc=mtree.query(Xw)[0].mean()
            if best is None or sc<best[0]: best=(sc,swap,list(s),uni)
    _,swap,s,uni=best
    def stage1(P): return (((P-cP)@vt.T)[:,swap]*np.array(s)*uni)@vtM+cM
    P1=stage1(proc); Q1=stage1(rspi)
    # stage 2: enantiomer roll registering proc pentamers to bowl sites
    def place(P,refl,phi):
        v=P-cm; al=(v@axis)[:,None]*axis; a=v@e1; b=(v@e2)*refl; ca,sa=np.cos(phi),np.sin(phi)
        return cm+al+(a*ca-b*sa)[:,None]*e1+(a*sa+b*ca)[:,None]*e2
    def perr(P):
        pc=np.array([P[f].mean(0) for f in pent]); return np.linalg.norm(sites[:,None]-pc[None],axis=2).min(1).mean()/HEX
    br=None
    for refl in (1,-1):
        for phi in np.linspace(0,2*np.pi,361):
            e=perr(place(P1,refl,phi))
            if br is None or e<br[0]: br=(e,refl,phi)
    _,refl,phi=br
    Pr=place(P1,refl,phi); Qr=place(Q1,refl,phi)
    out.append(dict(rank=k+1,C=r['C'],disc=r['disc'],pent_err=round(float(br[0]),3),
                    proc=Pr.tolist(),rspi=Qr.tolist(),
                    bonds_proc=[[int(a),int(b)] for a,b in bonds]))
    print("rank%02d C%d disc=%.2f%% registered (pent_err %.2f)"%(k+1,r['C'],r['disc'],br[0]))
json.dump({"meshV":Vm.tolist(),"meshF":[[int(x) for x in f] for f in Fm],
           "sites":sites.tolist(),"results":out}, open("/tmp/sc/registered.json","w"))
print("wrote /tmp/sc/registered.json (%d results, mesh %d verts)"%(len(out),len(Vm)))
