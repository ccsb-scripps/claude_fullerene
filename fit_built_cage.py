#!/usr/bin/env python3
"""Fit a KNOWN fullerene cage (e.g. built from an RSPI by the Fullerene program)
onto the current mesh, deterministically -- no stochastic remeshing. Derives the
cage's bonds + pentagon faces, does the same PCA+scale orientation-resolved fit as
finalize_rep, then the enantiomer-aware roll registration, and writes
/tmp/render_data.json for the Blender overlay.

Usage:  python fit_built_cage.py <cage.xyz>
Reads /tmp/mesh4a.npz (mesh) and /tmp/pentamers.npz (bowl-peak sites + axis).
"""
import sys, json, numpy as np
from scipy.spatial import cKDTree

def load_xyz(fn):
    P=[]
    for ln in open(fn).read().split('\n')[2:]:
        t=ln.split()
        if len(t)>=4: P.append([float(t[1]),float(t[2]),float(t[3])])
    return np.array(P)

def topology(X):
    """bonds + pentagon/hexagon faces of a 3-regular fullerene cage from coords."""
    n=len(X); tr=cKDTree(X); d,_=tr.query(X,k=4); L=np.median(d[:,1:4])
    pairs=sorted(tr.query_pairs(1.35*L))
    adj=[[] for _ in range(n)]
    for a,b in pairs: adj[a].append(b); adj[b].append(a)
    ctr=X.mean(0); oadj=[]
    for i in range(n):                                   # order neighbors CCW about outward normal
        nrm=X[i]-ctr; nrm/=np.linalg.norm(nrm)+1e-9
        t=np.array([1.,0,0]) if abs(nrm[0])<0.9 else np.array([0,1.,0])
        e1=t-(t@nrm)*nrm; e1/=np.linalg.norm(e1); e2=np.cross(nrm,e1)
        nb=adj[i]; ang=[np.arctan2((X[j]-X[i])@e2,(X[j]-X[i])@e1) for j in nb]
        oadj.append([nb[k] for k in np.argsort(ang)])
    def trace(step):
        seen=set(); faces=[]
        for u in range(n):
            for v in adj[u]:
                if (u,v) in seen: continue
                face=[]; a,b=u,v
                for _ in range(9):
                    seen.add((a,b)); face.append(a)
                    nb=oadj[b]; a,b=b,nb[(nb.index(a)+step)%len(nb)]
                    if (a,b)==(u,v): break
                faces.append(face)
        return faces
    faces=trace(1)
    if sum(len(f)==5 for f in faces)!=12: faces=trace(-1)   # pick consistent orientation
    pent=[f for f in faces if len(f)==5]; hexf=[f for f in faces if len(f)==6]
    return pairs, pent, hexf

X0=load_xyz(sys.argv[1])
bonds,pent,hexf=topology(X0)
print("built cage: %d atoms, %d bonds, %d pentagons, %d hexagons"%(len(X0),len(bonds),len(pent),len(hexf)))
assert len(pent)==12, "expected 12 pentagons, got %d (topology detection failed)"%len(pent)

m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']
pk=np.load("/tmp/pentamers.npz"); sites=pk['sites']; cm=pk['cm']; axis=pk['axis']/np.linalg.norm(pk['axis'])
cM=Vm.mean(0); _,_,vtM=np.linalg.svd(Vm-cM,full_matrices=False); Mc=(Vm-cM)@vtM.T
Rm=np.linalg.norm(Vm-cM,axis=1).mean(); mtree=cKDTree(Vm)
ar=0.5*np.linalg.norm(np.cross(Vm[Fm[:,1]]-Vm[Fm[:,0]],Vm[Fm[:,2]]-Vm[Fm[:,0]]),axis=1).sum()
HEX=float(np.sqrt(2*ar/(242*np.sqrt(3))))

# PCA + uniform scale + orientation-resolved fit (as finalize_rep)
cX=X0.mean(0); _,_,vt=np.linalg.svd(X0-cX,full_matrices=False); Xc=(X0-cX)@vt.T
best=None
for swap in ([0,1,2],[0,2,1]):
    for s in [(a,b,c) for a in(1,-1) for b in(1,-1) for c in(1,-1)]:
        Xt=Xc[:,swap]*np.array(s)
        uni=(Mc.max(0)-Mc.min(0)).mean()/(Xt.max(0)-Xt.min(0)).mean()
        Xw=(Xt*uni)@vtM+cM; sc=mtree.query(Xw)[0].mean()
        if best is None or sc<best[0]: best=(sc,Xw)
Xw=best[1]

# enantiomer-aware roll registration to bowl peaks (as roll_align)
tmp=np.array([1.,0,0]) if abs(axis[0])<0.9 else np.array([0,1.,0])
e1=tmp-(tmp@axis)*axis; e1/=np.linalg.norm(e1); e2=np.cross(axis,e1)
def place(P,refl,phi):
    v=P-cm; along=(v@axis)[:,None]*axis; a=v@e1; b=(v@e2)*refl; ca,sa=np.cos(phi),np.sin(phi)
    return cm+along+(a*ca-b*sa)[:,None]*e1+(a*sa+b*ca)[:,None]*e2
def perr(P):
    pc=np.array([P[f].mean(0) for f in pent]); return np.linalg.norm(sites[:,None]-pc[None],axis=2).min(1)/HEX
phis=np.linspace(0,2*np.pi,721); br=None
for refl in (1,-1):
    for phi in phis:
        e=perr(place(Xw,refl,phi)).mean()
        if br is None or e<br[0]: br=(e,refl,phi)
Xw=place(Xw,br[1],br[2])
dS,_=mtree.query(Xw)
print("fit: surface mean %.1f%% | pent_err %.2f (%s)"%(dS.mean()/Rm*100, perr(Xw).mean(), "mirror" if br[1]<0 else "rot"))

json.dump({"meshV":Vm.tolist(),"meshF":[[int(x) for x in f] for f in Fm],"atoms":Xw.tolist(),
           "bonds":[[int(a),int(b)] for a,b in bonds],"peaks":sites.tolist(),
           "pent":[[int(i) for i in f] for f in pent],"hex":[[int(i) for i in f] for f in hexf]},
          open("/tmp/render_data.json","w"))
print("wrote /tmp/render_data.json (C%d, exact isomer fitted to mesh)"%len(X0))
