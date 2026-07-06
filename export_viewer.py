#!/usr/bin/env python3
"""Extract fullerene faces and export JSON for the web viewer."""
import numpy as np, json
from scipy.spatial import cKDTree
from collections import defaultdict

def load_xyz(p):
    L=open(p).read().split('\n'); n=int(L[0].split()[0]); X=[]
    for ln in L[2:2+n]:
        t=ln.split(); X.append([float(t[1]),float(t[2]),float(t[3])])
    return np.array(X)

d=np.load("/tmp/fit_final.npz")
Xs=d['Xs']; bonds=[tuple(b) for b in d['bonds']]; cm=d['cm']
Xideal=d['Xideal_aligned']
m=np.load("/tmp/mesh4a.npz"); Vm,Fm=m['V'],m['F']

nadj=defaultdict(list)
for a,b in bonds: nadj[a].append(b); nadj[b].append(a)

# order neighbors CCW in tangent plane (normal = radial from centroid), using Xs
order={}
for v in range(len(Xs)):
    n=Xs[v]-cm; n/=np.linalg.norm(n)
    t=np.array([1.0,0,0]);
    if abs(n[0])>0.9: t=np.array([0,1.0,0])
    e1=t-n*np.dot(t,n); e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
    angs=[]
    for w in nadj[v]:
        dv=Xs[w]-Xs[v]; angs.append(np.arctan2(np.dot(dv,e2),np.dot(dv,e1)))
    order[v]=[w for _,w in sorted(zip(angs,nadj[v]))]

def next_edge(u,v):
    ring=order[v]; i=ring.index(u); return ring[(i+1)%len(ring)]

faces=[]; seen=set()
for a,b in bonds:
    for (u,v) in [(a,b),(b,a)]:
        if (u,v) in seen: continue
        face=[u]; cu,cv=u,v
        for _ in range(9):
            seen.add((cu,cv)); face.append(cv)
            w=next_edge(cu,cv); cu,cv=cv,w
            if cv==u and next_edge(cu,cv)==face[1]:
                seen.add((cu,cv)); break
        # dedup close
        if len(face)>=3: faces.append(face[:-1] if face[-1]==face[0] else face)

# keep unique faces by frozenset
uniq={}
for f in faces:
    k=frozenset(f)
    if k not in uniq and 4<=len(f)<=7: uniq[k]=f
faces=list(uniq.values())
pent=[f for f in faces if len(f)==5]
hexf=[f for f in faces if len(f)==6]
print("faces=%d pentagons=%d hexagons=%d"%(len(faces),len(pent),len(hexf)))

def center_scale(P):
    c=P.mean(0); s=np.abs(P-c).max(); return c,s
c0,s0=center_scale(Xs)
out={
  "meshV":[[float(x) for x in v] for v in (Vm)],
  "meshF":[[int(i) for i in f] for f in Fm],
  "atoms":[[float(x) for x in p] for p in Xs],
  "atomsIdeal":[[float(x) for x in p] for p in Xideal],
  "bonds":[[int(a),int(b)] for a,b in bonds],
  "pent":[[int(i) for i in f] for f in pent],
  "hex":[[int(i) for i in f] for f in hexf],
  "center":[float(x) for x in c0], "scale":float(s0),
}
json.dump(out,open("/Users/olson/mytestapp/viewer_data.json","w"))
print("wrote viewer_data.json  meshV=%d meshF=%d atoms=%d"%(len(Vm),len(Fm),len(Xs)))
