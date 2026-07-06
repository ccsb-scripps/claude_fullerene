#!/usr/bin/env python3
"""Regularity + bowl-consistency analysis of the equilateral C480."""
import numpy as np, json
from scipy.spatial import cKDTree

def load_xyz(p):
    L=open(p).read().split('\n'); n=int(L[0].split()[0]); X=[]
    for ln in L[2:2+n]:
        t=ln.split(); X.append([float(t[1]),float(t[2]),float(t[3])])
    return np.array(X)

X=load_xyz("/Users/olson/Dev/fullerenes/mesh4a_C480_ideal.xyz")
data=json.load(open("/Users/olson/mytestapp/viewer_data.json"))
bonds=[tuple(b) for b in data['bonds']]; pent=data['pent']; hexf=data['hex']
m=np.load("/tmp/mesh4a.npz"); Vm,cm_m=m['V'],m['c']
bw=np.load("/tmp/bowls.npz"); bowls=bw['pos']; hexdia=float(bw['hexdia'])

# ---- regularity of the equilateral cage ----
bl=np.array([np.linalg.norm(X[a]-X[b]) for a,b in bonds])
def poly_stats(faces,ideal_ang):
    edev=[]; adev=[]; plan=[]
    for f in faces:
        P=X[f]; k=len(f)
        e=[np.linalg.norm(P[i]-P[(i+1)%k]) for i in range(k)]
        edev.append((max(e)-min(e))/np.mean(e))
        for i in range(k):
            a=P[(i-1)%k]-P[i]; b=P[(i+1)%k]-P[i]
            ang=np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
            adev.append(abs(ang-ideal_ang))
        # planarity: distance of verts to best-fit plane / edge
        c=P.mean(0); u,s,vt=np.linalg.svd(P-c); nrm=vt[2]
        plan.append(np.max(np.abs((P-c)@nrm))/np.mean(e))
    return np.array(edev),np.array(adev),np.array(plan)
pe,pa,pp=poly_stats(pent,108.0)
he,ha,hp=poly_stats(hexf,120.0)
print("=== Equilateral C480 regularity ===")
print("all C-C bonds: mean %.4f  CV %.2f%%  (min %.3f max %.3f)"%(bl.mean(),bl.std()/bl.mean()*100,bl.min(),bl.max()))
print("pentagons: edge-spread %.1f%%  angle dev from 108 mean %.1f max %.1f  nonplanarity %.1f%%"%(
    pe.mean()*100,pa.mean(),pa.max(),pp.mean()*100))
print("hexagons:  edge-spread %.1f%%  angle dev from 120 mean %.1f max %.1f  nonplanarity %.1f%%"%(
    he.mean()*100,ha.mean(),ha.max(),hp.mean()*100))
frac_reg=np.mean(ha<5)
print("hexagons with all angles within 5 deg of 120: %.0f%%"%(frac_reg*100))

# ---- align cage to mesh, compare 12 pentagon centers to 12 bowls ----
pc=np.array([X[f].mean(0) for f in pent])          # pentagon centers (cage frame)
cX=X.mean(0)
uX,sX,vtX=np.linalg.svd(X-cX,full_matrices=False)
uM,sM,vtM=np.linalg.svd(Vm-cm_m,full_matrices=False)
# put both in PCA frame, sign-match by skew, scale by extents
Xc=(X-cX)@vtX.T; Mc=(Vm-cm_m)@vtM.T
sign=np.ones(3)
for k in range(3):
    if np.sign((Xc[:,k]**3).sum())!=np.sign((Mc[:,k]**3).sum()): sign[k]=-1
extX=(Xc*sign).max(0)-(Xc*sign).min(0); extM=Mc.max(0)-Mc.min(0)
sc=extM/extX
pcc=((pc-cX)@vtX.T*sign)*sc@vtM+cm_m               # pentagon centers in mesh world frame
tree=cKDTree(bowls)
d,idx=tree.query(pcc)
print("\n=== Pentagon vs curvature-bowl consistency (mesh frame) ===")
print("hexamer diameter = %.0f mesh units"%hexdia)
print("pentagon->nearest-bowl distance: mean %.0f max %.0f (in hexamer-diam units: mean %.2f max %.2f)"%(
    d.mean(),d.max(),d.mean()/hexdia,d.max()/hexdia))
within=np.sum(d<hexdia)
print("pentagons within ONE hexamer diameter of a bowl: %d/12"%within)
uniq=len(set(idx.tolist()))
print("distinct bowls matched: %d/12 (1-to-1 if 12)"%uniq)
