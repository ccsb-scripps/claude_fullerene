#!/usr/bin/env python3
"""Area-windowed fit for one mesh: search the physically-sized H window, heal
near-clean tilings, and report BOTH the best-overall defect-free fit (min pent_err)
and, when one exists, the best IPR fit (no adjacent pentamers). RSPIs via the
topology-based cage_rspi (program-independent).

Usage: python areafit.py <mesh.ply/stl> <name> [escalation_cap]
"""
import sys, os, json, subprocess, time, numpy as np
from scipy.spatial import cKDTree
from heal import heal
from cage_rspi import canonical_rspi
CWD="/Users/olson/Dev/fullerenes"; PY=CWD+"/.venv/bin/python"
IM=CWD+"/instant-meshes/InstantMeshesBatch"; MESH="/tmp/mesh4a.obj"
ply, name = sys.argv[1], sys.argv[2]; CAP=int(sys.argv[3]) if len(sys.argv)>3 else 200
os.makedirs("/tmp/af",exist_ok=True)
def sh(*a): return subprocess.run(list(a),cwd=CWD,capture_output=True,text=True)

sh(PY,"prep_mesh.py",ply); sh(PY,"principal_curvature.py"); sh(PY,"charge_partition_bowl.py")
est=json.load(open("/tmp/mesh_est.json")); Hest=est["H_est"]
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']; PEAKS=np.load("/tmp/pentamers.npz")['sites']
ar=0.5*np.linalg.norm(np.cross(Vm[Fm[:,1]]-Vm[Fm[:,0]],Vm[Fm[:,2]]-Vm[Fm[:,0]]),axis=1).sum()
HEX=float(np.sqrt(2*ar/(242*np.sqrt(3))))

def run_im(faces,out):
    for _ in range(4):
        try: os.remove(out)
        except OSError: pass
        try: subprocess.run([IM,MESH,"-o",out,"-r","6","-p","6","-f",str(faces)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=6)
        except subprocess.TimeoutExpired: subprocess.run(["pkill","-9","-f","InstantMeshesBatch"]); continue
        if os.path.exists(out): return True
    return False
def load_obj(fn):
    V=[];T=[]
    for ln in open(fn):
        p=ln.split()
        if not p: continue
        if p[0]=='v': V.append([float(x) for x in p[1:4]])
        elif p[0]=='f': T.append(tuple(int(x.split('/')[0])-1 for x in p[1:]))
    return np.array(V),T
def write_obj(fn,V,tris):
    with open(fn,"w") as f:
        for p in V: f.write("v %.6f %.6f %.6f\n"%(p[0],p[1],p[2]))
        for t in tris: f.write("f "+" ".join(str(i+1) for i in t)+"\n")
def metrics(V,tris):
    deg=np.zeros(len(V),int)
    for t in tris:
        for v in t: deg[v]+=1
    d5=np.where(deg==5)[0]; d5s=set(int(x) for x in d5)
    pe=float(np.linalg.norm(PEAKS[:,None]-V[d5][None],axis=2).min(1).mean())/HEX if len(d5) else 9.9
    adj=any(a in d5s and b in d5s for t in tris for a,b in ((t[i],t[(i+1)%3]) for i in range(3)))
    return int((deg<5).sum()+(deg>6).sum()), pe, (not adj)

cands=[]; ci=0; Hgrid=list(range(Hest-4,Hest+5))
subprocess.run(["pkill","-9","-f","InstantMeshesBatch"],stderr=subprocess.DEVNULL)
def gen(H):
    global ci
    of="/tmp/af/c_%d.obj"%ci; ci+=1
    if not run_im(2*H+20,of): return
    V,tris=load_obj(of)
    if len(V)==0: return
    dd,pe,ipr=metrics(V,tris)
    if 1<=dd<=6:
        healed,nd=heal(V,tris,budget=1.5)
        if nd==0: write_obj(of,V,healed); dd,pe,ipr=metrics(V,healed); tris=healed
    if dd==0: cands.append((pe,ipr,of,len(tris)))   # actual carbon count = #triangles
t0=time.time()
for H in Hgrid:
    for _ in range(6): gen(H)
tries=0
while (not cands or not any(c[1] for c in cands)) and tries<CAP and time.time()-t0<420:
    for H in Hgrid: gen(H); tries+=1
    if cands and any(c[1] for c in cands): break
Cest=est["C_est"]
# enforce physical size: keep candidates whose ACTUAL carbon count is near C_est
# (IM's -f target overshoots, so filter on the real C, not the nominal target)
inwin=[c for c in cands if abs(c[3]-Cest)<=8]
pool=inwin if inwin else sorted(cands,key=lambda c:abs(c[3]-Cest))[:max(3,len(cands)//3)]
print("%s: H_est=%d C_est=%d | %d defect-free cands (%d within +/-8 C of estimate, %d IPR) in %.0fs"%(
    name,Hest,Cest,len(cands),len(inwin),sum(c[1] for c in pool),time.time()-t0),flush=True)
if not cands:
    print("  NO defect-free fit found in window"); sys.exit()

def finalize(obj,track):
    json.dump({"representative":{"obj":obj,"H":0,"defects":0,"pent_err":0,"haus":0,"fit_mean":0}},open("/tmp/search_result.json","w"))
    fr=sh(PY,"finalize_rep.py"); sh(PY,"roll_align.py")
    import re; mf=re.search(r"fit: mean ([\d.]+)%",fr.stdout); fit=float(mf.group(1)) if mf else None
    D=json.load(open("/tmp/render_data.json"))
    faces=[list(f) for f in D['pent']]+[list(f) for f in D['hex']]
    pent=[set(f) for f in D['pent']]; np_adj=sum(1 for a in range(12) for b in range(a+1,12) if len(pent[a]&pent[b])>=2)
    r=canonical_rspi(faces)
    json.dump(D,open("/tmp/af/%s_%s.json"%(name,track),"w"))
    return len(D['atoms']),fit,np_adj,(" ".join(map(str,r)) if r else "NONE")

best=min(pool,key=lambda c:c[0])
C,fit,npair,rspi=finalize(best[2],"best")
print("  BEST FIT : C%d fit=%.1f%% pent_err=%.2f Np=%d (%s) RSPI=%s"%(
    C,fit,best[0],npair,"IPR" if npair==0 else "non-IPR",rspi))
ipr=[c for c in pool if c[1]]
if ipr:
    bi=min(ipr,key=lambda c:c[0])
    if bi[2]!=best[2]:
        C2,fit2,np2,rspi2=finalize(bi[2],"ipr")
        print("  IPR  FIT : C%d fit=%.1f%% pent_err=%.2f Np=%d RSPI=%s"%(C2,fit2,bi[0],np2,rspi2))
    else:
        print("  (best fit is already IPR)")
else:
    print("  no IPR (no-adjacent-pentamer) solution found in window")
