#!/usr/bin/env python3
"""Self-consistency discriminator experiment.

Generate N stochastic pipeline results for one mesh; for each, rebuild the exact
isomer from its canonical RSPI (Fullerene program, ICart=2), and measure the
geometric discrepancy between:
  (proc)  the procedure's FF-relaxed cage  vs
  (rspi)  the program's ideal free geometry of the same isomer.
Discrepancy = similarity-aligned two-sided Chamfer distance (scale/rotation/
reflection-invariant), normalized by cage size. Hypothesis: the smallest
discrepancy marks the isomer that sits most naturally at the target shape.

Usage: python selfconsistency.py <mesh.ply> [N]   -> /tmp/sc/results.json
"""
import numpy as np, subprocess, json, os, sys, re
from scipy.spatial import cKDTree
CWD="/Users/olson/Dev/fullerenes"; PY=[CWD+"/.venv/bin/python","-u"]
PLY=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 10
CAP=int(sys.argv[3]) if len(sys.argv)>3 else 200      # full_search escalation cap
os.makedirs("/tmp/sc",exist_ok=True)
def run(cmd): return subprocess.run(cmd,cwd=CWD,capture_output=True,text=True)
def read_xyz(fn):
    P=[]
    for ln in open(fn).read().split('\n')[2:]:
        t=ln.split()
        if len(t)>=4: P.append([float(t[1]),float(t[2]),float(t[3])])
    return np.array(P)
def prog(inp,out="/tmp/sc_prog.xyz"):
    open("/tmp/sc_prog.inp","w").write(inp)
    run(["./run_fullerene.sh","/tmp/sc_prog.inp",out])
    return open("/tmp/fullerene_run.log").read()
def parse_rspi(log,NA):
    for l in log.splitlines():
        t=l.split()
        if len(t)>=15 and 'x' in t and t[13]==str(NA):
            try: return [int(x) for x in t[1:13]]
            except ValueError: pass
    return None
def parse_np(log):
    m=re.search(r"pentagon signature\)\s*Np:\s*(\d+)",log); return int(m.group(1)) if m else -1

def align(proc,rcage):
    """scale+rotate+reflect rcage onto proc; return (norm_chamfer, rcage_in_proc_frame)."""
    pc=proc.mean(0); Ap=proc-pc; sp=np.sqrt((Ap**2).sum(1).mean())
    rc=rcage.mean(0); Br=(rcage-rc); sr=np.sqrt((Br**2).sum(1).mean()); Br=Br*(sp/sr)
    _,_,va=np.linalg.svd(Ap,full_matrices=False); A=Ap@va.T
    _,_,vb=np.linalg.svd(Br,full_matrices=False); B0=Br@vb.T
    treeA=cKDTree(A); best=(1e18,None)
    for sx in(1,-1):
        for sy in(1,-1):
            for sz in(1,-1):
                B=B0*np.array([sx,sy,sz],float)
                for _ in range(25):
                    _,idx=treeA.query(B); Bc=B.mean(0); Ac=A[idx].mean(0)
                    H=(B-Bc).T@(A[idx]-Ac); U,S,Vt=np.linalg.svd(H)
                    Dg=np.eye(3); Dg[2,2]=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@Dg@U.T
                    B=(B-Bc)@R.T+Ac
                ch=0.5*(treeA.query(B)[0].mean()+cKDTree(B).query(A)[0].mean())
                if ch<best[0]: best=(ch,B.copy())
    ch,B=best
    return ch/sp, B@va+pc          # normalized discrepancy, rcage mapped into proc frame

# deterministic prep once
run(PY+["prep_mesh.py",PLY]); run(PY+["principal_curvature.py"]); run(PY+["charge_partition_bowl.py"])
results=[]
for i in range(1,N+1):
    fs=run(PY+["full_search.py",str(CAP)]); fr=run(PY+["finalize_rep.py"])
    mpe=re.search(r"pent_err=([\d.]+)",fs.stdout); pe=float(mpe.group(1)) if mpe else None
    mf =re.search(r"fullerene C(\d+): pentagons=(\d+) hexagons=\d+ defects=(\d+)",fr.stdout)
    NA=int(mf.group(1)); npent=int(mf.group(2)); defc=int(mf.group(3))
    mfit=re.search(r"fit: mean ([\d.]+)%",fr.stdout); fit=float(mfit.group(1)) if mfit else None
    proc=read_xyz(CWD+"/mesh4a_C%d_representative.xyz"%NA)
    # canonical RSPI from proc coords
    dd,_=cKDTree(proc).query(proc,k=2); L=np.median(dd[:,1]); Xs=(proc-proc.mean(0))*(1.44/L)
    log=prog("c\n&General NA=%d iwext=1 ispsearch=2 /\n&Coord ICart=1 /\n&FFChoice Iopt=4 /\n&FFParameters /\n&Hamilton /\n&Isomers /\n&Graph /\n"%NA
             +"".join("6 %.5f %.5f %.5f\n"%tuple(p) for p in Xs))
    rspi=parse_rspi(log,NA)
    entry=dict(i=i,C=NA,defects=defc,pentagons=npent,pent_err=pe,fit=fit,rspi=rspi,Np=None,disc=None)
    if rspi and defc==0 and npent==12:
        blog=prog("b\n&General NA=%d IP=1 iwext=1 ispsearch=2 /\n&Coord ICart=2 rspi=%s/\n&FFChoice Iopt=3 /\n&FFParameters /\n&Hamilton /\n&Isomers /\n&Graph /\n"%(NA,",".join(map(str,rspi))),
                  "/tmp/sc/rspi_%d.xyz"%i)
        entry['Np']=parse_np(blog)
        rcage=read_xyz("/tmp/sc/rspi_%d.xyz"%i)
        disc,rspi_in_proc=align(proc,rcage)
        entry['disc']=round(float(disc)*100,3)          # % of cage size
        entry['proc']=proc.tolist(); entry['rspi_cage']=rspi_in_proc.tolist()
    results.append({k:entry[k] for k in entry if k not in('proc','rspi_cage')} |
                   ({'proc':entry.get('proc'),'rspi_cage':entry.get('rspi_cage')}))
    print("run %2d: C%d def=%d Np=%s pent_err=%s fit=%s%% disc=%s"%(
        i,NA,defc,entry['Np'],pe,fit,entry['disc']),flush=True)
ok=[r for r in results if r['disc'] is not None]; ok.sort(key=lambda r:r['disc'])
print("\n=== ranked by self-consistency discrepancy (smallest = best) ===")
for r in ok:
    print(" C%d Np=%d disc=%.3f%% | pent_err=%.3f fit=%.2f%% RSPI=%s"%(
        r['C'],r['Np'],r['disc'],r['pent_err'],r['fit'],r['rspi']))
json.dump(results,open("/tmp/sc/results.json","w"))
print("\nsaved /tmp/sc/results.json (%d with discrepancy)"%len(ok))
