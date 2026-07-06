#!/usr/bin/env python3
"""Run the current bowl-anchored procedure N times on mesh4a to characterize
run-to-run variation. Placement (charge_partition_bowl) is deterministic; the
variation source is Instant Meshes (random init + threading) inside full_search.
Each run: full_search -> finalize_rep -> roll_align, then capture the roll-aligned
cage + metrics into /tmp/ensemble/run_i.json and append a summary row."""
import numpy as np, json, subprocess, time, os
from scipy.spatial import cKDTree
os.makedirs("/tmp/ensemble",exist_ok=True)
HEXDIA=95.0
p=np.load("/tmp/pentamers.npz"); sites=p['sites']; cm=p['cm']; axis=p['axis']/np.linalg.norm(p['axis'])
m=np.load("/tmp/mesh4a.npz"); Vm=m['V']; Fm=m['F']
Rm=np.linalg.norm(Vm-Vm.mean(0),axis=1).mean(); mtree=cKDTree(Vm)
tS=(sites-cm)@axis; span=max(abs(tS).max(),1e-9); tSn=tS/span; belly=np.abs(tSn)<0.5
N=int(os.environ.get("NRUNS","10"))
def run(cmd):
    r=subprocess.run(cmd,cwd="/Users/olson/Dev/fullerenes",
                     capture_output=True,text=True,
                     env={**os.environ})
    return r.returncode
PY=["/Users/olson/Dev/fullerenes/.venv/bin/python","-u"]
summary=[]
t0=time.time()
for i in range(1,N+1):
    s=time.time()
    run(PY+["full_search.py"]); run(PY+["finalize_rep.py"]); run(PY+["roll_align.py"])
    D=json.load(open("/Users/olson/mytestapp/viewer_data.json"))
    atoms=np.array(D['atoms']); pent=[list(f) for f in D['pent']]; hexf=[list(f) for f in D['hex']]
    defect=[list(f) for f in D.get('defect',[])]
    NA=len(atoms); H=len(hexf)
    # fit metrics on the captured (roll-aligned) cage
    dS,_=mtree.query(atoms); ct=cKDTree(atoms); dM,_=ct.query(Vm)
    fit_mean=dS.mean()/Rm*100; haus=max(np.percentile(dS,95),np.percentile(dM,95))/Rm*100
    # pentamer placement vs bowl peaks
    pc=np.array([atoms[f].mean(0) for f in pent])
    pe=np.linalg.norm(sites[:,None]-pc[None],axis=2).min(1)/HEXDIA
    _,sv,_=np.linalg.svd(atoms-atoms.mean(0),full_matrices=False); aspect=(sv/sv.max()).round(3).tolist()
    # isolated-pentagon check (no two pentamers share an edge/atom)
    pset=[set(f) for f in pent]; ipr=all(len(pset[a]&pset[b])==0 for a in range(12) for b in range(a+1,12))
    rec=dict(run=i,C=NA,H=H,pentagons=len(pent),defects=len(defect),ipr=bool(ipr),
             fit_mean=round(fit_mean,2),haus=round(haus,2),
             pent_err=round(float(pe.mean()),3),pent_err_belly=round(float(pe[belly].mean()),3),
             pent_err_tip=round(float(pe[~belly].mean()),3),aspect=aspect)
    summary.append(rec)
    json.dump(dict(atoms=atoms.tolist(),bonds=[[int(a),int(b)] for a,b in D['bonds']],
                   pent=[[int(x) for x in f] for f in pent],hex=[[int(x) for x in f] for f in hexf],
                   metrics=rec),open("/tmp/ensemble/run_%d.json"%i,"w"))
    print("run %2d: C%-4d H=%d pent=%d def=%d IPR=%s fit=%.2f%% Haus=%.2f%% pent_err=%.3f (belly %.3f) aspect=%s [%.0fs]"%(
        i,NA,H,len(pent),len(defect),ipr,fit_mean,haus,pe.mean(),pe[belly].mean(),aspect,time.time()-s),flush=True)
json.dump({"mesh_c":Vm.mean(0).tolist(),"sites":sites.tolist(),"runs":summary,
           "meshV":Vm.tolist(),"meshF":[[int(x) for x in f] for f in Fm]},
          open("/tmp/ensemble/summary.json","w"))
# variation report
C=[r['C'] for r in summary]; fit=[r['fit_mean'] for r in summary]; pe=[r['pent_err'] for r in summary]
print("\n=== VARIATION over %d runs (%.0fs total) ==="%(N,time.time()-t0))
print("C atoms   : %s  (range %d-%d)"%(sorted(C),min(C),max(C)))
print("fit_mean  : %.2f-%.2f%% (mean %.2f, std %.2f)"%(min(fit),max(fit),np.mean(fit),np.std(fit)))
print("pent_err  : %.3f-%.3f  (mean %.3f, std %.3f)"%(min(pe),max(pe),np.mean(pe),np.std(pe)))
print("all defect-free & IPR: %s"%all(r['defects']==0 and r['ipr'] for r in summary))
print("saved /tmp/ensemble/{run_1..%d,summary}.json"%N)
