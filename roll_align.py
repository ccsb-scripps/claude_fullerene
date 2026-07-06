#!/usr/bin/env python3
"""Close the remaining fit gauge freedom: the PCA+scale fit leaves roll about the
long axis unconstrained. Optimize that 1 DOF so the cage pentamers register onto
the bowl-peak sites, then rewrite viewer_data + render_data for the overlay."""
import numpy as np, json
from collections import defaultdict
HEXDIA=95.0
p=np.load("/tmp/pentamers.npz"); sites=p['sites']; cm=p['cm']; axis=p['axis']/np.linalg.norm(p['axis'])
D=json.load(open("/Users/olson/mytestapp/viewer_data.json"))
atoms=np.array(D['atoms']); pent=[list(f) for f in D['pent']]; hexf=[list(f) for f in D['hex']]
bonds=[tuple(b) for b in D['bonds']]

def rot_about(P,ax,c,phi):
    v=P-c; k=ax
    return c+ v*np.cos(phi)+np.cross(k,v)*np.sin(phi)+np.outer((v@k)*(1-np.cos(phi)),k)
def perr(P):
    pc=np.array([P[f].mean(0) for f in pent])
    return np.linalg.norm(sites[:,None]-pc[None],axis=2).min(1)/HEXDIA

tS=(sites-cm)@axis; span=max(abs(tS).max(),1e-9); tSn=tS/span
base=perr(atoms)
phis=np.linspace(0,2*np.pi,721)
best=min(phis,key=lambda ph: perr(rot_about(atoms,axis,cm,ph)).mean())
Xw=rot_about(atoms,axis,cm,best)
pe=perr(Xw); belly=np.abs(tSn)<0.5
print("roll opt: phi=%.1f deg | pent_err %.2f->%.2f (belly %.2f->%.2f, tip %.2f->%.2f)"%(
    np.degrees(best),base.mean(),pe.mean(),base[belly].mean(),pe[belly].mean(),
    base[~belly].mean(),pe[~belly].mean()))

out=dict(D); out['atoms']=Xw.tolist(); out['atomsIdeal']=Xw.tolist()
out['center']=Xw.mean(0).tolist(); out['scale']=float(np.abs(Xw-Xw.mean(0)).max())
json.dump(out,open("/Users/olson/mytestapp/viewer_data.json","w"))

# render data for Blender: atoms, bonds, faces, bowl-peak sites
faces=pent+hexf
json.dump({"meshV":D['meshV'],"meshF":D['meshF'],"atoms":Xw.tolist(),
           "bonds":[[int(a),int(b)] for a,b in bonds],"peaks":sites.tolist(),
           "pent":[[int(i) for i in f] for f in pent],
           "hex":[[int(i) for i in f] for f in hexf]},
          open("/tmp/render_data.json","w"))
print("rewrote viewer_data.json + render_data.json (roll-aligned, bowl-anchored C%d)"%len(atoms))