#!/usr/bin/env python3
"""Close the remaining fit gauge freedom about the long axis so the cage pentamers
register onto the bowl-peak sites. The PCA+scale fit fixes the axes only up to an
improper rotation about the long axis, i.e. a roll AND a reflection (choice of
enantiomer -- which finalize can't distinguish by surface fit, since a fullerene
and its mirror fit a surface equally well). We optimize both: pick the enantiomer
and roll that minimize pentamer->bowl-marker distance. Reflecting to the mirror
image does NOT change the topology/RSPI (a graph and its mirror are the same
graph); it only improves the geometric registration (matters most at a wide end,
where a small azimuthal error becomes a large gap). Rewrites viewer/render data."""
import numpy as np, json
p=np.load("/tmp/pentamers.npz"); sites=p['sites']; cm=p['cm']; axis=p['axis']/np.linalg.norm(p['axis'])
D=json.load(open("/Users/olson/mytestapp/viewer_data.json"))
atoms=np.array(D['atoms']); pent=[list(f) for f in D['pent']]; hexf=[list(f) for f in D['hex']]
bonds=[tuple(b) for b in D['bonds']]
mV=np.array(D['meshV']); mF=np.array(D['meshF'])
ar=0.5*np.linalg.norm(np.cross(mV[mF[:,1]]-mV[mF[:,0]],mV[mF[:,2]]-mV[mF[:,0]]),axis=1).sum()
HEXDIA=float(np.sqrt(2*ar/(242*np.sqrt(3))))            # hexamer spacing from mesh area

# orthonormal basis perpendicular to the long axis
tmp=np.array([1.0,0,0]) if abs(axis[0])<0.9 else np.array([0,1.0,0])
e1=tmp-(tmp@axis)*axis; e1/=np.linalg.norm(e1); e2=np.cross(axis,e1)
def place(P,refl,phi):
    v=P-cm; along=(v@axis)[:,None]*axis
    a=v@e1; b=(v@e2)*refl                                # refl=-1 -> mirror (enantiomer)
    ca,sa=np.cos(phi),np.sin(phi)
    return cm+along+(a*ca-b*sa)[:,None]*e1+(a*sa+b*ca)[:,None]*e2
def perr(P):
    pc=np.array([P[f].mean(0) for f in pent])
    return np.linalg.norm(sites[:,None]-pc[None],axis=2).min(1)/HEXDIA

tS=(sites-cm)@axis; span=max(abs(tS).max(),1e-9); tSn=tS/span; belly=np.abs(tSn)<0.5
base=perr(atoms)
phis=np.linspace(0,2*np.pi,721)
best=None
for refl in (1,-1):
    for phi in phis:
        e=perr(place(atoms,refl,phi)).mean()
        if best is None or e<best[0]: best=(e,refl,phi)
_,refl,phi=best; Xw=place(atoms,refl,phi); pe=perr(Xw)
print("register opt: %s roll=%.1f deg | pent_err %.2f->%.2f (belly %.2f->%.2f, tip %.2f->%.2f)"%(
    "mirror+" if refl<0 else "rot", np.degrees(phi), base.mean(), pe.mean(),
    base[belly].mean(), pe[belly].mean(), base[~belly].mean(), pe[~belly].mean()))

out=dict(D); out['atoms']=Xw.tolist(); out['atomsIdeal']=Xw.tolist()
out['center']=Xw.mean(0).tolist(); out['scale']=float(np.abs(Xw-Xw.mean(0)).max())
json.dump(out,open("/Users/olson/mytestapp/viewer_data.json","w"))
json.dump({"meshV":D['meshV'],"meshF":D['meshF'],"atoms":Xw.tolist(),
           "bonds":[[int(a),int(b)] for a,b in bonds],"peaks":sites.tolist(),
           "pent":[[int(i) for i in f] for f in pent],
           "hex":[[int(i) for i in f] for f in hexf]},
          open("/tmp/render_data.json","w"))
print("rewrote viewer_data.json + render_data.json (registered, bowl-anchored C%d)"%len(atoms))
