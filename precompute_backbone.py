#!/usr/bin/env python3
"""Precompute per-chain Calpha backbones for each capsid (from its transforms +
canonical templates) -> capsid_models/<prefix>_backbone.npz {coords,lengths,types}
in the committed mesh frame. Used by blender_load_set.py (Blender = numpy only)."""
import numpy as np, json, glob, os, sys
sys.path.insert(0,"/Users/olson/Dev/fullerenes"); import build_atomic_capsid as B
tp=B.fetch_templates("capsomer_templates")
hx=B.canon(B.parse_pdb(tp["3H47.pdb1"],per_model_chain=True)); pt=B.canon(B.parse_pdb(tp["3P05.pdb"]))
def segs(T):
    ca=T['name']=='CA'; xyz=T['xyz'][ca]; ch=T['chain'][ca]; out=[]; s=0
    for i in range(1,len(ch)+1):
        if i==len(ch) or ch[i]!=ch[s]: out.append(xyz[s:i]); s=i
    return out
hs, ps = segs(hx), segs(pt)
for tj in sorted(glob.glob("capsid_models/*_capsid_transforms.json")):
    pre=os.path.basename(tj).replace("_capsid_transforms.json",""); doc=json.load(open(tj))
    coords=[]; lengths=[]; types=[]
    for c in doc['capsomers']:
        R=np.array(c['R']); t=np.array(c['t']); ss=hs if c['type']=='hex' else ps; ty=0 if c['type']=='hex' else 1
        for seg in ss: coords.append(seg@R.T+t); lengths.append(len(seg)); types.append(ty)
    C=np.vstack(coords).astype('f4')
    np.savez("capsid_models/%s_backbone.npz"%pre, coords=C, lengths=np.array(lengths), types=np.array(types))
    print("  %s: %d chains, %d Ca"%(pre,len(lengths),len(C)))
