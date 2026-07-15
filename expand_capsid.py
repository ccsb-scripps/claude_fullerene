#!/usr/bin/env python3
"""Rebuild the full atomic capsid from a compact <prefix>_capsid_transforms.json
(written by build_atomic_capsid.py --emit transforms). The transforms store one
rigid operator per capsomer (world = R @ canonical_template + t); the canonical
CA hexamer/pentamer are regenerated deterministically from the RCSB templates
(3H47/3P05, auto-fetched), so only the tiny JSON needs to be kept/shared.

Usage:
    python expand_capsid.py <prefix>_capsid_transforms.json [--cif] [--backbone] [--outdir DIR]

Requires the repo .venv (numpy/scipy).
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_atomic_capsid as B

def main():
    ap=argparse.ArgumentParser(description="Expand a capsid transforms file to full coordinates.")
    ap.add_argument("transforms"); ap.add_argument("--outdir", default=".")
    ap.add_argument("--cif", action="store_true", help="also write mmCIF")
    ap.add_argument("--backbone", action="store_true", help="also write an N,CA,C,O backbone PDB (pentamers flagged)")
    ap.add_argument("--templates", default=os.path.join(B.REPO,"capsomer_templates"))
    a=ap.parse_args()
    doc=json.load(open(a.transforms))
    tp=B.fetch_templates(a.templates)
    hx=B.canon(B.parse_pdb(tp["3H47.pdb1"],per_model_chain=True)); pt=B.canon(B.parse_pdb(tp["3P05.pdb"]))
    caps=doc["capsomers"]
    faces=[[0]*(6 if c["type"]=="hex" else 5) for c in caps]     # lengths only (pick hex/pent + count)
    R0=np.array([c["R"] for c in caps]); cen=np.array([c["t"] for c in caps]); phi=np.zeros(len(caps))
    os.makedirs(a.outdir,exist_ok=True)
    prefix=os.path.basename(a.transforms).replace("_capsid_transforms.json","")
    p=os.path.join(a.outdir,prefix+"_capsid_atomic.pdb"); n=B.write_pdb(p,faces,cen,phi,R0,hx,pt)
    print("wrote %s  (%d atoms, %d capsomers)"%(p,n,len(caps)))
    if a.cif:
        pc=os.path.join(a.outdir,prefix+"_capsid_atomic.cif"); B.write_cif(pc,faces,cen,phi,R0,hx,pt); print("wrote %s"%pc)
    if a.backbone:
        pb=os.path.join(a.outdir,prefix+"_capsid_backbone.pdb"); nb=B.write_pdb(pb,faces,cen,phi,R0,hx,pt,backbone=True)
        print("wrote %s  (%d backbone atoms)"%(pb,nb))

if __name__=="__main__":
    main()
