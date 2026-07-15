#!/usr/bin/env python3
"""Rebuild the atomic capsid from a compact <prefix>_capsid_transforms.json
(written by build_atomic_capsid.py --emit transforms). The transforms store one
rigid operator per capsomer (world = R @ canonical_template + t); the canonical
CA hexamer/pentamer are regenerated deterministically from the RCSB templates
(3H47/3P05, auto-fetched), so only the tiny JSON needs to be kept/shared.

Usage:
    python expand_capsid.py <prefix>_capsid_transforms.json [--outdir DIR]
        [--cif] [--backbone]     # full expanded coordinates (~2M atoms)
        [--assembly]             # instead: compact mmCIF with pdbx_struct_assembly
                                 # (2 reference capsomers + one oper per capsomer)

Requires the repo .venv (numpy/scipy).
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_atomic_capsid as B

def write_assembly_cif(path, doc, hx, pt):
    """Compact mmCIF: deposit ONE hexamer + ONE pentamer (canonical reference
    coordinates) and generate the whole capsid via pdbx_struct_assembly_gen +
    pdbx_struct_oper_list -- one operator (R,t) per capsomer, applied to the
    hexamer or pentamer reference chains. Viewers that support assemblies
    (Mol*/PDBe, ChimeraX 'sym', PyMOL) expand it to the full ~2M-atom capsid."""
    caps=doc["capsomers"]
    def asym_map(T, prefix):
        order=[]
        for c in T['chain']:
            if c not in order: order.append(c)
        return {c: prefix+c for c in order}         # e.g. hexamer chain 'A' -> 'HA'
    hmap=asym_map(hx,"H"); pmap=asym_map(pt,"P")
    hex_asyms=",".join(hmap.values()); pent_asyms=",".join(pmap.values())
    L=["data_capsid_assembly","#",
       "_struct.title 'HIV-1 CA pseudo-atomic capsid (generated via pdbx_struct_assembly)'","#"]
    # entities
    L+=["loop_","_entity.id","_entity.type","_entity.pdbx_description",
        "1 polymer 'HIV-1 capsid protein CA, hexamer subunit (from PDB 3H47)'",
        "2 polymer 'HIV-1 capsid protein CA, pentamer subunit (from PDB 3P05)'","#"]
    # asym -> entity
    L+=["loop_","_struct_asym.id","_struct_asym.entity_id"]
    L+=["%s 1"%a for a in hmap.values()]+["%s 2"%a for a in pmap.values()]+["#"]
    # assembly
    L+=["loop_","_pdbx_struct_assembly.id","_pdbx_struct_assembly.details",
        "_pdbx_struct_assembly.oligomeric_count",
        "1 'complete fullerene capsid' %d"%len(caps),"#"]
    # generation: one row per capsomer -> (its operator) applied to its reference chains
    L+=["loop_","_pdbx_struct_assembly_gen.assembly_id","_pdbx_struct_assembly_gen.oper_expression",
        "_pdbx_struct_assembly_gen.asym_id_list"]
    for i,c in enumerate(caps,1):
        L.append("1 %d %s"%(i, hex_asyms if c["type"]=="hex" else pent_asyms))
    L.append("#")
    # operator list: R (matrix) + t (vector), X' = R.X + t (maps reference -> instance)
    hdr=["_pdbx_struct_oper_list.id","_pdbx_struct_oper_list.type","_pdbx_struct_oper_list.name"]
    for r in (1,2,3):
        hdr+=["_pdbx_struct_oper_list.matrix[%d][%d]"%(r,cc) for cc in (1,2,3)]
        hdr.append("_pdbx_struct_oper_list.vector[%d]"%r)
    L+=["loop_"]+hdr
    for i,c in enumerate(caps,1):
        R=c["R"]; t=c["t"]; row=["%d"%i,"'build operation'","%s"%c["id"]]
        for r in range(3): row+=["%.8f"%R[r][0],"%.8f"%R[r][1],"%.8f"%R[r][2],"%.5f"%t[r]]
        L.append(" ".join(row))
    L.append("#")
    # reference coordinates: canonical hexamer + canonical pentamer, ONCE
    cols=["group_PDB","id","type_symbol","label_atom_id","label_comp_id","label_asym_id","label_entity_id",
          "label_seq_id","Cartn_x","Cartn_y","Cartn_z","occupancy","B_iso_or_equiv","auth_asym_id","auth_seq_id","pdbx_PDB_model_num"]
    L+=["loop_"]+["_atom_site.%s"%c for c in cols]
    serial=0
    for T,mp,ent in ((hx,hmap,1),(pt,pmap,2)):
        xyz=T['xyz']; nm=T['name']; rn=T['resn']; el=T['elem']; ch=T['chain']; rs=T['resSeq']
        for k in range(len(xyz)):
            serial+=1; a=mp[ch[k]]; x,y,z=xyz[k]
            L.append("ATOM %d %s %s %s %s %d %d %.3f %.3f %.3f 1.00 0.00 %s %d 1"%(
                serial,el[k],nm[k],rn[k],a,ent,rs[k],x,y,z,a,rs[k]))
    L.append("#")
    open(path,"w").write("\n".join(L)+"\n")
    return serial

def main():
    ap=argparse.ArgumentParser(description="Expand a capsid transforms file to coordinates or an mmCIF assembly.")
    ap.add_argument("transforms"); ap.add_argument("--outdir", default=".")
    ap.add_argument("--cif", action="store_true", help="also write full-atom mmCIF")
    ap.add_argument("--backbone", action="store_true", help="also write an N,CA,C,O backbone PDB (pentamers flagged)")
    ap.add_argument("--assembly", action="store_true",
                    help="write a compact mmCIF assembly (reference capsomers + pdbx_struct_assembly/oper_list) INSTEAD of expanding all atoms")
    ap.add_argument("--templates", default=os.path.join(B.REPO,"capsomer_templates"))
    a=ap.parse_args()
    doc=json.load(open(a.transforms)); caps=doc["capsomers"]
    tp=B.fetch_templates(a.templates)
    hx=B.canon(B.parse_pdb(tp["3H47.pdb1"],per_model_chain=True)); pt=B.canon(B.parse_pdb(tp["3P05.pdb"]))
    os.makedirs(a.outdir,exist_ok=True)
    prefix=os.path.basename(a.transforms).replace("_capsid_transforms.json","")
    if a.assembly:
        p=os.path.join(a.outdir,prefix+"_capsid_assembly.cif"); nref=write_assembly_cif(p,doc,hx,pt)
        nhex=sum(c["type"]=="hex" for c in caps)
        print("wrote %s"%p)
        print("  %d reference atoms (1 hexamer + 1 pentamer) + %d operators (%d hex + %d pent) generating %d capsomers"%(
            nref,len(caps),nhex,len(caps)-nhex,len(caps)))
        print("  expand in a viewer: ChimeraX 'sym #1 assembly 1', Mol*/PDBe assembly 1, PyMOL 'set assembly, 1'")
        return
    faces=[[0]*(6 if c["type"]=="hex" else 5) for c in caps]     # lengths only (pick hex/pent + count)
    R0=np.array([c["R"] for c in caps]); cen=np.array([c["t"] for c in caps]); phi=np.zeros(len(caps))
    p=os.path.join(a.outdir,prefix+"_capsid_atomic.pdb"); n=B.write_pdb(p,faces,cen,phi,R0,hx,pt)
    print("wrote %s  (%d atoms, %d capsomers)"%(p,n,len(caps)))
    if a.cif:
        pc=os.path.join(a.outdir,prefix+"_capsid_atomic.cif"); B.write_cif(pc,faces,cen,phi,R0,hx,pt); print("wrote %s"%pc)
    if a.backbone:
        pb=os.path.join(a.outdir,prefix+"_capsid_backbone.pdb"); nb=B.write_pdb(pb,faces,cen,phi,R0,hx,pt,backbone=True)
        print("wrote %s  (%d backbone atoms)"%(pb,nb))

if __name__=="__main__":
    main()
