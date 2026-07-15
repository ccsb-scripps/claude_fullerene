#!/usr/bin/env python3
"""Step 6 -- pseudo-atomic capsid from a fullerene cage.

Decorate a fitted fullerene cage (12 pentagons + H hexagons) with real HIV-1 CA
capsomer atomic structures to build a pseudo-atomic capsid, then remove all steric
overlap by rigid-body relaxation.

Pipeline (mesh -> atomic model):
    ./run_all.sh mesh.stl                       # existing pipeline -> mesh_C###_representative.xyz
    python build_atomic_capsid.py mesh_C###_representative.xyz mesh.stl

What it does:
  1. Fetch capsomer templates from RCSB (cached in ./capsomer_templates/):
        3H47  CA hexamer  (biological assembly = 6 MODELs)
        3P05  CA pentamer (5 chains in the asymmetric unit)
     Canonicalize each: centroid at origin, symmetry axis on +z, NTD facing out.
  2. Detect the cage faces (cage_rspi.faces_of); scale the cage so the median
     hexagon-hexagon face spacing = --spacing (default 95.5 A, the CA lattice).
  3. Place a capsomer on every face: axis along the local facet normal, subunits
     rotated --azimuth (default 30 deg, toward the inter-capsomer 2-fold edges =
     the CTD-dimer arrangement, which minimizes initial clashes).
  4. Rigid-body de-clash (translation + axial spin, tethered so nothing runs away):
     a coarse Calpha pass then an all-heavy-atom pass. Capsomers move only ~1-2 A /
     a few degrees; result: no inter-capsomer heavy-atom contact < 2.6 A at near-
     native ~96 A spacing. (Rigid isolated capsomers cannot tile at native density
     without this; the real capsid uses complementary flexible interfaces.)
  5. Fit the cage to the input mesh and carry the mesh surface into the model
     reference frame -> a co-registered .obj.
  6. Write outputs (same Angstrom frame):
        <prefix>_capsid_atomic.pdb    all-atom, hybrid-36 serials, segID = capsomer
        <prefix>_capsid_atomic.cif    same as mmCIF (unique chain id per subunit)
        <prefix>_capsid_backbone.pdb  N,CA,C,O only; pentamers flagged (segID P*,
                                      B-factor=100) so color-by-B shows the 12
        <mesh>_surface.obj            cryo-ET surface in the model frame

Requires numpy + scipy (use the repo .venv; the system python3 has neither).
"""
import os, sys, json, struct, time, argparse, urllib.request
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cage_rspi import load_xyz, faces_of

REPO = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_URLS = {"3H47.pdb1": "https://files.rcsb.org/download/3H47.pdb1",
                 "3P05.pdb":  "https://files.rcsb.org/download/3P05.pdb"}

# ---------------------------------------------------------------- templates
def fetch_templates(tdir):
    os.makedirs(tdir, exist_ok=True); out={}
    for name, url in TEMPLATE_URLS.items():
        p = os.path.join(tdir, name)
        if not os.path.exists(p):
            print("  fetching", url); urllib.request.urlretrieve(url, p)
        out[name] = p
    return out

def parse_pdb(fn, per_model_chain=False):
    recs=[]; model=0; C="ABCDEFGHIJ"
    for ln in open(fn):
        if ln.startswith("MODEL"): model=int(ln.split()[1])
        if not ln.startswith("ATOM"): continue
        try: rs=int(ln[22:26])
        except ValueError: continue
        ch=C[model-1] if (per_model_chain and model>0) else ln[21]
        elem=ln[76:78] if len(ln)>=78 else (" "+ln[12:16].strip()[0])
        recs.append((ch,rs,ln[12:16],ln[16],ln[17:20],ln[26],elem,
                     float(ln[30:38]),float(ln[38:46]),float(ln[46:54])))
    return recs

def canon(recs):
    """Return dict with canonicalized coords + per-atom metadata."""
    xyz=np.array([[r[7],r[8],r[9]] for r in recs]); ch=np.array([r[0] for r in recs])
    rs=np.array([r[1] for r in recs]); xyz=xyz-xyz.mean(0)
    us=sorted(set(ch)); cents=np.array([xyz[ch==u].mean(0) for u in us])
    _,_,vt=np.linalg.svd(cents-cents.mean(0)); axis=vt[2]; proj=xyz@axis
    if proj[rs<=145].mean() < proj[rs>=150].mean(): axis=-axis      # NTD (res<=145) outward
    z=np.array([0,0,1.]); v=np.cross(axis,z); s=np.linalg.norm(v); c=axis@z
    if s<1e-8: R1=np.eye(3) if c>0 else np.diag([1.,-1,-1])
    else:
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); R1=np.eye(3)+vx+vx@vx*((1-c)/s/s)
    xyz=xyz@R1.T
    caA=xyz[ch==us[0]].mean(0); ang=np.arctan2(caA[1],caA[0])       # subunit A -> +x
    cz,sz=np.cos(-ang),np.sin(-ang); xyz=xyz@np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]]).T
    mid=[r[2]+r[3]+r[4]+" "+ch[i]+("%4d"%rs[i])+r[5] for i,r in enumerate(recs)]   # PDB cols 13-27
    return dict(xyz=xyz, name=np.array([r[2].strip() for r in recs]),
                elem=np.array([r[6].strip() for r in recs]),
                resn=np.array([r[4].strip() for r in recs]),
                chain=ch, resSeq=rs, mid=mid)

# ---------------------------------------------------------------- cage
def load_cage(cagefile, spacing, faces_json=None):
    # faces come from the AUTHORITATIVE 12-pentagon list (best.json pent/hex) when
    # available; faces_of() (coordinate-based) is only reliable on a relaxed/uniform
    # cage. Pair a relaxed .xyz (uniform bonds -> clean packing) with --faces for the
    # cleanest result; a fit best.json can be locally compressed and leave clashes.
    meshV=meshF=None
    if cagefile.endswith(".json"):
        D=json.load(open(cagefile)); X=np.array(D['atoms'])
        faces=[list(map(int,f)) for f in D['pent']]+[list(map(int,f)) for f in D['hex']]
        if 'meshV' in D: meshV=np.array(D['meshV'],float); meshF=np.array(D['meshF'])  # cage & mesh share this frame
    else:
        X=load_xyz(cagefile)
        if faces_json:
            D=json.load(open(faces_json)); faces=[list(map(int,f)) for f in D['pent']]+[list(map(int,f)) for f in D['hex']]
        else:
            faces=faces_of(X)
    nhex=sum(len(f)==6 for f in faces); npent=sum(len(f)==5 for f in faces)
    Xc=X-X.mean(0); fcent=np.array([Xc[f].mean(0) for f in faces])
    v2f=defaultdict(list)
    for fi,f in enumerate(faces):
        for v in f: v2f[v].append(fi)
    pairs=set()
    for fs in v2f.values():
        for i in range(len(fs)):
            for j in range(i+1,len(fs)):
                a,b=sorted((fs[i],fs[j]))
                if len(set(faces[a])&set(faces[b]))>=2: pairs.add((a,b))
    hexset={i for i,f in enumerate(faces) if len(f)==6}
    hh=[np.linalg.norm(fcent[a]-fcent[b]) for a,b in pairs if a in hexset and b in hexset]
    sc=spacing/np.median(hh); Xc*=sc; fcent*=sc
    return X, faces, Xc, fcent, sorted(pairs), nhex, npent, sc, meshV, meshF

def build_frames(Xc, fcent, faces, azim_deg):
    gc=Xc.mean(0); A=np.radians(azim_deg); R0=[]; N=[]
    for fi,f in enumerate(faces):
        c=fcent[fi]; P=Xc[f]; _,_,vt=np.linalg.svd(P-P.mean(0)); nrm=vt[2]
        if (c-gc)@nrm<0: nrm=-nrm
        v0=Xc[f[0]]-c; x=v0-(v0@nrm)*nrm; x/=np.linalg.norm(x)+1e-9
        xr=np.cos(A)*x+np.sin(A)*np.cross(nrm,x); y=np.cross(nrm,xr)
        R0.append(np.column_stack([xr,y,nrm])); N.append(nrm)
    return np.array(R0), np.array(N)

# ---------------------------------------------------------------- relaxation
def _Rz(a): c,s=np.cos(a),np.sin(a); M=np.zeros((len(a),3,3)); M[:,0,0]=c;M[:,0,1]=-s;M[:,1,0]=s;M[:,1,1]=c;M[:,2,2]=1; return M

def relax_ca(faces, R0, N, Ctar, hx, pt, iters, THR=4.5, lr_t=0.05, k_t=0.06, cap=30., lr_rot=6e-5):
    hx_ca=hx['xyz'][hx['name']=='CA']; pt_ca=pt['xyz'][pt['name']=='CA']
    NF=len(faces); hxi=np.array([i for i,f in enumerate(faces) if len(f)==6])
    pti=np.array([i for i,f in enumerate(faces) if len(f)==5])
    base=[]; nrm=[]; ids=[]
    for fi,f in enumerate(faces):
        t=hx_ca if len(f)==6 else pt_ca
        base.append(t@R0[fi].T+Ctar[fi]); nrm.append(np.repeat(N[fi][None],len(t),0)); ids.append(np.full(len(t),fi))
    base=np.vstack(base); nrm=np.vstack(nrm); ids=np.concatenate(ids)
    cen=Ctar.copy(); phi=np.zeros(NF)
    for it in range(iters):
        Reff=np.einsum('fij,fjk->fik',R0,_Rz(phi))
        ph=np.einsum('fij,pj->fpi',Reff[hxi],hx_ca)+cen[hxi][:,None,:]
        pp=np.einsum('fij,pj->fpi',Reff[pti],pt_ca)+cen[pti][:,None,:]
        pts=np.vstack([ph.reshape(-1,3),pp.reshape(-1,3)])
        fid=np.concatenate([np.repeat(hxi,hx_ca.shape[0]),np.repeat(pti,pt_ca.shape[0])])
        pr=np.array(list(cKDTree(pts).query_pairs(THR)),dtype=int)
        F=np.zeros((NF,3)); tau=np.zeros(NF)
        if len(pr):
            i,j=pr[:,0],pr[:,1]; fi,fj=fid[i],fid[j]; m=fi!=fj; i,j,fi,fj=i[m],j[m],fi[m],fj[m]
            if len(i):
                dv=pts[i]-pts[j]; dist=np.linalg.norm(dv,axis=1)+1e-9; f=(np.clip(THR-dist,0,None)/dist)[:,None]*dv
                np.add.at(F,fi,f); np.add.at(F,fj,-f)
                np.add.at(tau,fi,np.einsum('pi,pi->p',np.cross(pts[i]-cen[fi],f),N[fi]))
                np.add.at(tau,fj,np.einsum('pi,pi->p',np.cross(pts[j]-cen[fj],-f),N[fj]))
        F=F-k_t*(cen-Ctar); cen=cen+lr_t*F
        off=cen-Ctar; L=np.linalg.norm(off,axis=1); big=L>cap; cen[big]=Ctar[big]+off[big]/L[big,None]*cap
        phi=phi+np.clip(lr_rot*tau,-0.03,0.03)
    return cen, phi

def relax_heavy(faces, pairs, R0, N, cen, phi, hx, pt, iters, THR=3.2, lr_t=0.05, k_t=0.05,
                cap=45., lr_rot=3e-5, reanchor=100):
    NF=len(faces); Ctar=cen.copy()
    hxi=np.array([i for i,f in enumerate(faces) if len(f)==6]); pti=np.array([i for i,f in enumerate(faces) if len(f)==5])
    hx_xyz=hx['xyz']; pt_xyz=pt['xyz']
    def build():
        Reff=np.einsum('fij,fjk->fik',R0,_Rz(phi)); arr=[None]*NF
        ph=np.einsum('fij,pj->fpi',Reff[hxi],hx_xyz)+cen[hxi][:,None,:]
        pp=np.einsum('fij,pj->fpi',Reff[pti],pt_xyz)+cen[pti][:,None,:]
        for k,fi in enumerate(hxi): arr[fi]=ph[k]
        for k,fi in enumerate(pti): arr[fi]=pp[k]
        return arr
    for it in range(iters):
        if it>0 and it%reanchor==0: Ctar=cen.copy()   # re-anchor tether to spread positions (lets it pass the tether floor)
        arr=build(); trees=[cKDTree(a) for a in arr]; F=np.zeros((NF,3)); tau=np.zeros(NF); nhard=0
        for a,b in pairs:
            M=trees[a].sparse_distance_matrix(trees[b],THR,output_type='coo_matrix')
            if M.nnz==0: continue
            ia,ib,dist=M.row,M.col,np.maximum(M.data,1e-6)
            pa=arr[a][ia]; pb=arr[b][ib]; f=((THR-dist)/dist)[:,None]*(pa-pb); Fa=f.sum(0)
            F[a]+=Fa; F[b]-=Fa
            tau[a]+=(np.cross(pa-cen[a],f)@N[a]).sum(); tau[b]+=(np.cross(pb-cen[b],-f)@N[b]).sum()
            nhard+=int((dist<2.6).sum())
        F=F-k_t*(cen-Ctar); cen=cen+lr_t*F
        off=cen-Ctar; L=np.linalg.norm(off,axis=1); big=L>cap; cen[big]=Ctar[big]+off[big]/L[big,None]*cap
        phi=phi+np.clip(lr_rot*tau,-0.02,0.02)
        if it%20==0 or it==iters-1: print("    heavy it%3d  inter-atom<2.6A=%d"%(it,nhard))
    return cen, phi

# ---------------------------------------------------------------- mesh -> model frame
def load_mesh(fn):
    buf=open(fn,"rb").read()
    if buf[:3]==b"ply": return load_ply(buf)
    return load_stl(buf)
def load_stl(buf):
    ntri=struct.unpack("<I",buf[80:84])[0]
    tris=np.frombuffer(buf,dtype=np.dtype([("n","<3f4"),("v","<3,3f4"),("a","<u2")]),count=ntri,offset=84)
    P=tris["v"].reshape(-1,3).astype(np.float64); diag=np.linalg.norm(P.max(0)-P.min(0)); tol=1e-5*diag
    key=np.round(P/tol).astype(np.int64); _,first,inv=np.unique(key,axis=0,return_index=True,return_inverse=True)
    inv=np.asarray(inv).ravel(); order=np.argsort(first); remap=np.empty(len(first),np.int64); remap[order]=np.arange(len(first))
    return P[first][order], remap[inv].reshape(ntri,3)
def load_ply(buf):
    hpos=buf.index(b"end_header"); he=buf.index(b"\n",hpos)+1; hdr=buf[:he].decode("ascii","replace").splitlines()
    fmt=next(l for l in hdr if l.startswith("format")); nv=int(next(l for l in hdr if l.startswith("element vertex")).split()[-1])
    nf=int(next(l for l in hdr if l.startswith("element face")).split()[-1]); pv=0; inv=False
    for l in hdr:
        if l.startswith("element vertex"): inv=True; continue
        if l.startswith("element "): inv=False
        if inv and l.startswith("property"): pv+=1
    if "ascii" in fmt:
        it=iter(buf[he:].decode("ascii","replace").split())
        V=np.array([[float(next(it)) for _ in range(pv)][:3] for _ in range(nv)],float); F=[]
        for _ in range(nf):
            k=int(next(it)); idx=[int(next(it)) for _ in range(k)]
            if k==3: F.append(idx)
        return V,np.array(F)
    off=he; V=np.frombuffer(buf,dtype="<f4",count=nv*pv,offset=off).reshape(nv,pv)[:,:3].astype(np.float64); off+=nv*pv*4; F=[]
    for _ in range(nf):
        k=buf[off]; off+=1; idx=np.frombuffer(buf,dtype="<u4",count=k,offset=off); off+=k*4
        if k==3: F.append(list(idx))
    return V,np.array(F)

def mesh_to_model_frame(X, P_A, meshfile):
    M,Fc=load_mesh(meshfile)
    cX=X.mean(0); _,_,vt=np.linalg.svd(X-cX,full_matrices=False); Xc=(X-cX)@vt.T
    cM=M.mean(0); _,_,vtM=np.linalg.svd(M-cM,full_matrices=False); Mc=(M-cM)@vtM.T; mt=cKDTree(M); best=None
    for swap in ([0,1,2],[0,2,1]):
        for s0 in (1,-1):
            for s1 in (1,-1):
                for s2 in (1,-1):
                    Xt=Xc[:,swap]*np.array([s0,s1,s2]); uni=(Mc.max(0)-Mc.min(0)).mean()/(Xt.max(0)-Xt.min(0)).mean()
                    Xw=(Xt*uni)@vtM+cM; sc=mt.query(Xw)[0].mean()
                    if best is None or sc<best[0]: best=(sc,Xw)
    Rm=np.linalg.norm(M-cM,axis=1).mean(); Xw=best[1]
    src,dst=Xw,P_A; mp,mq=src.mean(0),dst.mean(0); Xs=src-mp; Ys=dst-mq
    Cov=Ys.T@Xs/len(src); U,S,Vt=np.linalg.svd(Cov); d=np.sign(np.linalg.det(U@Vt))
    R=U@np.diag([1,1,d])@Vt; s=(S*np.array([1,1,d])).sum()/((Xs**2).sum()/len(src)); t=mq-s*R@mp
    return s*(M@R.T)+t, Fc, best[0], best[0]/Rm*100

# ---------------------------------------------------------------- writers
_DU="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; _DL=_DU.lower()
def _pure(d,v):
    r=[]
    while v: v,m=divmod(v,len(d)); r.append(d[m])
    return "".join(reversed(r)) if r else d[0]
def hy36(v,w=5):
    if v<10**w: return ("%%%dd"%w)%v
    v-=10**w
    if v<26*36**(w-1): return _pure(_DU,v+10*36**(w-1))
    v-=26*36**(w-1); return _pure(_DL,v+10*36**(w-1))
def _Rz1(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def write_pdb(path, faces, cen, phi, R0, hx, pt, backbone=False):
    BB={'N','CA','C','O'}; serial=0; nh=npv=0
    hx_keep=[i for i,n in enumerate(hx['name']) if (not backbone or n in BB)]
    pt_keep=[i for i,n in enumerate(pt['name']) if (not backbone or n in BB)]
    hx_xyz=hx['xyz'][hx_keep]; hx_mid=[hx['mid'][i] for i in hx_keep]; hx_el=hx['elem'][hx_keep]
    pt_xyz=pt['xyz'][pt_keep]; pt_mid=[pt['mid'][i] for i in pt_keep]; pt_el=pt['elem'][pt_keep]
    with open(path,"w") as fo:
        fo.write("HEADER    HIV-1 CA PSEUDO-ATOMIC CAPSID%s\n"%("  (BACKBONE)" if backbone else ""))
        fo.write("REMARK   1 235-style hexamers (3H47) + 12 pentamers (3P05) on a fullerene cage.\n")
        fo.write("REMARK   1 chainID=subunit; segID(73-76)=capsomer Hnnn/Pnnn.%s\n"%
                 ("  Pentamers B-factor=100." if backbone else ""))
        for fi,f in enumerate(faces):
            hexf=len(f)==6
            xyz,mid,el=(hx_xyz,hx_mid,hx_el) if hexf else (pt_xyz,pt_mid,pt_el)
            if hexf: nh+=1; seg="H%03d"%nh; b=0.0
            else: npv+=1; seg="P%03d"%npv; b=100.0
            W=xyz@(R0[fi]@_Rz1(phi[fi])).T+cen[fi]; buf=[]
            for k in range(len(xyz)):
                serial+=1; x,y,z=W[k]
                buf.append("ATOM  %5s %s   %8.3f%8.3f%8.3f  1.00%6.2f      %-4s%2s\n"%(
                    hy36(serial),mid[k],x,y,z,b,seg,el[k]))
            buf.append("TER\n"); fo.write("".join(buf))
        fo.write("END\n")
    return serial

def write_cif(path, faces, cen, phi, R0, hx, pt):
    cols=["group_PDB","id","type_symbol","label_atom_id","label_comp_id","label_asym_id","label_seq_id",
          "Cartn_x","Cartn_y","Cartn_z","occupancy","B_iso_or_equiv","auth_asym_id","auth_seq_id","pdbx_PDB_model_num"]
    serial=0; nh=npv=0
    with open(path,"w") as fo:
        fo.write("data_capsid\n#\nloop_\n"+"".join("_atom_site.%s\n"%c for c in cols))
        for fi,f in enumerate(faces):
            hexf=len(f)==6; T=hx if hexf else pt
            if hexf: nh+=1; seg="H%03d"%nh
            else: npv+=1; seg="P%03d"%npv
            W=T['xyz']@(R0[fi]@_Rz1(phi[fi])).T+cen[fi]; nm,rn,el,ch,rs=T['name'],T['resn'],T['elem'],T['chain'],T['resSeq']; buf=[]
            for k in range(len(W)):
                serial+=1; uid=seg+ch[k]; x,y,z=W[k]
                buf.append("ATOM %d %s %s %s %s %d %.3f %.3f %.3f 1.00 0.00 %s %d 1\n"%(
                    serial,el[k],nm[k],rn[k],uid,rs[k],x,y,z,uid,rs[k]))
            fo.write("".join(buf))
        fo.write("#\n")
    return serial

def write_obj(path, V, F):
    with open(path,"w") as f:
        f.write("# cryo-ET surface aligned to the atomic capsid model frame (Angstrom)\no surface\n")
        for v in V: f.write("v %.4f %.4f %.4f\n"%(v[0],v[1],v[2]))
        for face in F: f.write("f "+" ".join(str(int(i)+1) for i in face)+"\n")

def write_template_pdb(path, T):
    """Canonical capsomer (small, <1MB) so instances can be rebuilt from transforms."""
    with open(path,"w") as fo:
        for k in range(len(T['xyz'])):
            x,y,z=T['xyz'][k]
            fo.write("ATOM  %5d %s   %8.3f%8.3f%8.3f  1.00  0.00      %2s\n"%((k+1)%100000,T['mid'][k],x,y,z,T['elem'][k]))
        fo.write("END\n")

def write_transforms(path, faces, cen, phi, R0, meta):
    """Compact instance file: one rigid transform per capsomer (world = R @ canonical + t).
    Expands to the full model via expand_capsid.py -- ~KB instead of ~180MB."""
    caps=[]; nh=npv=0
    for fi,f in enumerate(faces):
        hexf=len(f)==6
        if hexf: nh+=1; cid="H%03d"%nh
        else: npv+=1; cid="P%03d"%npv
        R=R0[fi]@_Rz1(phi[fi])
        caps.append({"id":cid,"type":"hex" if hexf else "pent",
                     "R":[[round(float(v),9) for v in row] for row in R],
                     "t":[round(float(v),5) for v in cen[fi]]})
    json.dump({**meta,"n_capsomers":len(caps),"capsomers":caps}, open(path,"w"), indent=0)

# ---------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser(description="Build a pseudo-atomic HIV CA capsid from a fullerene cage.")
    ap.add_argument("cage_xyz", help="representative fullerene cage (.xyz) from run_all.sh")
    ap.add_argument("mesh", help="the source mesh (.stl or .ply) to co-register the surface")
    ap.add_argument("--out-prefix", default=None, help="output name stem (default: from cage filename)")
    ap.add_argument("--outdir", default=".", help="output directory (default: cwd)")
    ap.add_argument("--spacing", type=float, default=95.5, help="target hex-hex face spacing, A (default 95.5)")
    ap.add_argument("--azimuth", type=float, default=30.0, help="subunit azimuth toward 2-fold edges, deg (default 30)")
    ap.add_argument("--ca-iters", type=int, default=200); ap.add_argument("--heavy-iters", type=int, default=400)
    ap.add_argument("--templates", default=os.path.join(REPO,"capsomer_templates"))
    ap.add_argument("--emit", choices=["full","transforms","both"], default="both",
                    help="full=big PDB/CIF/backbone; transforms=compact instance JSON + canonical templates (rebuild with expand_capsid.py); both (default)")
    ap.add_argument("--faces", default=None, help="best.json with authoritative pent/hex faces, to pair with a relaxed .xyz geometry")
    a=ap.parse_args()
    prefix=a.out_prefix or os.path.basename(a.cage_xyz).replace("_representative","").replace(".xyz","")
    os.makedirs(a.outdir, exist_ok=True); t0=time.time()

    print("[1/6] templates"); tp=fetch_templates(a.templates)
    hx=canon(parse_pdb(tp["3H47.pdb1"],per_model_chain=True)); pt=canon(parse_pdb(tp["3P05.pdb"]))
    print("      hexamer %d atoms / pentamer %d atoms"%(len(hx['xyz']),len(pt['xyz'])))
    print("[2/6] cage + faces"); X,faces,Xc,fcent,pairs,nhex,npent,scale,meshV,meshF=load_cage(a.cage_xyz,a.spacing,a.faces)
    print("      C%d: %d hexamers + %d pentamers, %d adjacent pairs"%(len(X),nhex,npent,len(pairs)))
    R0,N=build_frames(Xc,fcent,faces,a.azimuth)
    print("[3/6] placement (azimuth %.0f, spacing %.1f A)"%(a.azimuth,a.spacing))
    print("[4/6] rigid-body de-clash: Calpha (%d) then heavy-atom (%d)"%(a.ca_iters,a.heavy_iters))
    cen,phi=relax_ca(faces,R0,N,fcent,hx,pt,a.ca_iters)
    cen,phi=relax_heavy(faces,pairs,R0,N,cen,phi,hx,pt,a.heavy_iters)
    print("[5/6] surface -> model frame")
    obj=os.path.join(a.outdir,os.path.basename(a.mesh).rsplit(".",1)[0]+"_surface.obj")
    if meshV is not None:                       # cage-json: mesh & cage share a frame -> exact co-registration (robust for near-spherical shapes)
        V,F=(meshV-X.mean(0))*scale, meshF; write_obj(obj,V,F)
        print("      surface co-registered from best.json meshV (scale %.4f); wrote %s"%(scale,obj))
    else:                                       # xyz cage: fit the differently-framed mesh onto it (PCA + orientation search)
        V,F,fitA,fitpct=mesh_to_model_frame(X,Xc,a.mesh); write_obj(obj,V,F)
        print("      cage->mesh fit %.1f A (%.1f%% of Rm); wrote %s"%(fitA,fitpct,obj))
    print("[6/6] writing outputs (emit=%s)"%a.emit)
    if a.emit in ("transforms","both"):
        os.makedirs(a.templates,exist_ok=True)
        hxt=os.path.join(a.templates,"hexamer_canonical.pdb"); ptt=os.path.join(a.templates,"pentamer_canonical.pdb")
        write_template_pdb(hxt,hx); write_template_pdb(ptt,pt)
        p_tr=os.path.join(a.outdir,prefix+"_capsid_transforms.json")
        write_transforms(p_tr,faces,cen,phi,R0,{
            "description":"HIV-1 CA pseudo-atomic capsid -- per-capsomer instance transforms",
            "frame":"Angstrom; world = R @ canonical_template + t",
            "hexamer_template":"RCSB 3H47 (canonical: centroid origin, axis +z, NTD out)",
            "pentamer_template":"RCSB 3P05 (canonical)","spacing_A":a.spacing,"azimuth_deg":a.azimuth,
            "cage":os.path.basename(a.cage_xyz),"mesh_surface_obj":os.path.basename(obj)})
        print("      %s  (%d capsomer transforms; rebuild: expand_capsid.py)"%(p_tr,len(faces)))
    if a.emit in ("full","both"):
        p_pdb=os.path.join(a.outdir,prefix+"_capsid_atomic.pdb"); n=write_pdb(p_pdb,faces,cen,phi,R0,hx,pt)
        p_cif=os.path.join(a.outdir,prefix+"_capsid_atomic.cif"); write_cif(p_cif,faces,cen,phi,R0,hx,pt)
        p_bb =os.path.join(a.outdir,prefix+"_capsid_backbone.pdb"); nb=write_pdb(p_bb,faces,cen,phi,R0,hx,pt,backbone=True)
        print("      %s  (%d atoms)"%(p_pdb,n)); print("      %s"%p_cif); print("      %s  (%d backbone atoms)"%(p_bb,nb))
    print("done in %.0f s"%(time.time()-t0))

if __name__=="__main__":
    main()
