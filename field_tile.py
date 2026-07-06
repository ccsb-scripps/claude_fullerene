#!/usr/bin/env python3
"""Step 2b/2c: field-guided advancing front. Place capsomers by stepping geodesic
distance s along the 6 hexagonal field directions (frames transported by the
designed connection Levi-Civita + x), merging coincident capsomers. The 12 field
singularities become degree-5 pentamers automatically.  Output: clean capsomer
graph (target 12 pentagons + H hexagons)."""
import numpy as np, sys
from collections import defaultdict
from scipy.spatial import cKDTree

H=int(sys.argv[1]) if len(sys.argv)>1 else 230
N=12+H
m=np.load("/tmp/mesh4a.npz"); V=m['V']; F=m['F']; cm=m['c']
of=np.load("/tmp/orient_field.npz"); x=of['x']
area=0.0
for f in F: area+=0.5*np.linalg.norm(np.cross(V[f[1]]-V[f[0]],V[f[2]]-V[f[0]]))
s=np.sqrt(2*area/(N*np.sqrt(3)))               # capsomer spacing for N capsomers
print("H=%d N=%d spacing s=%.2f"%(H,N,s))

# per-face frame
e1=np.zeros((len(F),3)); e2=np.zeros((len(F),3)); nrm=np.zeros((len(F),3)); O=np.zeros((len(F),3))
for fi,f in enumerate(F):
    a,b,c=V[f[0]],V[f[1]],V[f[2]]; u=b-a; n=np.cross(b-a,c-a); n/=np.linalg.norm(n)+1e-12
    u/=np.linalg.norm(u)+1e-12; v=np.cross(n,u); e1[fi]=u;e2[fi]=v;nrm[fi]=n;O[fi]=a
# edges -> faces, edge id map (matches orient_field ordering)
edges={}
def eid(a,b):
    k=(min(a,b),max(a,b));
    if k not in edges: edges[k]=len(edges)
    return edges[k]
for f in F:
    for i in range(3): eid(f[i],f[(i+1)%3])
e2f=defaultdict(list)
for fi,f in enumerate(F):
    for i in range(3): e2f[(min(f[i],f[(i+1)%3]),max(f[i],f[(i+1)%3]))].append(fi)
def edge_angle(fi,a,b):
    d=V[b]-V[a]; return np.arctan2(d@e2[fi], d@e1[fi])
# transport f->g across edge (a,b): theta_g = theta_f + (ang_g - ang_f) + x_signed
trans={}
for (a,b),fs in e2f.items():
    if len(fs)!=2: continue
    f,g=fs; ei=edges[(a,b)]; af=edge_angle(f,a,b); ag=edge_angle(g,a,b)
    trans[(f,g)]=(ag-af)+x[ei]; trans[(g,f)]=(af-ag)-x[ei]

def to2d(fi,P): d=P-O[fi]; return np.array([d@e1[fi], d@e2[fi]])
def to3d(fi,p2): return O[fi]+p2[0]*e1[fi]+p2[1]*e2[fi]

def walk(f,P,theta,dist):
    """trace geodesic from 3D point P on face f, heading angle theta (in f frame),
    for length dist. Return (face, 3D point, theta)."""
    for _ in range(400):
        tv=[to2d(f,V[F[f][k]]) for k in range(3)]
        p=to2d(f,P); d=np.array([np.cos(theta),np.sin(theta)])
        # intersect ray p+t d with 3 edges; take smallest t>eps
        best_t=None; best_edge=None; best_q=None
        for k in range(3):
            A=tv[k]; B=tv[(k+1)%3]; ab=B-A
            den=d[0]*(-ab[1])-d[1]*(-ab[0])          # cross(d,-ab)
            if abs(den)<1e-12: continue
            rhs=A-p
            t=(rhs[0]*(-ab[1])-rhs[1]*(-ab[0]))/den   # param along ray
            u=(d[0]*rhs[1]-d[1]*rhs[0])/den           # param along edge
            if t>1e-7 and -1e-6<=u<=1+1e-6:
                if best_t is None or t<best_t: best_t=t; best_edge=k; best_q=p+t*d
        if best_t is None:                            # numerical dead-end
            return f,P,theta
        if best_t>=dist:                              # endpoint inside this face
            return f, to3d(f,p+dist*d), theta
        # cross edge best_edge to neighbor
        a=F[f][best_edge]; b=F[f][(best_edge+1)%3]; key=(min(a,b),max(a,b))
        nb=[gg for gg in e2f[key] if gg!=f]
        q3=to3d(f,best_q)
        if not nb: return f,q3,theta                  # boundary (shouldn't happen, closed)
        g=nb[0]; theta=theta+trans[(f,g)]; P=q3; dist-=best_t; f=g
    return f,P,theta

# --- advancing front ---
caps=[]           # (3D pos, face, theta0)  theta0 = orientation of hex-frame at capsomer
cappos=[]
def add_cap(P,f,th):
    caps.append((P,f,th)); cappos.append(P); return len(caps)-1
# seed near the middle, on the face closest to centroid-shifted point
seedf=int(np.argmin(np.linalg.norm(V[F].mean(1)-cm-np.array([0,0,0]),axis=1)))
add_cap(V[F[seedf]].mean(0), seedf, 0.0)
adj=defaultdict(set)
merge_r=0.55*s
frontier=[0]; guard=0
while frontier and guard<20000:
    guard+=1
    ci=frontier.pop()
    P,f,th=caps[ci]
    tree=cKDTree(np.array(cappos))
    for k in range(6):
        ang=th+k*np.pi/3
        gf,Q,th2=walk(f,P,ang,s)
        # neighbor's hex frame orientation: transported heading minus the k*60 we added,
        # i.e. align so its "0" direction is consistent (th2 - k*60 back toward incoming)
        thn=th2-k*np.pi/3
        # merge?
        dists,idx=tree.query(Q,k=1)
        if dists<merge_r:
            j=int(idx)
            if j!=ci: adj[ci].add(j); adj[j].add(ci)
        else:
            j=add_cap(Q,gf,thn); adj[ci].add(j); adj[j].add(ci)
            frontier.append(j)
cappos=np.array(cappos)
deg=np.array([len(adj[i]) for i in range(len(caps))])
hist={k:int((deg==k).sum()) for k in range(3,9) if (deg==k).sum()}
print("advancing front: capsomers=%d  valence=%s  (target N=%d, want {5:12,6:%d})"%(
    len(caps),hist,N,N-12))
np.savez("/tmp/field_tiling.npz",pos=cappos,adj=np.array([[i,j] for i in adj for j in adj[i] if i<j]),
         deg=deg)
print("saved /tmp/field_tiling.npz")
