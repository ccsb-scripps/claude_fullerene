import numpy as np
from collections import defaultdict

d=np.load('mesh_clean0.npz')
V,F,VN,c,axis=d['V'],d['F'],d['VN'],d['centroid'],d['axis']
nv=len(V)

adj=defaultdict(set)
for f in F:
    for i in range(3):
        adj[f[i]].add(f[(i+1)%3]); adj[f[i]].add(f[(i+2)%3])

kmin=np.full(nv,np.nan); kmax=np.full(nv,np.nan)
for i in range(nv):
    ring=set(adj[i])
    for j in list(ring): ring|=adj[j]
    ring.discard(i)
    nb=list(ring)
    if len(nb)<6: continue
    n=VN[i]
    t=np.array([1.0,0,0]);
    if abs(n[0])>0.9: t=np.array([0,1.0,0])
    e1=t-n*np.dot(t,n); e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
    dd=V[nb]-V[i]; x=dd@e1; y=dd@e2; z=dd@n
    A=np.column_stack([x*x,x*y,y*y,x,y,np.ones_like(x)])
    coef,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    a,b,cc=coef[0],coef[1],coef[2]
    W=np.array([[2*a,b],[b,2*cc]])
    ev=np.linalg.eigvalsh(W)
    # outward normal: convex/dome => neighbors below tangent (z<0) => a,c<0 => negate so convex=+
    kmin[i]=-ev.max()  # smaller principal curvature after negation = -max
    kmax[i]=-ev.min()
valid=~np.isnan(kmin)
print("kmin (convex+) range: %.5f .. %.5f ; bowls(kmin>0): %d/%d"%(np.nanmin(kmin),np.nanmax(kmin),np.sum(kmin[valid]>0),valid.sum()))

# local maxima of kmin: vertex whose kmin >= all 1-ring neighbors
islocal=np.zeros(nv,bool)
for i in range(nv):
    if not valid[i] or kmin[i]<=0: continue
    if all(kmin[i]>=kmin[j] for j in adj[i] if valid[j]):
        islocal[i]=True
cand=np.where(islocal)[0]
print("local-maxima bowl candidates:",len(cand))

# pick 12 strongest, well separated
A_area=1892128; N=216
spacing=np.sqrt(2*A_area/(N*np.sqrt(3))); sep=0.85*spacing
cand=cand[np.argsort(-kmin[cand])]
seeds=[]
for idx in cand:
    if all(np.linalg.norm(V[idx]-V[s])>sep for s in seeds):
        seeds.append(int(idx))
    if len(seeds)==12: break
seeds=np.array(seeds,dtype=int)
print("spacing~%.1f sep%.1f -> %d seeds"%(spacing,sep,len(seeds)))
print("seed kmin:",kmin[seeds].round(5))
t=(V[seeds]-c)@axis
print("axial t (sorted):",np.sort(t).round(0))
print("t range %.0f..%.0f ; #mid-body(|t|<0.4*halfspan):"%(t.min(),t.max()),
      np.sum(np.abs(t)<0.4*max(abs(t.min()),abs(t.max()))))
np.save('bowl_seeds_idx.npy',seeds)
np.save('bowl_kmin.npy',kmin)
print("saved.")
