#!/usr/bin/env python3
"""Planarize cage faces. The finalize FF relax uses only bond + angle terms, so
faces can pucker (a pentagon can hold 108 deg angles yet buckle, like cyclopentane).
planarize() adds a per-face planarity force (pull each face atom toward the face's
best-fit plane) alongside bond+angle. Importable by finalize_rep; CLI rewrites a
render json.  Usage: python planarize_cage.py <render.json> [out.json]
"""
import sys, json, numpy as np
from collections import defaultdict

def planarize(A, bonds, faces, iters=700, KB=1.0, KA=0.7, KP=0.6, step=0.05):
    A=np.asarray(A,float); bonds=np.asarray(bonds)
    b0,b1=bonds[:,0],bonds[:,1]; L0=np.median(np.linalg.norm(A[b0]-A[b1],axis=1))
    nbr=defaultdict(list)
    for a,b in bonds: nbr[a].append(b); nbr[b].append(a)
    ang=[]
    for i in range(len(A)):
        nb=nbr[i]
        for p in range(len(nb)):
            for q in range(p+1,len(nb)): ang.append((nb[p],i,nb[q]))
    ang=np.array(ang)
    for _ in range(iters):
        F=np.zeros_like(A)
        e=A[b0]-A[b1]; L=np.linalg.norm(e,axis=1,keepdims=True); f=(L-L0)/L*e
        np.add.at(F,b0,-KB*f); np.add.at(F,b1,KB*f)
        rij=A[ang[:,0]]-A[ang[:,1]]; rik=A[ang[:,2]]-A[ang[:,1]]
        lij=np.linalg.norm(rij,axis=1,keepdims=True); lik=np.linalg.norm(rik,axis=1,keepdims=True)
        c=np.clip((rij*rik).sum(1,keepdims=True)/(lij*lik),-1,1); th=np.arccos(c); ss=np.maximum(np.sin(th),1e-6)
        dE=(th-np.radians(120.0)); dcj=rik/(lij*lik)-c*rij/lij**2; dck=rij/(lij*lik)-c*rik/lik**2
        gj=dE/ss*dcj; gk=dE/ss*dck
        np.add.at(F,ang[:,0],KA*gj); np.add.at(F,ang[:,2],KA*gk); np.add.at(F,ang[:,1],-KA*(gj+gk))
        for face in faces:
            P=A[face]; cc=P.mean(0); n=np.linalg.svd(P-cc)[2][2]; dist=(P-cc)@n
            np.add.at(F,face,-KP*dist[:,None]*n)
        A=A+step*F
    return A

def face_planarity(A,faces,L0):
    A=np.asarray(A,float)
    return np.array([np.abs((A[f]-A[f].mean(0))@np.linalg.svd(A[f]-A[f].mean(0))[2][2]).max()/L0 for f in faces])

if __name__=="__main__":
    fn=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else fn
    D=json.load(open(fn)); A=np.array(D['atoms']); bonds=np.array(D['bonds'])
    faces=[list(f) for f in D['pent']]+[list(f) for f in D['hex']]
    L0=np.median(np.linalg.norm(A[bonds[:,0]]-A[bonds[:,1]],axis=1))
    pre=face_planarity(A,faces,L0); A2=planarize(A,bonds,faces); post=face_planarity(A2,faces,L0)
    bl=np.linalg.norm(A2[bonds[:,0]]-A2[bonds[:,1]],axis=1)
    print("planarity mean %.3f->%.3f  pentMax %.3f->%.3f  hexMax %.3f->%.3f  bondCV %.2f%%"%(
        pre.mean(),post.mean(),pre[:12].max(),post[:12].max(),pre[12:].max(),post[12:].max(),bl.std()/bl.mean()*100))
    D['atoms']=A2.tolist(); json.dump(D,open(out,"w")); print("wrote",out)
