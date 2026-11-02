from pathlib import Path
import json,csv,time
import numpy as np
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator,eigs
from scipy.optimize import brentq
import baseline_core as s

ROOT=(Path(__file__).resolve().parents[2] / 'results' / 'evolution')
TOPOLOGIES=['dense','sparse','skewed']

def write_csv(name,rows):
    if not rows:return
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def model(rng,n,topology):
    p=n//2
    if topology=='skewed':
        F=rng.lognormal(0,1.5,(p,p))*rng.lognormal(0,.8,(p,1))
        G=rng.lognormal(0,1.5,(p,p))*rng.lognormal(0,.8,(p,1))
    else:
        F=rng.uniform(.1,1,(p,p));G=rng.uniform(.1,1,(p,p))
        if topology=='sparse':
            mf=rng.random((p,p))<.05;mg=rng.random((p,p))<.05
            # Explicit alternating cycle guarantees strong connectivity; no seed filtering.
            mf[np.arange(p),np.arange(p)]=True
            mg[np.arange(p),(np.arange(p)+1)%p]=True
            F*=mf;G*=mg
    F/=F.sum(axis=1).mean();G/=G.sum(axis=1).mean()
    d=rng.uniform(.8,1.2,n);w=np.array([0]*8+[1]*8);rng.shuffle(w)
    return F,G,d,w

def sparse_operator(op):
    return s.Operator(op.d.copy(),csr_matrix(op.X),csr_matrix(op.Y),op.word.copy())

def perron(op):
    vals,vs=eig(op.dense());j=int(np.argmax(abs(vals)));q=float(abs(vals[j]));v=vs[:,j].real
    if v.sum()<0:v=-v
    assert min(v)>0
    return q,v/max(v)

def calibrate(F,G,d,w,target):
    def f(beta):
        op=s.prepare(F,G,d,beta,w)
        q=abs(eigs(LinearOperator((len(d),len(d)),matvec=op.apply,dtype=float),k=1,which='LM',v0=np.ones(len(d)),tol=1e-12,maxiter=2000,return_eigenvectors=False)[0])
        return np.log(q)/op.period-target
    upper=8.
    while f(upper)<0:upper*=2
    return brentq(f,0,upper,xtol=1e-12)

def screen(op,v,guard=1e-10,budget=16,profile=False):
    t0=time.perf_counter();vv=np.array(v,copy=True);vv/=max(vv);lo=0.;hi=float('inf');hist=[];apply_s=0.;reduce_s=0.;norm_s=0.
    for k in range(1,budget+1):
        t=time.perf_counter();y=op.apply(vv);apply_s+=time.perf_counter()-t
        t=time.perf_counter();rat=y/vv;lo=max(lo,float(min(rat)));hi=min(hi,float(max(rat)));ver=-1 if hi<1-guard else 1 if lo>1+guard else 0;reduce_s+=time.perf_counter()-t
        hist.append((lo,hi,ver))
        # Retain a positive candidate-derived vector even if already decisive.
        if ver and profile:
            nextv=vv  # Profile the original screen, which stops before normalization.
        else:
            t=time.perf_counter();nextv=y/max(y);norm_s+=time.perf_counter()-t
        if ver:break
        vv=nextv
    return dict(verdict=ver,steps=k,lo=lo,hi=hi,history=hist,vector=nextv,total_s=time.perf_counter()-t0,apply_s=apply_s,reduce_s=reduce_s,norm_s=norm_s)

def mutate_random(rng,F,G,d,w,step):
    f=F.copy();g=G.copy();dd=d.copy();ww=w.copy();p=len(F);kind=['weight','loss','redistribute','order','joint'][(step-1)%5]
    row=int(rng.integers(p));row2=int(rng.integers(p));amount=float(rng.choice([-.1,-.05,.05,.1]))
    if kind in ['weight','joint']:f[row]*=1+amount
    if kind=='joint':g[row2]*=1+amount
    if kind=='loss':dd[row]*=1+amount
    if kind=='redistribute':
        a=abs(amount);f[row]=(1-a)*f[row]+a*np.roll(f[row],1)
    if kind=='order':
        i=int(rng.integers(len(w)));other=np.flatnonzero(ww!=ww[i]);j=int(rng.choice(other));ww[i],ww[j]=ww[j],ww[i]
    else:i=j=-1
    return f,g,dd,ww,dict(kind=kind,row=row,row2=row2,amount=amount,swap_i=i,swap_j=j)
