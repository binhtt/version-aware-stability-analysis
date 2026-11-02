"""Independent high-precision and interval audit near the stability boundary.
The interval conclusions refer to supplied binary64 model inputs treated as exact.
The reference-initialized guard controls isolate arithmetic behavior, NOT warm reuse performance.
"""
from pathlib import Path
import json,time
import numpy as np
import mpmath as mp
from scipy.optimize import brentq
import baseline_core as s
from evolution_core import ROOT,model,perron,screen,write_csv,TOPOLOGIES
SEED=2026110104
mp.mp.dps=100
mp.iv.dps=80
TARGETS=[-.01,-1e-6,-1e-10,-1e-12,0.,1e-12,1e-10,1e-6,.01]
GUARDS=[0.,1e-14,1e-12,1e-10,1e-8]

def coefficients(F,G,d,beta,h,interval=False):
    ctx=mp.iv if interval else mp.mp
    conv=lambda x:ctx.mpf(float(x))
    ds=[conv(x) for x in d];hh=conv(h);bb=conv(beta);a=[ctx.exp(-x*hh) for x in ds];diag=[x*x for x in a];p=len(F)
    def c(i,j,w):
        t=hh*a[i] if d[i]==d[j] else (a[j]-a[i])/(ds[i]-ds[j])
        return bb*conv(w)*t*a[i]
    X=[[c(i,p+j,F[i,j]) for j in range(p)] for i in range(p)]
    Y=[[c(p+i,j,G[i,j]) for j in range(p)] for i in range(p)]
    return diag,X,Y

def phase(coef,z,k):
    d,X,Y=coef;p=len(X)
    if k==0:return [d[i]*z[i]+sum(X[i][j]*z[p+j] for j in range(p)) for i in range(p)]+[d[p+i]*z[p+i] for i in range(p)]
    return [d[i]*z[i] for i in range(p)]+[d[p+i]*z[p+i]+sum(Y[i][j]*z[j] for j in range(p)) for i in range(p)]

def apply(coef,z,word):
    for k in word:z=phase(coef,z,int(k))
    return z

def mp_perron(coef,word):
    n=len(coef[0]);M=mp.eye(n)
    for k in word:
        columns=[phase(coef,[M[i,j] for i in range(n)],int(k)) for j in range(n)]
        M=mp.matrix([[columns[j][i] for j in range(n)] for i in range(n)])
    vals,vs=mp.eig(M,left=False,right=True);j=max(range(n),key=lambda j:abs(vals[j]));q=mp.re(vals[j]);v=[mp.re(vs[i,j]) for i in range(n)]
    if sum(v)<0:v=[-x for x in v]
    assert min(v)>0
    vv=[x/max(v) for x in v]
    residual=max(abs(x-q*y) for x,y in zip(M*mp.matrix(vv),vv))
    assert residual<mp.mpf('1e-85')
    return q,vv,residual

def iv_bounds(coef,word,vector):
    # Decimal strings define exact positive test-vector entries; any positive vector is valid.
    v=[mp.iv.mpf(mp.nstr(x,95)) for x in vector]
    z=apply(coef,v,word);rat=[z[i]/v[i] for i in range(len(v))]
    lo=min(r.a for r in rat);hi=max(r.b for r in rat)
    ver=-1 if hi<1 else 1 if lo>1 else 0
    return ver,lo,hi

def endpoint(x,lower=True):
    return mp.mpf(x._mpi_[0 if lower else 1])

def main():
    rng=np.random.default_rng(SEED);records=[];guards=[];models=[];cache_records=[];start=time.perf_counter();cid=0
    design=dict(seed=SEED,sizes=[6,12,24],topologies=TOPOLOGIES,h=[.005,.125],target_growth_rates=TARGETS,guards=GUARDS,mp_dps=100,interval_dps=80,controls='Reference-vector initialization isolates boundary arithmetic; exact beta values remain binary64 and target zero is NOT assumed to be an exact stability boundary')
    (ROOT/'numerical_design.json').write_text(json.dumps(design,indent=2))
    for h in [.005,.125]:
        s.H=h
        for n in [6,12,24]:
            for topology in TOPOLOGIES:
                F,G,d,w=model(rng,n,topology)
                group=len(models);models.append(dict(n=n,h=h,topology=topology,F=F.tolist(),G=G.tolist(),losses=d.tolist(),word=w.tolist()))
                for target in TARGETS:
                    beta=brentq(lambda b:np.log(s.reference(s.prepare(F,G,d,b,w)))/(2*h*len(w))-target,0,32,xtol=5e-15,rtol=4*np.finfo(float).eps)
                    op=s.prepare(F,G,d,beta,w);qd,vd=perron(op)
                    coef=coefficients(F,G,d,beta,h);qm,vm,resid=mp_perron(coef,w)
                    ivc=coefficients(F,G,d,beta,h,True);ivver,lo,hi=iv_bounds(ivc,w,vm)
                    # Actual signs are determined by interval inequalities, not target labels or dense rounding.
                    sign=-1 if qm<1 else 1 if qm>1 else 0
                    assert not ivver or ivver==sign
                    if ivver:assert ivver==sign
                    maxerr=abs(mp.mpf(float(qd))-qm)
                    rec=dict(case=cid,model=group,n=n,h=h,period=op.period,topology=topology,target=target,beta=beta,actual_mp_lambda=mp.nstr(mp.log(qm)/mp.mpf(float(op.period)),45),rho_dense=qd,rho_mp=mp.nstr(qm,85),interval_verdict=ivver,interval_lo=str(lo),interval_hi=str(hi),interval_width=mp.nstr(endpoint(hi,False)-endpoint(lo,True),15),dense_rho_abs_error=mp.nstr(maxerr,15),mp_eigen_residual=mp.nstr(resid,15))
                    records.append(rec)
                    for guard in GUARDS:
                        rr=screen(op,vd,guard=guard)
                        mismatch=int(rr['verdict']!=0 and ivver!=0 and rr['verdict']!=ivver)
                        guards.append(dict(case=cid,n=n,h=h,topology=topology,target=target,guard=guard,float_verdict=rr['verdict'],interval_verdict=ivver,steps=rr['steps'],mismatch=mismatch,lo=rr['lo'],hi=rr['hi']))
                    cid+=1
                (ROOT/'near_boundary_intervals.json').write_text(json.dumps(records,indent=2));write_csv('guard_sweep.csv',guards)
                print('numeric',h,n,topology,'cases',cid,'elapsed',time.perf_counter()-start,flush=True)
    (ROOT/'numerical_models.json').write_text(json.dumps(models))
    # Interval validation of the composed cache layer, separately from direct interval action.
    s.H=.125
    for n in [6,12]:
        for topology in TOPOLOGIES:
            for rep in range(2):
                F,G,d,w=model(rng,n,topology);beta=brentq(lambda b:np.log(s.reference(s.prepare(F,G,d,b,w)))/4+.002,0,32)
                old=s.prepare(F,G,d,beta,w);_,vold=perron(old);co=coefficients(F,G,d,beta,.125,True)
                v0=[mp.iv.mpf(float(x)) for x in vold];states=[v0]
                for k in w:states.append(phase(co,states[-1],int(k)))
                qr=[states[-1][i]/v0[i] for i in range(n)];qlo=min(r.a for r in qr);qhi=max(r.b for r in qr)
                for direction in [-1,1]:
                    ff=F.copy();ff[0]*=1+direction*.2;cn=coefficients(ff,G,d,beta,.125,True);L=qlo;U=qhi
                    for k,symbol in enumerate(w):
                        yy=phase(cn,states[k],int(symbol));rat=[yy[i]/states[k+1][i] for i in range(n)];L*=min(x.a for x in rat);U*=max(x.b for x in rat)
                    ver=-1 if U<1 else 1 if L>1 else 0
                    new=s.prepare(ff,G,d,beta,w);_,vnum=perron(new)
                    direct,dl,du=iv_bounds(cn,w,[mp.mpf(float(x)) for x in vnum])
                    assert direct!=0
                    assert not ver or ver==direct
                    cache_records.append(dict(n=n,topology=topology,rep=rep,weight_change=direction*.2,cache_interval_verdict=ver,direct_interval_verdict=direct,cache_lo=str(L.a),cache_hi=str(U.b),direct_lo=str(dl),direct_hi=str(du)))
    (ROOT/'cache_interval_checks.json').write_text(json.dumps(cache_records,indent=2))
    (ROOT/'numerical_metadata.json').write_text(json.dumps(dict(n_boundary_cases=len(records),n_guard_evaluations=len(guards),n_cache_interval_cases=len(cache_records),elapsed_s=time.perf_counter()-start,mpmath=mp.__version__,default_guard_mismatches=sum(r['mismatch'] for r in guards if r['guard']==1e-10)),indent=2))
    print('NUMERICAL AUDIT COMPLETE',len(records),len(cache_records),'elapsed',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
