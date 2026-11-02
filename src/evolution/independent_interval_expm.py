"""Independent interval matrix-exponential path on 18 selected boundary controls.
Uses positive Taylor series plus a norm-bounded remainder, not the analytic block formula.
"""
import json,time
import mpmath as mp
import numpy as np
from evolution_core import ROOT
from numerical_audit import coefficients,mp_perron,endpoint
mp.mp.dps=100;mp.iv.dps=80

def expm_interval_generator(F,G,d,beta,h,direction,N=120):
    iv=mp.iv;n=len(d);p=n//2;hh=iv.mpf(float(h));cc=iv.mpf(float(max(d)));bb=iv.mpf(float(beta))
    C=iv.matrix(n,n)
    for i in range(n):C[i,i]=cc-iv.mpf(float(d[i]))
    for i in range(p):
        for j in range(p):
            if direction==0:C[i,p+j]=bb*iv.mpf(float(F[i,j]))
            else:C[p+i,j]=bb*iv.mpf(float(G[i,j]))
    X=C*hh
    x=max(sum(X[i,j] for j in range(n)).b for i in range(n))
    assert x/(N+2)<1
    term=iv.eye(n);S=iv.eye(n)
    for j in range(1,N+1):term=term*X/j;S+=term
    # For ||X|| <= x and nonnegative X, every omitted entry is in [0,delta].
    delta=x**(N+1)/iv.factorial(N+1)/(1-x/(N+2))
    scalar=iv.exp(-cc*hh)
    E=iv.matrix(n,n)
    tail=iv.mpf([0,endpoint(delta,False)])
    for i in range(n):
        idle=iv.exp(-iv.mpf(float(d[i]))*hh)
        for j in range(n):E[i,j]=idle*scalar*(S[i,j]+tail)
    return E,delta

def main():
    recs=json.loads((ROOT/'near_boundary_intervals.json').read_text());models=json.loads((ROOT/'numerical_models.json').read_text());selected=[r for r in recs if r['n']==6 and r['target'] in [-1e-10,0.,1e-10]];assert len(selected)==18
    # Post-hoc independent replay of every mismatch found in the zero-guard sweep.
    import csv
    failures={int(x['case']) for x in csv.DictReader((ROOT/'guard_sweep.csv').open()) if int(x['mismatch'])}
    selected += [r for r in recs if r['case'] in failures and r not in selected]
    result=[];start=time.perf_counter()
    for r in selected:
        m=models[r['model']];F=np.array(m['F']);G=np.array(m['G']);d=np.array(m['losses']);w=m['word'];beta=r['beta'];h=r['h']
        _,v,_=mp_perron(coefficients(F,G,d,beta,h),w)
        EX,dx=expm_interval_generator(F,G,d,beta,h,0);EY,dy=expm_interval_generator(F,G,d,beta,h,1)
        vv=mp.iv.matrix([mp.iv.mpf(mp.nstr(x,95)) for x in v]);z=vv.copy()
        for k in w:z=(EX if k==0 else EY)*z
        rat=[z[i]/vv[i] for i in range(6)];lo=min(x.a for x in rat);hi=max(x.b for x in rat)
        ver=-1 if hi<1 else 1 if lo>1 else 0
        assert ver==r['interval_verdict'],(r['case'],ver,r['interval_verdict'])
        result.append(dict(case=r['case'],topology=r['topology'],h=h,target=r['target'],independent_interval_verdict=ver,analytic_interval_verdict=r['interval_verdict'],lo=str(lo),hi=str(hi),taylor_order=120,tail_X_upper=str(dx.b),tail_Y_upper=str(dy.b)))
    (ROOT/'independent_interval_expm.json').write_text(json.dumps(result,indent=2));print('Independent interval Taylor checks',len(result),'elapsed',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
