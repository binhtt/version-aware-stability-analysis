"""Replay and persist cache interval checks, also checking the actual float64 cache verdicts."""
import json,numpy as np,mpmath as mp
from scipy.optimize import brentq
import baseline_core as s
import cache_core as c
from evolution_core import ROOT,model,TOPOLOGIES,perron
from numerical_audit import SEED,coefficients,phase,iv_bounds

def main():
    rng=np.random.default_rng(SEED)
    # Reconstruct the generator state preceding cache validation in numerical_audit.py.
    for h in [.005,.125]:
        for n in [6,12,24]:
            for topology in TOPOLOGIES:model(rng,n,topology)
    s.H=.125;records=[];configs=[]
    for n in [6,12]:
        for topology in TOPOLOGIES:
            for rep in range(2):
                F,G,d,w=model(rng,n,topology);beta=brentq(lambda b:np.log(s.reference(s.prepare(F,G,d,b,w)))/4+.002,0,32)
                old=s.prepare(F,G,d,beta,w);_,vold=perron(old);co=coefficients(F,G,d,beta,.125,True)
                configs.append(dict(n=n,topology=topology,rep=rep,beta=beta,F=F.tolist(),G=G.tolist(),losses=d.tolist(),word=w.tolist(),vector=vold.tolist()))
                v0=[mp.iv.mpf(float(x)) for x in vold];states=[v0]
                for k in w:states.append(phase(co,states[-1],int(k)))
                qr=[states[-1][i]/v0[i] for i in range(n)];qlo=min(r.a for r in qr);qhi=max(r.b for r in qr)
                flcache=c.cache_trajectory(old,vold)
                for direction in [-1,1]:
                    ff=F.copy();ff[0]*=1+direction*.2;cn=coefficients(ff,G,d,beta,.125,True);L=qlo;U=qhi
                    for k,symbol in enumerate(w):
                        yy=phase(cn,states[k],int(symbol));rat=[yy[i]/states[k+1][i] for i in range(n)];L*=min(x.a for x in rat);U*=max(x.b for x in rat)
                    ver=-1 if U<1 else 1 if L>1 else 0
                    new=s.prepare(ff,G,d,beta,w);_,vnum=perron(new);direct,dl,du=iv_bounds(cn,w,[mp.mpf(float(x)) for x in vnum]);assert direct!=0
                    fv,fl,fu=c.delta_bounds(old,new,flcache,c.delta_prepare(old,new))
                    assert not ver or ver==direct
                    assert not fv or fv==direct
                    records.append(dict(n=n,topology=topology,rep=rep,weight_change=direction*.2,cache_interval_verdict=ver,direct_interval_verdict=direct,float_cache_verdict=fv,float_cache_lo=fl,float_cache_hi=fu,cache_lo=str(L.a),cache_hi=str(U.b),direct_lo=str(dl),direct_hi=str(du)))
    (ROOT/'cache_interval_checks.json').write_text(json.dumps(records,indent=2));(ROOT/'cache_interval_configs.json').write_text(json.dumps(configs,indent=2));print('Cache interval and float replay passed',len(records))
if __name__=='__main__':main()
