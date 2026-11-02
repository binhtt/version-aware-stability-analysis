from pathlib import Path
import json,time,platform,os
import numpy as np,scipy
import baseline_core as s
import cache_core as cache
from evolution_core import *

SEQUENTIAL_SEED=2026110101
TOPOLOGY_SEED=2026110102
PROFILE_SEED=2026110103

def sequential():
    rng=np.random.default_rng(SEQUENTIAL_SEED);rows=[];configs=[];begin=time.perf_counter();cid=0
    for h in [.005,.125]:
        s.H=h
        for n in [24,96]:
            for topology in TOPOLOGIES:
                for rep in range(3):
                    F,G,d,w=model(rng,n,topology);beta=calibrate(F,G,d,w,-.02);old=s.prepare(F,G,d,beta,w);q0,v0=perron(old)
                    assert abs(np.log(q0)/old.period+.02)<1e-8
                    carried=v0.copy();fixed=v0.copy();chain=dict(chain=cid,n=n,h=h,topology=topology,rep=rep,beta=beta,F=F.tolist(),G=G.tolist(),losses=d.tolist(),word=w.tolist(),updates=[])
                    for step in range(1,41):
                        F,G,d,w,delta=mutate_random(rng,F,G,d,w,step);chain['updates'].append(delta)
                        t=time.perf_counter();op=s.prepare(F,G,d,beta,w);prep=time.perf_counter()-t
                        t=time.perf_counter();q,vref=perron(op);refsec=time.perf_counter()-t
                        truth=-1 if q<1-1e-10 else 1 if q>1+1e-10 else 0
                        for method,v in [('carried',carried),('fixed_initial',fixed),('cold',np.ones(n))]:
                            r=screen(op,v)
                            assert not r['verdict'] or r['verdict']==truth
                            assert r['lo']<=q+1e-8 and q<=r['hi']+1e-8
                            rows.append(dict(chain=cid,n=n,h=h,period=op.period,topology=topology,rep=rep,step=step,kind=delta['kind'],method=method,lambda_new=np.log(q)/op.period,truth=truth,verdict=r['verdict'],steps=r['steps'],resolved1=int(any(x[2] for x in r['history'][:1])),resolved4=int(any(x[2] for x in r['history'][:4])),resolved16=int(r['verdict']!=0),prep_s=prep,screen_s=r['total_s'],reference_s=refsec))
                            if method=='carried':carried=r['vector']
                        # All updates are applied to fixed analysis chains, including unstable/unknown versions.
                        # Reference q/vector never initializes carried or fixed/cold methods.
                    configs.append(chain);cid+=1
                write_csv('sequential.csv',rows)
                print('sequential',h,n,topology,'chains',cid,'elapsed',time.perf_counter()-begin,flush=True)
    (ROOT/'sequential_configs.json').write_text(json.dumps(configs))
    return dict(n_chains=cid,n_versions=cid*40,n_method_evaluations=len(rows),seed=SEQUENTIAL_SEED,elapsed_s=time.perf_counter()-begin,semantics='Unconditional analysis chains, NOT autonomous deployment/acceptance; carried final positive iterate reused without oracle refresh')

def topology_scaling():
    rng=np.random.default_rng(TOPOLOGY_SEED);rows=[];configs=[];begin=time.perf_counter()
    s.H=.125
    for n in [96,192,384,768]:
        for topology in TOPOLOGIES:
            for rep in range(3):
                F,G,d,w=model(rng,n,topology);beta=calibrate(F,G,d,w,-.02);old=s.prepare(F,G,d,beta,w);q0,v=perron(old)
                assert abs(np.log(q0)/old.period+.02)<1e-8
                configs.append(dict(n=n,topology=topology,rep=rep,beta=beta,F=F.tolist(),G=G.tolist(),losses=d.tolist(),word=w.tolist()))
                density=(np.count_nonzero(F)+np.count_nonzero(G))/(2*len(F)**2)
                cv=float(np.std(np.r_[F[F>0],G[G>0]])/np.mean(np.r_[F[F>0],G[G>0]]))
                for scope in ['one_row','ten_percent']:
                    rcount=1 if scope=='one_row' else max(1,int(np.ceil(.1*len(F))))
                    for kind in ['weight','joint','loss']:
                        ff=F.copy();gg=G.copy();dd=d.copy()
                        if kind in ['weight','joint']:ff[:rcount]*=1.1
                        if kind=='joint':gg[:rcount]*=1.1
                        if kind=='loss':dd[:rcount]*=.9
                        t=time.perf_counter();op=s.prepare(ff,gg,dd,beta,w);prep=time.perf_counter()-t
                        t=time.perf_counter();q=s.reference(op);dense=time.perf_counter()-t
                        truth=-1 if q<1-1e-10 else 1 if q>1+1e-10 else 0
                        tt={m:[] for m in ['warm','cold']};res={}
                        for repeat in range(5):
                            methods=['warm','cold'] if repeat%2==0 else ['cold','warm']
                            for method in methods:
                                t=time.perf_counter();rr=s.screen(op,v if method=='warm' else np.ones(n));tt[method].append(time.perf_counter()-t);res[method]=rr
                        sparse_setup=0.;sparse_time=np.nan
                        if topology=='sparse':
                            t=time.perf_counter();sop=sparse_operator(op);sparse_setup=time.perf_counter()-t;st=[]
                            for repeat in range(5):
                                t=time.perf_counter();sr=s.screen(sop,v);st.append(time.perf_counter()-t)
                                assert sr[0]==res['warm'][0]
                            sparse_time=float(np.median(st))
                        # Dedicated repeated dense timing on a fixed subset; no outcome selection.
                        repeat_dense=[]
                        if rep==0 and kind=='weight':
                            for _ in range(5):
                                t=time.perf_counter();s.reference(op);repeat_dense.append(time.perf_counter()-t)
                        fields={}
                        for method,(ver,k,lo,hi,hist) in res.items():
                            assert not ver or ver==truth
                            fields.update({method+'_verdict':ver,method+'_steps':k,method+'_onepass':int(hist[0][2]!=0),method+'_median_s':float(np.median(tt[method]))})
                        rows.append(dict(n=n,topology=topology,rep=rep,scope=scope,changed_rows=rcount,kind=kind,density=density,weight_cv=cv,lambda_new=np.log(q)/op.period,truth=truth,prep_s=prep,dense_s=dense,dense_repeat_median_s=float(np.median(repeat_dense)) if repeat_dense else np.nan,sparse_setup_s=sparse_setup,sparse_warm_median_s=sparse_time,**fields))
            write_csv('topology_scaling.csv',rows)
            print('topology',n,topology,'cases',len(rows),'elapsed',time.perf_counter()-begin,flush=True)
    (ROOT/'topology_configs.json').write_text(json.dumps(configs))
    return dict(n_cases=len(rows),seed=TOPOLOGY_SEED,elapsed_s=time.perf_counter()-begin,scope='Structured two-population architecture retained; sparse support preserves an alternating Hamiltonian cycle; 10% scope scales receiving rows per population; density=nonzeros/(2p^2)')

def cost_profile():
    rng=np.random.default_rng(PROFILE_SEED);rows=[];begin=time.perf_counter();s.H=.125
    for n in [24,96,192]:
        for rep in range(3):
            F,G,d,w=model(rng,n,'dense')
            for regime,target in [('far',-.2),('near',-.02),('critical',-.002)]:
                beta=calibrate(F,G,d,w,target);old=s.prepare(F,G,d,beta,w);qold,v=perron(old)
                t=time.perf_counter();cached=cache.cache_trajectory(old,v);cache_offline=time.perf_counter()-t
                for kind in ['weight','joint','order']:
                    ff,gg,dd,ww=s.mutate(F,G,d,w,kind,.1)
                    new=s.prepare(ff,gg,dd,beta,ww);q=s.reference(new);truth=-1 if q<1-1e-10 else 1 if q>1+1e-10 else 0
                    dt=[];ct=[];fall=[];prof={'warm':[],'cold':[]};plain={'warm':[],'cold':[]};warmres=coldres=None
                    for repeat in range(11):
                        methods=['warm','cold'] if repeat%2==0 else ['cold','warm']
                        for method in methods:
                            vv=v if method=='warm' else np.ones(n)
                            t=time.perf_counter();r=s.screen(new,vv);plain[method].append(time.perf_counter()-t)
                            pr=screen(new,vv,profile=True);prof[method].append(pr)
                            assert pr['verdict']==r[0] and pr['steps']==r[1]
                            assert not r[0] or r[0]==truth
                        t=time.perf_counter();delta=cache.delta_prepare(old,new);dt.append(time.perf_counter()-t)
                        t=time.perf_counter();cv,lo,hi=cache.delta_bounds(old,new,cached,delta);ct.append(time.perf_counter()-t)
                        assert not cv or cv==truth
                    fields={}
                    for method in ['warm','cold']:
                        fields[method+'_steps']=prof[method][0]['steps'];fields[method+'_median_s']=float(np.median(plain[method]))
                        for part in ['apply_s','reduce_s','norm_s','total_s']:fields[method+'_profile_'+part]=float(np.median([r[part] for r in prof[method]]))
                    prepmed=float(np.median(dt));cachemed=float(np.median(ct))
                    rows.append(dict(n=n,rep=rep,regime=regime,kind=kind,cache_decisive=int(cv!=0),delta_prepare_s=prepmed,cache_eval_s=cachemed,cache_offline_s=cache_offline,hybrid_estimated_s=prepmed+cachemed+(fields['warm_median_s'] if not cv else 0),**fields))
        write_csv('cost_profile.csv',rows)
    print('profile',len(rows),'elapsed',time.perf_counter()-begin,flush=True)
    return dict(n_cases=len(rows),seed=PROFILE_SEED,repeats=11,elapsed_s=time.perf_counter()-begin,scope='Instrumented partition times reported separately from uninstrumented full screen timings; derived hybrid is an additive cost accounting, not an independent timer')

if __name__=='__main__':
    design=dict(sequential=dict(sizes=[24,96],h=[.005,.125],topologies=TOPOLOGIES,chains_per_cell=3,updates_per_chain=40),topology=dict(sizes=[96,192,384,768],topologies=TOPOLOGIES,reps=3,scopes=['one_row','ten_percent'],kinds=['weight','joint','loss']),profile=dict(sizes=[24,96,192],reps=3,regimes=['far','near','critical'],kinds=['weight','joint','order'],timing_repeats=11))
    (ROOT/'design.json').write_text(json.dumps(design,indent=2));meta={}
    for name,fn in [('sequential',sequential),('topology',topology_scaling),('profile',cost_profile)]:
        meta[name]=fn();(ROOT/'extensions_progress.json').write_text(json.dumps(meta,indent=2))
    meta['environment']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,machine=platform.machine(),threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']})
    (ROOT/'extensions_metadata.json').write_text(json.dumps(meta,indent=2));print('ALL NONNUMERICAL STAGES COMPLETE',flush=True)
