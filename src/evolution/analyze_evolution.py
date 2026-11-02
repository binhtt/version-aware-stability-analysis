from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=(Path(__file__).resolve().parents[2] / 'results' / 'evolution')

def read(name):
    rows=list(csv.DictReader((P/name).open()))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v)
            except ValueError:pass
    return rows

def avg(rs,k):return float(np.mean([r[k] for r in rs]))
def med(rs,k):return float(np.median([r[k] for r in rs]))
def save(name,rows):
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    seq=read('sequential.csv');top=read('topology_scaling.csv');profile=read('cost_profile.csv');guard=read('guard_sweep.csv');iv=json.loads((P/'near_boundary_intervals.json').read_text());civ=json.loads((P/'cache_interval_checks.json').read_text());tay=json.loads((P/'independent_interval_expm.json').read_text())
    ss=[]
    for h in [.005,.125]:
        for method in ['carried','fixed_initial','cold']:
            r=[x for x in seq if x['h']==h and x['method']==method]
            ss.append(dict(period=32*h,method=method,cases=len(r),onepass=avg(r,'resolved1'),fourpass=avg(r,'resolved4'),resolved16=avg(r,'resolved16'),mean_passes=avg(r,'steps'),median_screen_ms=1000*med(r,'screen_s')))
    save('sequential_summary.csv',ss)
    # Paired chain-level bootstrap stratified by topology and n (3 independent chains per cell).
    ci=[];rng=np.random.default_rng(2026110105)
    for h in [.005,.125]:
        draws=np.zeros(2000)
        for n in [24,96]:
            for topology in ['dense','sparse','skewed']:
                vals=[]
                chains=sorted({r['chain'] for r in seq if r['h']==h and r['n']==n and r['topology']==topology})
                for chain in chains:
                    a=[r for r in seq if r['chain']==chain and r['method']=='carried'];b=[r for r in seq if r['chain']==chain and r['method']=='cold'];vals.append(avg(a,'resolved16')-avg(b,'resolved16'))
                vals=np.array(vals);draws+=vals[rng.integers(0,3,(2000,3))].mean(axis=1)/6
        lo,hi=np.quantile(draws,[.025,.975]);ci.append(dict(period=32*h,difference_ci_low=lo,difference_ci_high=hi))
    save('sequential_ci.csv',ci)
    topology_summary=[]
    for n in [96,192,384,768]:
        for topology in ['dense','sparse','skewed']:
            for scope in ['one_row','ten_percent']:
                r=[x for x in top if x['n']==n and x['topology']==topology and x['scope']==scope]
                topology_summary.append(dict(n=n,topology=topology,scope=scope,cases=len(r),density=avg(r,'density'),weight_cv=avg(r,'weight_cv'),warm_resolution=np.mean([x['warm_verdict']!=0 for x in r]),cold_resolution=np.mean([x['cold_verdict']!=0 for x in r]),warm_passes=avg(r,'warm_steps'),cold_passes=avg(r,'cold_steps'),warm_ms=1000*med(r,'warm_median_s'),cold_ms=1000*med(r,'cold_median_s'),prepare_ms=1000*med(r,'prep_s'),sparse_setup_ms=1000*med(r,'sparse_setup_s'),sparse_warm_ms=1000*med(r,'sparse_warm_median_s')))
    save('topology_summary.csv',topology_summary)
    # Matched, repeated timing subset: first config, weight update, both scopes.
    scaling=[]
    for n in [96,192,384,768]:
        for topology in ['dense','sparse','skewed']:
            r=[x for x in top if x['n']==n and x['topology']==topology and x['rep']==0 and x['kind']=='weight']
            assert len(r)==2
            scaling.append(dict(n=n,topology=topology,matched_scenarios=2,dense_ms=1000*med(r,'dense_repeat_median_s'),warm_ms=1000*med(r,'warm_median_s'),cold_ms=1000*med(r,'cold_median_s'),sparse_ms=1000*med(r,'sparse_warm_median_s'),prepare_ms=1000*med(r,'prep_s'),sparse_setup_ms=1000*med(r,'sparse_setup_s')))
    save('scaling_timing_summary.csv',scaling)
    ps=[]
    for reg in ['far','near','critical']:
        r=[x for x in profile if x['regime']==reg]
        equal=[x for x in r if x['warm_steps']==x['cold_steps']]
        ps.append(dict(regime=reg,cases=len(r),warm_passes=avg(r,'warm_steps'),cold_passes=avg(r,'cold_steps'),equal_pass_cases=len(equal),cold_faster_equal=sum(x['cold_median_s']<x['warm_median_s'] for x in equal),cache_decisive=np.mean([x['cache_decisive']!=0 for x in r]),delta_prepare_us=1e6*med(r,'delta_prepare_s'),cache_eval_us=1e6*med(r,'cache_eval_s'),warm_us=1e6*med(r,'warm_median_s'),cold_us=1e6*med(r,'cold_median_s'),derived_hybrid_us=1e6*med(r,'hybrid_estimated_s'),warm_apply_fraction=med([dict(z=x['warm_profile_apply_s']/x['warm_profile_total_s']) for x in r],'z')))
    save('profile_summary.csv',ps)
    gs=[]
    for g in [0,1e-14,1e-12,1e-10,1e-8]:
        r=[x for x in guard if x['guard']==g];gs.append(dict(guard=g,n=len(r),decisive=sum(x['float_verdict']!=0 for x in r),unknown=sum(x['float_verdict']==0 for x in r),wrong=int(sum(x['mismatch'] for x in r))))
    save('guard_summary.csv',gs)
    # Figures.
    colors={'carried':'#176b93','fixed_initial':'#d18419','cold':'#76818a'}
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),sharey=False)
    for ax,h in zip(axs,[.005,.125]):
        for method in colors:
            y=[]
            for a,b in [(1,10),(11,20),(21,30),(31,40)]:
                r=[x for x in seq if x['h']==h and x['method']==method and a<=x['step']<=b];y.append(avg(r,'steps'))
            ax.plot([5.5,15.5,25.5,35.5],y,'-o',color=colors[method],label=method.replace('_',' '))
        ax.set(title=f'Period P = {32*h:g}',xlabel='Update position (10-version blocks)',xticks=[5.5,15.5,25.5,35.5],xticklabels=['1-10','11-20','21-30','31-40']);ax.grid(alpha=.2)
    axs[0].set_ylim(0,17);axs[1].set_ylim(0,4);axs[0].set_ylabel('Mean passes (capped at 16)');axs[1].set_ylabel('Mean passes');axs[1].legend();fig.tight_layout();fig.savefig(P/'sequential_versions.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,topology in zip(axs,['dense','sparse','skewed']):
        rr=[r for r in scaling if r['topology']==topology]
        for key,label in [('dense_ms','Dense reference'),('warm_ms','Warm (dense blocks)'),('cold_ms','Cold (dense blocks)')]:ax.loglog([r['n'] for r in rr],[r[key] for r in rr],'-o',label=label)
        if topology=='sparse':ax.loglog([r['n'] for r in rr],[r['sparse_ms'] for r in rr],'-o',label='Warm (CSR action)')
        from matplotlib.ticker import NullLocator
        ax.set(title=topology,xlabel='Agents');ax.set_xticks([96,192,384,768],['96','192','384','768']);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.2)
    axs[0].set_ylabel('Online time (ms)');handles,labels=axs[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(P/'topology_scaling.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.2));x=np.arange(5);ax.bar(x,[g['decisive'] for g in gs],label='Decisive');ax.bar(x,[g['unknown'] for g in gs],bottom=[g['decisive'] for g in gs],label='Unknown');ax.set_xticks(x,['0','1e-14','1e-12','1e-10','1e-8']);ax.set(xlabel='Numerical guard',ylabel='Boundary controls (n=162)',title='Guard ablation: zero guard has one wrong decisive verdict');ax.legend();fig.tight_layout();fig.savefig(P/'guard_sensitivity.png',dpi=200);plt.close(fig)
    print('SEQUENTIAL',json.dumps(ss,indent=2));print('SCALING',json.dumps([r for r in scaling if r['n']>=384],indent=2));print('PROFILE',json.dumps(ps,indent=2));print('GUARD',json.dumps(gs,indent=2));print('INDEPENDENT TAYLOR',len(tay));print('CACHE IV decisive',sum(x['cache_interval_verdict']!=0 for x in civ),'of',len(civ))

if __name__=='__main__': main()
