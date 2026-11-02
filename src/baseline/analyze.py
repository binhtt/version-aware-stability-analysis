"""Generate manuscript-facing tables, figures and a candid Vietnamese report."""
from pathlib import Path
import csv,json,platform,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=(Path(__file__).resolve().parents[2] / 'results' / 'baseline')

def read(name):
    rows=list(csv.DictReader((P/name).open()))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v)
            except ValueError:pass
    return rows

def mean(rs,key):return float(np.mean([r[key] for r in rs]))
def median(rs,key):return float(np.median([r[key] for r in rs]))
def save(name,rows):
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

def ci_cluster(rs,key,B=2000):
    """Stratified paired cluster bootstrap: config is a cluster, strata are n."""
    rng=np.random.default_rng(202610291);sizes=sorted({r['n'] for r in rs});strata=[]
    for n in sizes:
        rr=[r for r in rs if r['n']==n];reps=sorted({r['rep'] for r in rr})
        vals=[]
        for rep in reps:
            cr=[r for r in rr if r['rep']==rep]
            vals.append(float(np.mean([key(r) if callable(key) else r[key] for r in cr])))
        strata.append(np.array(vals))
    draws=np.zeros(B)
    for vals in strata:draws+=vals[rng.integers(0,len(vals),size=(B,len(vals)))].mean(axis=1)/len(strata)
    return np.quantile(draws,[.025,.975]).tolist()

def main():
    rows=read('cases.csv');ts=read('timings.csv');ext=read('duration_sensitivity.csv');base=read('baselines.csv');m=json.loads((P/'metadata.json').read_text());em=json.loads((P/'extensions_metadata.json').read_text())
    stats=[]
    for reg in ['core','far','near','critical']:
        rr=[r for r in rows if r['regime']==reg]
        for budget in [1,2,4,8,16]:
            warm=mean(rr,f'warm_resolved_{budget}');cold=mean(rr,f'cold_resolved_{budget}')
            ci=ci_cluster(rr,lambda r:r[f'warm_resolved_{budget}']-r[f'cold_resolved_{budget}'])
            stats.append(dict(regime=reg,budget=budget,n=len(rr),warm_rate=warm,cold_rate=cold,difference=warm-cold,paired_cluster_ci_low=ci[0],paired_cluster_ci_high=ci[1]))
    save('resolution_summary.csv',stats)
    timing=[]
    for n in m['sizes']:
        rr=[r for r in ts if r['n']==n]
        timing.append(dict(n=n,n_scenarios=len(rr),dense_ms=1000*median(rr,'dense_median_s'),warm_ms=1000*median(rr,'warm_median_s'),cold_ms=1000*median(rr,'cold_median_s'),arpack_ms=1000*median(rr,'arpack_median_s'),median_paired_dense_over_warm=float(np.median([r['dense_median_s']/r['warm_median_s'] for r in rr])),median_paired_cold_over_warm=float(np.median([r['cold_median_s']/r['warm_median_s'] for r in rr])),max_arpack_error=max(r['arpack_max_rho_error'] for r in rr)))
    save('timing_summary.csv',timing)
    durations=[]
    for h in [.005,.025,.125,.5]:
        rr=[r for r in ext if r['h']==h]
        ci=ci_cluster(rr,lambda r:r['warm_resolved_16']-r['cold_resolved_16'])
        durations.append(dict(h=h,period=32*h,n=len(rr),warm1=mean(rr,'warm_resolved_1'),cold1=mean(rr,'cold_resolved_1'),warm16=mean(rr,'warm_resolved_16'),cold16=mean(rr,'cold_resolved_16'),warm_steps=mean(rr,'warm_steps'),cold_steps=mean(rr,'cold_steps'),difference_ci_low=ci[0],difference_ci_high=ci[1]))
    save('duration_summary.csv',durations)
    # Failure and bound checks are assertions in study and extensions, retained logs too.
    wrong=[r for r in rows if any(r[x+'_verdict'] and r[x+'_verdict']!=r['truth'] for x in ['warm','cold'])]
    assert not wrong
    # Main resolution curves.
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for ax,reg in zip(axs,['core','near','critical']):
        rr=[r for r in stats if r['regime']==reg]
        ax.plot([r['budget'] for r in rr],[100*r['warm_rate'] for r in rr],'-o',label='Previous-version vector')
        ax.plot([r['budget'] for r in rr],[100*r['cold_rate'] for r in rr],'--s',label='Ones vector')
        ax.set(xlabel='Period passes (budget)',title=reg,ylim=(-3,103));ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.grid(alpha=.2)
    axs[0].set_ylabel('Resolved decisions (%)');axs[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(P/'resolution.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for key,label in [('dense_ms','Dense product + eig'),('warm_ms','Warm CW screen'),('cold_ms','Cold CW screen'),('arpack_ms','Matrix-free ARPACK (point estimate)')]:
        ax.loglog([r['n'] for r in timing],[r[key] for r in timing],'-o',label=label)
    ax.set(xlabel='Number of agents',ylabel='Median online time (ms)',title='Timing: 9 cases/size, 7 repeats/case, one BLAS thread');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(P/'timing.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5));x=np.arange(4);width=.35
    ax.bar(x-width/2,[100*r['warm16'] for r in durations],width,label='Previous-version vector');ax.bar(x+width/2,[100*r['cold16'] for r in durations],width,label='Ones vector');ax.set_xticks(x,[str(r['period']) for r in durations]);ax.set(xlabel='Period duration P (exploratory extension)',ylabel='Resolved within 16 passes (%)',ylim=(0,108));ax.legend();fig.tight_layout();fig.savefig(P/'duration.png',dpi=180);plt.close(fig)

if __name__=='__main__': main()
