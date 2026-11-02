from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import read,mean,median,save,ci_cluster
P=(Path(__file__).resolve().parents[2] / 'results' / 'baseline')

def main():
    cache=read('cache_cases.csv');tim=read('cache_timings.csv');held=read('heldout_cases.csv');hm=json.loads((P/'heldout_metadata.json').read_text())
    cr=[]
    for reg in ['core','far','near','critical']:
        rr=[r for r in cache if r['regime']==reg]
        cr.append(dict(regime=reg,n=len(rr),cache_resolved=float(np.mean([r['cache_verdict']!=0 for r in rr]))))
    hs=[]
    for reg in ['core','far','near','critical']:
        rr=[r for r in held if r['regime']==reg]
        speed=np.median([r['warm_median_s']/r['adaptive_median_s'] for r in rr])
        hs.append(dict(regime=reg,n=len(rr),cache_route=float(np.mean([r['route']=='cache' for r in rr])),adaptive_resolved=float(np.mean([r['adaptive_verdict']!=0 for r in rr])),adaptive_us=1e6*median(rr,'adaptive_median_s'),warm_us=1e6*median(rr,'warm_median_s'),cold_us=1e6*median(rr,'cold_median_s'),paired_warm_over_adaptive=float(speed)))
    save('heldout_summary.csv',hs);save('cache_summary.csv',cr)
    nt=[]
    for n in [12,48,192]:
        for regime in ['core','far','near','critical']:
            rr=[r for r in held if r['n']==n and r['regime']==regime]
            nt.append(dict(n=n,regime=regime,paired_warm_over_adaptive=float(np.median([r['warm_median_s']/r['adaptive_median_s'] for r in rr]))))
    save('heldout_size_summary.csv',nt)
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(len(hs));w=.25
    for off,key,label in [(-w,'adaptive_us','Adaptive'),(0,'warm_us','Warm only'),(w,'cold_us','Cold only')]:ax.bar(x+off,[r[key] for r in hs],w,label=label)
    ax.set_xticks(x,[r['regime'] for r in hs]);ax.set(ylabel='Median time (microseconds)',title='Independent-seed assessment: 450 cases per group');ax.legend();fig.tight_layout();fig.savefig(P/'heldout.png',dpi=180);plt.close(fig)

if __name__=='__main__': main()
