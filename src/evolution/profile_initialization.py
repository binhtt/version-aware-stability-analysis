import json,time,numpy as np
from evolution_core import ROOT,write_csv,perron
import baseline_core as s

def main():
    rows=[]
    for c in json.loads((ROOT/'sequential_configs.json').read_text()):
        s.H=c['h'];t=time.perf_counter();op=s.prepare(np.array(c['F']),np.array(c['G']),np.array(c['losses']),c['beta'],np.array(c['word']));prep=time.perf_counter()-t
        samples=[]
        for _ in range(5):
            t=time.perf_counter();perron(op);samples.append(time.perf_counter()-t)
        rows.append(dict(chain=c['chain'],n=c['n'],h=c['h'],topology=c['topology'],prepare_s=prep,initial_dense_vector_median_s=float(np.median(samples))))
    write_csv('initialization_profile.csv',rows);print('Initial-vector cost profiled separately on',len(rows),'chains')
if __name__=='__main__':main()
