"""Reproduce studies or regenerate summaries from archived results."""
import argparse, os, subprocess, sys
from pathlib import Path
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode',choices=['summaries','baseline','evolution','all','checks'],default='summaries')
a=parser.parse_args()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
tasks=[]
if a.mode in ('baseline','all'):tasks += [('baseline','reproduce.py')]
if a.mode in ('evolution','all'):tasks += [('evolution','reproduce_evolution.py')]
if a.mode=='summaries':tasks=[('baseline','analyze.py'),('baseline','summarize_extra.py'),('evolution','analyze_evolution.py')]
if a.mode=='checks':tasks=[('baseline','test_checks.py')]
for folder,script in tasks:
    cwd=P/'src'/folder
    print(f'Running {folder}/{script}',flush=True)
    subprocess.run([sys.executable,script],cwd=cwd,env=env,check=True)
print('Completed:',a.mode)
