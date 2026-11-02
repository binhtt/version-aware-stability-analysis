"""Run the successive-version, topology, profiling, and numerical studies. Numerical audit is the slowest stage (~2 minutes here)."""
from pathlib import Path
import os,subprocess,sys
root=Path(__file__).resolve().parent
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for script in ['run_extensions.py','numerical_audit.py','independent_interval_expm.py','validate_cache.py','profile_initialization.py','analyze_evolution.py']:
    print('Running',script,flush=True)
    subprocess.run([sys.executable,str(root/script)],cwd=root,env=env,check=True)
print('Evolution studies completed.')
