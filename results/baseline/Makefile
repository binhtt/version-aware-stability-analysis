install:
	python -m pip install -r requirements.txt

check:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_checks.py

reproduce:
	python reproduce.py
