PY := .venv/bin/python
PYFLAGS := -u

.PHONY: all layers analyses tables clean help

all: layers analyses tables

layers: data/synthetic_hiring.csv
	$(PY) $(PYFLAGS) layer1_proxy_lfr.py
	$(PY) $(PYFLAGS) layer2_dag_scm.py
	$(PY) $(PYFLAGS) layer3_metrics.py
	$(PY) $(PYFLAGS) layer4_fair_training.py
	$(PY) $(PYFLAGS) layer5_explain.py
	$(PY) $(PYFLAGS) layer6_router.py

analyses:
	$(PY) $(PYFLAGS) gamma_sensitivity.py
	$(PY) $(PYFLAGS) ablation.py
	$(PY) $(PYFLAGS) shap_vs_cefhf.py

tables:
	$(PY) $(PYFLAGS) make_tables.py

data/synthetic_hiring.csv:
	$(PY) $(PYFLAGS) synthetic_scm.py

run_all:
	$(PY) $(PYFLAGS) run_all.py

clean:
	rm -rf outputs/*.csv outputs/*.png outputs/*.md outputs/*.json data/*.pkl __pycache__

help:
	@echo "make all        - run every layer + analysis + tables"
	@echo "make layers     - run layers 1-6 (+ synthetic ground truth)"
	@echo "make analyses   - gamma-sweep, ablation, shap-vs-cefhf"
	@echo "make tables     - assemble final tables/figures"
	@echo "make run_all    - python run_all.py end-to-end"