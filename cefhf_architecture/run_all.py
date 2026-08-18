"""
run_all.py -- Master pipeline runner for the CEFHF six-layer architecture.

Runs every layer + analysis end-to-end.  Individual steps can be run directly:
    python synthetic_scm.py
    python layer1_proxy_lfr.py
    ...

Usage:
    python run_all.py            # run everything
    python run_all.py --start 4  # resume from layer 4
"""

import subprocess
import sys
import os
import time

STEPS = [
    ("synthetic_scm.py",       "Layer-2 ground truth: synthetic hiring SCM"),
    ("layer1_proxy_lfr.py",    "Layer 1: proxy detection + D_fair"),
    ("layer2_dag_scm.py",      "Layer 2: DAG + SCM + counterfactuals"),
    ("layer3_metrics.py",      "Layer 3: metric stack (CFVR-SCM, DF eps, WCDE)"),
    ("layer4_fair_training.py","Layer 4: CF-augmentation + fairlearn + lambda-sweep"),
    ("layer5_explain.py",      "Layer 5: causal Shapley + PS + stability"),
    ("layer6_router.py",       "Layer 6: HITL router + audit schema"),
    ("gamma_sensitivity.py",   "Gamma-sweep (H1)"),
    ("ablation.py",            "Layer ablation (T3)"),
    ("shap_vs_cefhf.py",       "Headline: SHAP vs CEFHF"),
    ("make_tables.py",         "Assemble final tables/figures"),
]


def run_step(script, desc, i, total):
    print(f"\n{'=' * 70}\n  STEP {i}/{total}: {desc}\n  Script: {script}\n{'=' * 70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed (exit {result.returncode}). Fix and re-run.")
        sys.exit(result.returncode)
    print(f"[OK] {script} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    print("=" * 70)
    print("  CEFHF SIX-LAYER ARCHITECTURE -- FULL PIPELINE")
    print("=" * 70)
    start = 1
    for k, a in enumerate(sys.argv):
        if a == "--start" and k + 1 < len(sys.argv):
            start = int(sys.argv[k + 1])
    timings = []
    for i, (script, desc) in enumerate(STEPS, 1):
        if i < start:
            print(f"  [SKIP] step {i}: {desc}")
            continue
        t0 = time.time()
        run_step(script, desc, i, len(STEPS))
        timings.append((i, desc, time.time() - t0))
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    for i, d, t in timings:
        print(f"  step {i}: {d:<42} {t:>6.1f}s")
    print(f"  total: {sum(t for _, _, t in timings):.1f}s")
    print(f"  outputs -> outputs/   data -> data/")
    print("=" * 70)
