"""
=============================================================
RUN_ALL.py -- Master Pipeline Runner
=============================================================
Runs all 9 steps in sequence.

Usage:
    python run_all.py

Or run individual steps:
    python 00_download_dataset.py
    python 01_eda_bias_check.py
    python 02_preprocess.py
    python 03_train_baseline.py
    python 04_shap_analysis.py
    python 05_fairness_metrics.py
    python 06_cfvr.py
    python 07_debiasing_reweighing.py
    python 08_final_table.py
    python 09_adult_uci_validation.py
=============================================================
"""

import subprocess
import sys
import os
import time

SCRIPTS = [
    ("00_download_dataset.py",    "Download Dataset"),
    ("01_eda_bias_check.py",      "EDA + Bias Check"),
    ("02_preprocess.py",          "Preprocessing"),
    ("03_train_baseline.py",      "Train Baseline Models"),
    ("04_shap_analysis.py",       "SHAP Analysis"),
    ("05_fairness_metrics.py",    "Fairness Metrics (SPD/DIR/EOD)"),
    ("06_cfvr.py",                "CFVR -- Novel Metric"),
    ("07_debiasing_reweighing.py","Debiasing (Reweighing)"),
    ("08_final_table.py",         "Final Results Table"),
    ("09_adult_uci_validation.py","Adult UCI Generalization"),
]

def run_step(script, description, step_num, total):
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}/{total}: {description}")
    print(f"  Script: {script}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed! (exit code {result.returncode})")
        print(f"  Fix the error in {script} and re-run from that step.")
        sys.exit(result.returncode)
    print(f"\n[OK] Step {step_num} completed in {elapsed:.1f}s")
    return elapsed


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RECRUITMENT BIAS & FAIRNESS AI -- FULL PIPELINE")
    print("=" * 70)

    # Allow skipping steps via arg: python3 run_all.py --start 3
    start_from = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--start" and i + 1 < len(sys.argv):
            start_from = int(sys.argv[i + 1])

    if start_from > 1:
        print(f"\n[INFO] Starting from step {start_from}")

    timings = []
    for i, (script, desc) in enumerate(SCRIPTS, 1):
        if i < start_from:
            print(f"  [SKIP] Step {i}: {desc}")
            continue
        elapsed = run_step(script, desc, i, len(SCRIPTS))
        timings.append((i, desc, elapsed))

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE -- SUMMARY")
    print("=" * 70)
    total_time = sum(t for _, _, t in timings)
    for step, desc, t in timings:
        print(f"  Step {step}: {desc:<40} {t:>6.1f}s")
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"\n  All outputs saved to: outputs/")
    print(f"  All models saved to:  data/")
    print("=" * 70 + "\n")
