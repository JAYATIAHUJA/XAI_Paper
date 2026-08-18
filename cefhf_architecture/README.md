# CEFHF — Causal Explainability Fair Hiring Framework (reference implementation)

Reference implementation of the **six-layer CEFHF architecture** from
*“From Glass Box to Fair Box: A Causal Explainability Framework for Ethical AI
Hiring Systems.”* This closes the gap the independent review flagged: the
`XAI_Paper` repository was a *conventional* fairness audit (SHAP + SPD/DIR/EOD +
a gender flip-test); this folder implements the real causal architecture the
paper describes — SCMs, counterfactual fairness, fairness-constrained training,
causal explanations, and a HITL router — and demonstrates it beats
correlational SHAP/LIME.

## Datasets (and *why*)
There is no public *hiring* dataset that exposes the true causal structure, so
no fairness metric on one can be validated. We therefore use:
1. **Synthetic hiring SCM** (`synthetic_scm.py`, generated locally, 20k rows) —
   the hiring dataset with a **known DAG**: `gender→{gap_years,negotiation,
   screening_score}` (discriminatory), `race→zip→university_quality` (proxy),
   `SES→university→skills→score→Y` (legitimate), and an unobserved `U→{gender,Y}`
   confounder whose strength Γ the sensitivity sweep probes. **This is the only
   dataset on which “CEFHF > SHAP/LIME” can be shown rigorously.**
2. **Adult UCI** (real, in `XAI_Paper/data/adult`) — standard real benchmark.
3. **Recruitment dataset** (in `XAI_Paper/data/archive`) — **null-control only**
   (AUC≈0.46, label independent of features); shows CFVR flags instability on noise.

## The six layers
| Layer | Script | Does |
|---|---|---|
| Ground truth | `synthetic_scm.py` | generates the synthetic hiring SCM + ground-truth DAG |
| L1 | `layer1_proxy_lfr.py` | proxy AUC/MI per feature → D_fair (proxy-removal, reweighing, residualisation) |
| L2 | `layer2_dag_scm.py` | expert DAG, pgmpy CI tests, additive-noise SCM (abduction–action–prediction), counterfactuals, Γ |
| L3 | `layer3_metrics.py` | CFVR-SCM vs CFVR-flip, DF ε, WCDE, τ-sweep, bootstrap CIs |
| L4 | `layer4_fair_training.py` | CF-augmentation (SCM twins, λ-sweep), fairlearn-EG, Pareto (H2) |
| L5 | `layer5_explain.py` | interventional vs standard SHAP, Probability-of-Sufficiency, actionability, stability τ |
| L6 | `layer6_router.py` | HITL router simulation (escalation vs τ, precision), audit-log schema |
| Analyses | `gamma_sensitivity.py` `ablation.py` `shap_vs_cefhf.py` `make_tables.py` | H1, ablation, the headline SHAP-vs-CEFHF, final tables |

## Run
```bash
python run_all.py            # everything, end-to-end
python run_all.py --start 4  # resume from layer 4
```
Outputs land in `outputs/` (CSVs + figures) and `data/` (models, SCM, synthetic data).

## Headline results (see `outputs/consolidated_results.md`)
- **CFVR-SCM catches bias the flip-test misses** (synthetic LogReg: CFVR-flip=0.000
  but CFVR-SCM=0.382) — the paper’s central claim (review P3).
- **Reweighing fixes SPD but not CFVR-SCM** (XGBoost+RW: SPD 0.073→0.039,
  CFVR-SCM 0.295→0.293) — statistical debiasing ≠ counterfactual fairness.
- **CF-augmentation λ-sweep** cuts CFVR-SCM 0.295→0.101 at zero accuracy cost
  (synthetic) → the Pareto frontier, H2 (replaces Proposition 2).
- **Γ-sweep** widens the CFVR interval (H1, replaces Proposition 1).
- **SHAP vs CEFHF**: SHAP’s top features are the downstream mediators
  (`screening_score, skills`); CEFHF flags the upstream discriminatory
  (`gender→score`) and proxy (`race→zip→university`) paths — the root causes.

## Stack notes (honest limitations)
- **DoWhy `gcm` is incompatible** with numpy 2 / pandas 3 / py3.14 (it imports
  `numpy.dual`, `np.row_stack`, both removed). Layer 2 therefore uses a
  transparent **additive-noise SCM** (abduction–action–prediction) — the same
  algorithm DoWhy `gcm` runs for additive-noise models — and the review’s
  “linear/logistic-Gaussian SCM” recipe.
- **AIF360 `LFR` is broken** on scipy≥1.12 (`fmin_l_bfgs_b(disp=…)`); Layer 1
  uses transparent **residualisation** as the learned-representation variant
  alongside proxy-removal and reweighing.
- **FairGBM** and **DiCE** are unavailable on this stack (no py3.14 wheels);
  both are attempted and documented. FairGBM is replaced by fairlearn-EG.
- The LLM matched-pair audit (§6.8) was **skipped** per the user’s choice; the
  HITL router is simulated instead. Propositions 1 & 2 are downgraded to the
  empirically-tested hypotheses H1 (Γ) and H2 (λ).
```
python -m venv .venv && .venv/bin/pip install -r environment.yml  # see environment.yml
```
