# CEFHF ("From Glass Box to Fair Box") — Independent Review & Improvement Plan

*Reviewed: first draft (`CEFHF_Research_Paper.docx`), latest draft (`XAI.pdf`, 12 pp.), the change-plan doc, the `XAI_Paper` repository (scripts 00–13), the datasets, and every CSV/PNG in `outputs/`.*

---

## 0. One-paragraph verdict

The **idea is publishable** — a hiring-specific pipeline that ties causal fairness (SCM + counterfactuals) to explainability and governance is a genuine gap, and the writing in the latest draft is already reasonably tight. But right now the paper and the code are **two different projects**: the paper describes a six-layer causal framework with theorems and case-study numbers, while the repository is a *conventional* fairness audit (LR/RF/XGB + SHAP + SPD/DIR/EOD + a gender flip-test + AIF360 reweighing) that never builds an SCM, never generates a real counterfactual, never trains a fairness-constrained model, and — most seriously — runs its main experiment on a **recruitment dataset whose label is statistically independent of every feature** (AUC ≈ 0.50–0.52 for every model), so *no fairness number computed on it means anything*. A Springer reviewer will catch each of these. Below: what's wrong, what each layer actually is/should be, and a concrete, minimum-viable plan to close the gap fast.

---

## 1. Blocking problems (fix before submitting anywhere)

### P1. The recruitment dataset is noise
`data/archive/Dataset.csv` (Kaggle *recruitment-bias-and-fairness-ai-dataset*, 2,000 rows, 6 columns). I re-checked it:

| Evidence | Value |
|---|---|
| Correlation of `age`, `experience_years`, `screening_score` with `shortlisted` | 0.007, −0.027, −0.014 |
| Shortlist rate Male / Female | 0.385 / 0.421 |
| Shortlist rate by education (HS/Bach/MS/PhD) | 0.40 / 0.40 / 0.40 / 0.46 |
| 5-fold CV AUC, GradientBoosting, all features | **0.516** |
| Your own `robust_results_recruitment.csv` AUC (all 5 models) | 0.497 – 0.524 |

Consequences:
* Every metric on this dataset (DIR 0.81, EOD 0.11, CFVR-Prob 0.47, etc.) is a property of **model noise**, not of hiring bias. CFVR-Prob = 0.47 for tuned XGBoost simply means an over-fitted tree model wobbles when *any* input flips — exactly what `10_bias_vs_instability.py` was written to warn about.
* Reweighing "results" and the paired t-tests on this dataset (all p > 0.16) are therefore uninterpretable, not "non-significant".
* The Adult UCI results are the only real results you currently have — and Adult is income prediction, not hiring.

**Fix:** drop it as a primary benchmark (or keep it only as a *null-control* to show CFVR flags instability — one sentence). Replace with datasets that carry hiring signal (§4).

### P2. The first draft reports numbers that were never produced
Table 1 (ΔDP 0.21→0.04, AUC 0.82→0.78), PS 0.31→0.07, KL 0.08 vs 0.31 vs 0.44, "89 % actionable", "24 HR professionals, 5.8/7", "340 ms / 4.2× overhead", "Γ ≤ 1.4", "COMPAS-Employment synthetic dataset (25,000)", "ResumeNet 8,500 resumes", "1,200 LLaMA-3 prompts" — none of this exists in the repo. The latest draft correctly removed almost all of it, but still contains **"Kendall τ = 0.89 vs 0.62"** (§7.2) and the conclusion says the case study lets the authors **"prove"** the glass→fair box shift. Remove or produce. Fabricated results are a desk-reject / retraction risk, not a style issue.

### P3. "CFVR" as implemented is not counterfactual fairness
`06_cfvr.py` / `12_full_robust_pipeline.py`: flip the (standardised) gender column, re-predict, count |Δp| > 0.10 (CFVR-Prob) or label flips (CFVR-Class).
* This is a **Level-2 intervention on A alone** (a "flip test"/direct-effect sensitivity), not Kusner-style counterfactual fairness, which requires an SCM to propagate A→descendants (experience, screening score…) via abduction–action–prediction. The paper defines CFVR with `PS(A→A'; x, M)` under an SCM `M` — there is no `M` in the code.
* It only works because gender is *kept as a feature*; the paper's Layer 1 says proxies/sensitive signals are removed. If A is not in X the flip test is identically 0. The two are inconsistent.
* Threshold artefact: in `final_results_table.csv` Logistic Regression has SPD = 0.935 (predictions almost fully gender-determined) but CFVR = 0.0 — because probabilities hover near 0.5 so |Δp| < 0.10 while the *class* flips. CFVR-Prob with a fixed τ can miss the most biased model in the table. Report CFVR-Class as primary and show a τ-sweep.

**Fix:** implement CFVR properly on top of a fitted SCM (§3, Layer 2/3), and keep the flip test as an explicit "direct-effect / unawareness violation" baseline. Also cite the closest prior metric so it isn't over-claimed as novel (e.g., counterfactual-flip rates in Kusner et al. 2017; "counterfactual token fairness" in Garg et al. 2019; individual fairness consistency in Zemel et al. 2013).

### P4. References — several are misattributed or apparently non-existent
I spot-checked the ones the SOTA table (Table 1) rests on:

| Ref (latest draft) | Finding |
|---|---|
| [17] "Oprescu, Syrgkanis, Risteski (2024) Causal fairness under unobserved confounding: a neural sensitivity framework, ICLR 2024" | Real paper, **wrong authors**: Schröder, Frauen & Feuerriegel (LMU) — [openreview](https://openreview.net/forum?id=DqD59dQP37), [arXiv 2311.18460](https://arxiv.org/abs/2311.18460). "GMSM" comes from Frauen et al., *Sharp bounds for generalized causal sensitivity analysis*, NeurIPS 2023. |
| [19] "CausalFair-LLM, arXiv:2503.14201" | Could not find any such paper/ID. |
| [20] "FairDiffusion, Sauer & Geiger, ICLR 2025" | The real FairDiffusion is Luo et al., *Science Advances* 2025 (medical imaging), not a tabular fair-synthetic-data method — [arXiv 2412.20374](https://arxiv.org/abs/2412.20374). |
| [21] "InFair, Roy & Zhang, KDD 2024" | Could not find. |
| [18] "FACTS, Gupta et al., WWW 2024" | Could not find. FACE (Poyiadzi et al., AIES 2020) and FACTS (Kavouras et al., NeurIPS 2023 — *Fairness Aware Counterfactuals for Subgroups*) exist. |
| [30] "Yang et al. 2024, arXiv 2410.00903" | That ID is Imai & Nakamura, *Causal representation learning with generative AI: texts as treatments*. |
| [16] LD3 | Real — Maasch et al., **AAAI 2025** ([ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/34130)); update venue. |
| [12] Bjøru et al. arXiv 2512.02735 | Real ([arXiv](https://arxiv.org/abs/2512.02735)). |
| First draft [27] "Jin et al." for the same sensitivity paper | Also wrong. |

Because Table 1 (comparative positioning) is built on [18]–[22], **rebuild it with verified methods** (see §4.4). Every citation should be checked against DOI/arXiv before submission — Springer LNCS reviewers routinely do this.

### P5. Propositions 1 & 2 are not theorems as stated
* **Prop 1** `|CF(M*) − CF(M̂)| ≤ 2Γσ_Aσ_Y/(1+Γ²σ_A²)`: CF is a rate in [0,1]; Γ in MSM-style models is an odds-ratio bound (dimensionless); σ_A, σ_Y are standard deviations of a binary A and binary Y. The expression is dimensionally arbitrary and no derivation exists in the PDF ("Appendix A" is absent). A genuine version would come from the GMSM bounds of Frauen et al./Schröder et al.: under confounding strength Γ the counterfactual probability lies in an interval [CF⁻(Γ), CF⁺(Γ)]; you can *report* the width, but you should not invent a closed form.
* **Prop 2** `ΔAcc/ΔFair ≤ λ·|∂Penalty/∂h|`: ∂/∂h of a functional is undefined without a parametrisation; at an optimum ∂L/∂h = −λ∂Penalty/∂h so the "bound" is circular; "there exists an optimal λ*" and "Pareto-optimal" don't follow. The correct, well-known statement is: for convex L and Penalty, the minimisers of L + λ·Penalty for λ ≥ 0 are Pareto-optimal (weighted-sum scalarisation). State that (with a citation), and then **measure** the frontier via a λ-sweep.

**Fix:** either (a) drop "theoretical guarantees" from Objective 3 and present two *empirically tested hypotheses* (H1: CFVR gap shrinks with tighter Γ; H2: fairness–accuracy Pareto frontier is traced by λ), or (b) write real proofs for weaker, defensible claims. (a) is faster and safer.

### P6. Internal inconsistencies between drafts and within the latest draft
* Layer 1 = "adversarial debiasing" (draft 1) vs "LFR" (draft 2). Layer 3 = "counterfactual fairness evaluation" (draft 1) vs "four-metric stack" (draft 2). Pick one and align Fig. 2, the layer text, and Algorithm 1.
* Layer 4 says the objective is `L + λ·(DP+EO+CF penalty)` and, two sentences later, that FairXGBoost "assigns weights to each feature according to SCM importance scores". Those are two different mechanisms; the code implements neither.
* Layer 1 uses DECAF, which *needs the DAG produced by Layer 2*. Either reorder (causal discovery first) or make Layer 1's output a proxy report and move DECAF into Layer 2/4. This ordering bug will be noticed.
* Layer 6 trigger "when the GMSM sensitivity falls *within* its bounds" is inverted — escalation should trigger when the fairness interval *crosses* the threshold.
* The abstract promises "a case study … to demonstrate the role and impact of each of the six layers"; §6.8 is a 2-paragraph narrative with no data, model, prompts or numbers.
* Draft 2's Figure 1 ("Difference between LIME & SHAP") is not referenced in text; Figures 2–3 are placeholders in the PDF.
* §7 (UHV / Fayol's principles) reads like a course-requirement section; reviewers at ML venues will see it as padding. Fold one paragraph into Ethics/Governance or cut.
* Repo README says "Novel metric – CFVR"; keep terminology identical to the paper (Counterfactual *Fairness* Violation Rate).

---

## 2. What the repository actually does (vs. what the paper says)

| Paper component | Repo status | File |
|---|---|---|
| EDA / selection-rate check | ✅ | `01_eda_bias_check.py` |
| Preprocessing (scale/one-hot/split) | ✅ | `02_preprocess.py`, `12_…prepare_data` |
| LR / RF / XGB baselines | ✅ | `03_`, `12_` |
| SHAP (TreeExplainer, global + gender impact) | ✅ (correlational SHAP only) | `04_shap_analysis.py` |
| SPD / DIR / EOD (fairlearn) | ✅ | `05_`, `12_` |
| CFVR-Prob / CFVR-Class (flip test) | ✅ but not SCM-based | `06_`, `12_` |
| Reweighing (Kamiran–Calders) | ✅ | `07_`, `12_` |
| 5-fold CV + RandomizedSearch tuning | ✅ | `12_` |
| Adult UCI benchmark | ✅ | `09_`, `12_` |
| Bias vs instability check | ✅ | `10_` |
| Paired t-tests | ✅ (but see note) | `13_` |
| **L1** proxy detection / LFR / DECAF | ❌ | — |
| **L2** LD3 / SCM / DAG / Γ | ❌ | — |
| **L3** Differential Fairness ε, SCM-CFVR | ❌ | — |
| **L4** FairXGBoost / λ objective | ❌ | — |
| **L5** causal SHAP, PS scores, actionability filter | ❌ (LIME in requirements, unused) | — |
| **L6** HITL routing / audit log | ❌ | — |
| Ablation, λ-sweep, Γ-sweep, LLM matched pairs | ❌ | — |
| `final_results_table.csv` "CEFHF (Proposed)" | row is `--` | `08_` |

Smaller code issues worth fixing: paired t-test on 5 CV folds violates independence (use the Nadeau–Bengio corrected resampled t-test, or 5×2 CV, or bootstrap CIs and say "descriptive"); `final_results_table.csv` and `robust_results_*.csv` disagree (different runs — keep one source of truth); no seed/version manifest; `run_all.py` doesn't run 09–13.

---

## 3. The six-layer architecture — what it is, what it must output, how to improve it

Reading both drafts, here is the cleanest consistent definition, followed by what a reviewer will want to see and the concrete implementation I recommend.

### Layer 1 — Bias-aware preprocessing → `D_fair`
**Does:** flags proxies of protected attributes, learns a representation with reduced dependence on A, optionally augments sparse intersectional cells with causally-fair synthetic rows.
**Outputs:** proxy report (per feature: dependence on A), transformed data `D_fair`, mutual-information / adversarial-AUC scores before/after.

Improvements:
* Replace vague "latent-space clustering" with a **measurable proxy score**: for each feature x_j, AUC of a small classifier predicting A from x_j (and from x_j | legitimate qualifications Q). Report a table; flag if AUC > 0.6 (or MI > ε_proxy). This is trivial to compute and gives you a real Layer-1 output.
* LFR is in AIF360 (`aif360.algorithms.preprocessing.LFR`); reweighing you already have. Add both as Layer-1 variants; report their effect on the proxy scores.
* DECAF needs the DAG → **move DECAF after Layer 2** (or run L2 discovery first and iterate). Also note DECAF requires the original TensorFlow repo; if it's too heavy, a simple SCM-based generator that removes A→X edges is an acceptable, transparent substitute ("DECAF-style").

### Layer 2 — Causal inference engine → SCM `M`, DAG `G`, sensitivity Γ
**Does:** builds a DAG (expert priors + local discovery), fits structural equations, tags edges as legitimate / proxy / discriminatory, attaches a confounding-strength parameter for unobserved U.
**Outputs:** validated `G` (figure), fitted `M` (equations/coefficients), edge classification table, Γ setting.

Improvements:
* LD3 gives you the **parents of Y and a valid adjustment set** — not a full SCM. To generate counterfactuals you need equations for the descendants of A. Practical recipe: (1) expert DAG for a small hiring vocabulary (A={gender, age}; mediators = experience, education, test score, gap years; Y = shortlist); (2) verify edges with conditional-independence tests (`pgmpy`, `causal-learn`); (3) fit structural equations — linear/logistic-Gaussian, or a small normalising-flow SCM — with `DoWhy` (has a `gcm` module that fits an SCM and does abduction/counterfactuals directly); (4) counterfactual for candidate x: abduct U, set A←a', predict.
* Γ: implement a **marginal-sensitivity-model bound** on the counterfactual probability (Schröder et al. 2024 have code) or, cheaper, a Rosenbaum-style Γ-sweep: for Γ ∈ {1, 1.2, 1.5, 2} report [CFVR⁻, CFVR⁺]. Plot it. That is your "Γ sensitivity study".
* **Ground truth**: add a synthetic hiring SCM where you *know* direct, proxy and confounded paths (see §4). Without it you cannot show that Layer 2 recovers the right structure.

### Layer 3 — Fairness evaluation → metric stack
**Does:** DIR/SPD, EOD, CFVR (SCM-based), Differential Fairness ε across intersectional groups (gender × age band), with CIs.
**Outputs:** one table per model per dataset, ± CI over folds/bootstraps; τ-sweep for CFVR; ε per subgroup.

Improvements:
* Implement DF ε (Foulds et al.) — a few lines: max over subgroup pairs of |log P(Ŷ=1|g) − log P(Ŷ=1|g')|.
* Report **direct vs. path-specific effect** using LD3's WCDE — this is what makes Layer 3 "causal" and differentiates it from fairlearn.
* Two CFVR flavours, named honestly: *CFVR-flip* (A only) and *CFVR-SCM* (A + descendants).

### Layer 4 — Fairness-constrained prediction → `h*`
**Does:** minimises `L(Y,Ŷ) + λ·FairnessPenalty(h, G, M)`.
**Outputs:** trained model per λ; λ-sweep table/plot (accuracy vs. CFVR/SPD Pareto curve); chosen λ*.

Improvements — three concrete, implementable options (pick 1–2, name them precisely, drop the undefined "FairXGBoost"):
1. **Counterfactual-augmentation training** (simplest, and it directly targets CFVR): for every training row generate its SCM counterfactual twin, keep the same label, train XGBoost on D ∪ D_cf, optionally with a consistency penalty weight λ. This is the CEFHF-specific choice because it *uses Layer 2's SCM*.
2. **Fairlearn reductions** (`ExponentiatedGradient` / `GridSearch`) with an XGBoost base learner and DP or EO constraints — a standard, citable baseline.
3. **FairGBM** (Cruz et al., ICLR 2023 — real, pip-installable) — gradient boosting with in-training fairness constraints; use it as the "statistical fair" comparator.
Run λ ∈ {0, 0.1, 0.25, 0.5, 1, 2, 5}; report the Pareto plot. That plot replaces Proposition 2.

### Layer 5 — Counterfactual explainability → per-candidate explanation + PS
**Does:** causal attributions, probability-of-sufficiency for actionable interventions, actionability filter.
**Outputs:** for each rejected candidate: top-k causal attributions, ranked actionable interventions with PS, a rendered sentence; global PS by subgroup; explanation-stability score.

Improvements:
* "Restrict SHAP to causal parents" is *not* causal SHAP. Use a real method and cite it: **Asymmetric Shapley values** (Frye et al., NeurIPS 2020) or **Causal Shapley values** (Heskes et al., NeurIPS 2020); `shap` also lets you contrast interventional vs. observational (`feature_perturbation="interventional"`) — show that ranking differs from standard SHAP.
* PS per actionable feature: intervene on experience/education/test score through the SCM, `PS = P(Ŷ_do(z=z') = 1 | x, Ŷ = 0)`; immutable features (age, gender) excluded by the actionability filter; monotone constraints (education can only go up). DiCE is a fine baseline for "non-causal counterfactuals".
* Explanation stability (the "Kendall τ" claim): compute it — τ between SHAP rankings across bootstrap models / LIME seeds — then you can legitimately report it.

### Layer 6 — Human-in-the-loop governance → decision + audit record
**Does:** routes cases to a human when PS > τ_PS, CFVR-SCM interval crosses threshold, or DF ε breached; logs everything.
**Outputs:** escalation rate vs. threshold curve, audit-log JSON schema, example record.

Improvements:
* You can't run a real HITL study quickly, but you can **simulate the router**: % of candidates escalated at each τ_PS, precision of escalation on the synthetic set (does it catch the truly discriminated cases?), and reviewer workload. That is a real result.
* Provide the audit-log schema (fields: input hash, Z, PS per A, CFVR interval, explanation, reviewer id/decision/reason, timestamps). Map to IEEE 7003-2024 clauses *cautiously* ("designed to support", not "satisfies").
* If time permits, a small user study (n ≥ 15 HR/students, 3 explanation formats, comprehension + trust Likert) turns Layer 5/6 into a contribution instead of a claim.

---

## 4. Data & experiments that will make the paper credible

### 4.1 Datasets (recommended set)
1. **Synthetic hiring SCM (mandatory).** You control the DAG: `Gender → {gap_years, negotiation}`, `SES → University → Skills → Score → Y`, `Race → ZIP → University` (proxy), unobserved U with strength Γ. Generate 20k rows. This is the *only* way to show Layer 2 recovers structure, CFVR-SCM ≠ CFVR-flip, and Γ bounds are honest.
2. **Adult UCI** — keep (standard, non-hiring; say so).
3. **A real hiring-adjacent dataset with signal**: options — *FairJob* (Criteo, 2024; large real job-ad clicks with gender proxy), *Bias in Bios* (De-Arteaga et al. 2019; occupation from biography text, gender bias — good for the text/LLM angle), or the *Law School Admissions* dataset (used by Kusner et al., lets you compare directly with counterfactual-fairness baselines). Pick one you can obtain quickly.
4. **LLM matched-pair audit** — 500–1,000 resume pairs (identical content; names/pronouns/graduation year varied), an open model (e.g., Llama-3-8B-Instruct or Qwen), fixed prompt & temperature 0, ≥3 seeds; report shortlist rate gap and PS with bootstrap CIs, before/after Layer-1 name redaction. Cheap, and it turns §6.8 from a story into evidence.

### 4.2 Minimum table/figure set (what the change-plan doc already suggests, made concrete)
* T1 Dataset summary (n, features, protected attrs, base rates).
* T2 Main results: Unconstrained XGB · Reweighing · Fairlearn-EG · FairGBM · CF-augmentation (CEFHF-L4) · full CEFHF — Acc/F1/AUC · SPD/DIR/EOD · CFVR-flip/CFVR-SCM · DF ε, mean ± CI, per dataset.
* T3 Layer ablation (drop L1, drop L2/SCM → CFVR-flip only, drop L4, drop L5 filter).
* F1 λ-sweep Pareto (AUC vs CFVR-SCM).
* F2 Γ-sensitivity intervals.
* T4 LLM matched-pair before/after.
* T5 Explanation quality: stability τ, % actionable, PS coverage.
* T6 Runtime per layer (ms/candidate).
* Statistical testing: corrected resampled t-test or bootstrap CIs; say "5-fold" honestly.

### 4.3 Suggested new scripts (keep your numbering convention)
`14_synthetic_scm.py` · `15_layer1_proxy_lfr.py` · `16_layer2_dag_scm.py` (DoWhy gcm + CI tests + LD3-style parent search) · `17_layer3_metrics.py` (DF ε, CFVR-SCM, WCDE) · `18_layer4_fair_training.py` (CF-augmentation, fairlearn EG, FairGBM, λ-sweep) · `19_layer5_explain.py` (asymmetric/causal Shapley, PS, actionability, τ-stability) · `20_layer6_router.py` · `21_gamma_sensitivity.py` · `22_llm_matched_pairs.py` · `23_ablation.py` · `24_make_tables.py`. Add `configs/seeds.yaml`, `environment.yml`, and a `Makefile`/`run_all.py` that runs everything end-to-end.

### 4.4 Rebuild the SOTA comparison table with verifiable rows
Kusner et al. 2017 (counterfactual fairness) · Nabi & Shpitser 2018 (path-specific) · Zhang & Bareinboim 2018 (fairness in decision-making — causal explanation formula) · Chiappa 2019 (path-specific CF fairness) · van Breugel et al. 2021 DECAF · Maasch et al. 2025 LD3 · Schröder et al. 2024 (sensitivity) · Cruz et al. 2023 FairGBM · Agarwal et al. 2018 (fairlearn reductions) · Karimi et al. 2021 (algorithmic recourse: causal) · Frye 2020 / Heskes 2020 (causal Shapley) · Kavouras 2023 FACTS · Bjøru 2025 (PS explanations) · Foulds 2023 (DF). Columns: causal SCM · CF metric · fairness-constrained training · causal explanation · intersectional · sensitivity · HITL · hiring evaluation.

---

## 5. Paper-level rewrite guidance (Springer LNCS style)

1. **Reframe contributions honestly**: (i) formal problem + CFVR (flip vs SCM), (ii) six-layer architecture with a *reference implementation*, (iii) empirical study on synthetic + Adult + one hiring dataset + LLM audit, (iv) ablation/λ/Γ analyses. Drop "theoretical guarantees" unless proved.
2. Structure: Intro (1 p) · Related work + positioning table (1.5 p) · Problem & metrics (1 p) · CEFHF (2.5 p incl. Fig 2 architecture + Fig 3 DAG + Algorithm 1) · Experimental setup (1 p) · Results (2.5 p) · Discussion/limitations/ethics (1 p) · Conclusion. LNCS 12–16 pages.
3. Kill or shrink: §7 UHV/Fayol; repeated "glass box vs fair box" paragraphs; over-strong legal claims ("satisfies EU AI Act", "verifiably equitable") → "designed to support".
4. Fix Layer descriptions so text, Fig 2, Algorithm 1 and code agree (see P6).
5. Add a Reproducibility statement (repo URL, commit hash, seeds, versions, hardware).
6. Limitations should now say what is *still* missing (real HITL study, single-country legal frame, DAG misspecification) rather than "the framework lacks experimental validation".
7. Proof-read: "Theorectical", inconsistent hyphenation, "Level 3 of Pearl's framework", figure references, reference formatting (LNCS style, no bare URLs for [3]–[5]).

---

## 6. Venue notes (Springer)
Good fits for a causal-XAI-fairness systems paper with an empirical study: **World Conference on eXplainable AI (xAI)** (Springer CCIS proceedings — most on-topic), **CD-MAKE** (Springer LNCS, XAI track), **PAKDD** / **ECML-PKDD** (LNCS/LNAI, competitive), **ICCS**, **IDEAL**; journals: *AI and Ethics* (Springer), *Machine Learning*, *Neural Computing & Applications*. Check current CFP deadlines — I have not verified 2026/27 dates. Do **not** submit before P1–P5 are fixed; a desk reject costs more time than the fixes.

---

## 7. Prioritised roadmap (≈3–4 weeks of focused work)

| Week | Deliverable |
|---|---|
| 1 | Synthetic SCM generator + real hiring dataset chosen; Layer 2 (DAG + DoWhy SCM + counterfactual generation); CFVR-SCM; proxy scores (L1) |
| 2 | Layer 4 (CF-augmentation, fairlearn EG, FairGBM) + λ-sweep; Layer 3 DF ε; Γ-sweep; ablation harness |
| 3 | Layer 5 (causal Shapley + PS + actionability + τ-stability); Layer 6 router simulation + audit schema; LLM matched-pair audit |
| 4 | Tables/figures, corrected statistics, rewrite paper (fix P2–P6, references, LNCS format), reproducibility package, internal review |

If time is very short, the *minimum* credible version is: synthetic SCM + Adult + LLM audit, Layers 2/3/4/5 implemented, λ & Γ plots, ablation, propositions downgraded to hypotheses, references cleaned. That is a defensible Springer conference paper; the current draft is not yet.
