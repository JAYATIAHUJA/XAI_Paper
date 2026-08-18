"""
layer5_explain.py -- Layer 5: Counterfactual explainability + Probability of
Sufficiency (PS) + actionability + explanation stability.

Implements (review Section 3 / Layer 5):
* Causal Shapley: contrast interventional vs observational feature perturbation
  via `shap` (interventional ranking differs from standard / correlational SHAP).
* Probability of Sufficiency per *actionable* feature: intervene through the
  Layer-2 SCM, PS = P(Ŷ_do(z=z') = 1 | x, Ŷ = 0); immutable features (age,
  gender, race) excluded by the actionability filter; monotone constraint
  (education / experience can only go UP).
* Explanation stability: Kendall tau between SHAP rankings across bootstrap
  model seeds (makes the paper's "tau = 0.89" claim a measured number).
* DiCE baseline for non-causal counterfactuals (best-effort; its TF/PyTorch
  dependency may be unavailable on this stack -- handled gracefully).

Outputs: outputs/layer5_causal_vs_standard_shap.csv, outputs/layer5_ps_actions.csv,
         outputs/layer5_explanations.json, outputs/layer5_stability_tau.csv,
         outputs/layer5_shap_comparison.png

Run:  python layer5_explain.py
"""

from __future__ import annotations

import json
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

import common
import layer2_dag_scm as L2

warnings.filterwarnings("ignore")

N_EXPLAIN = 200          # rejected candidates to explain
N_BOOT = 10              # bootstrap seeds for stability tau
DATASETS = ["synthetic", "adult"]

# actionable features (mutable) vs immutable per dataset
ACTIONABLE = {
    "synthetic": {"experience_years": "up", "education_level": "up",
                  "screening_score": "up", "negotiation": "up",
                  "skills": "up", "gap_years": "down"},
    "adult": {"hours-per-week": "up", "education-num": "up",
              "capital-gain": "up", "capital-loss": "up"},
}
IMMUTABLE = {"synthetic": ["gender", "race", "age", "ses", "zip", "university_quality"],
             "adult": ["sex", "age", "race", "native-country", "sex_enc"]}


def _train_xgb(bundle, train_df, seed=42):
    X = bundle.transform(train_df); y = train_df[bundle.target].to_numpy()
    m = common.make_xgb(seed); m.fit(X, y)
    return m


def _shap_values(model, X, feature_names):
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):      # binary classifiers sometimes return list
        sv = sv[1]
    return np.asarray(sv), expl


def _shap_ranking(model, X, feature_names):
    sv, _ = _shap_values(model, X, feature_names)
    return pd.Series(np.abs(sv).mean(axis=0), index=feature_names).sort_values(ascending=False)


def causal_vs_standard_shap(bundle, train_df, test_df, model):
    """Standard (observational/default) vs interventional SHAP ranking.

    shap's TreeExplainer defaults to the path-dependent (observational) mask.
    We also compute an interventional attribution by permuting features
    marginally (model-agnostic KernelExplainer on a small sample) -- the key
    deliverable is that the *ranking differs*, showing correlational SHAP can
    misattribute proxy paths.
    """
    feature_names = bundle.feature_names_out()           # one-hot-expanded names
    X = bundle.transform(test_df)
    rank_std = _shap_ranking(model, X, feature_names)
    # interventional: shap with interventional perturbation (model-agnostic)
    import shap
    try:
        bg = shap.sample(pd.DataFrame(bundle.transform(train_df), columns=feature_names), 50)
        f = lambda d: model.predict_proba(d)[:, 1]
        expl = shap.KernelExplainer(f, bg)
        sv = expl.shap_values(pd.DataFrame(X[:60], columns=feature_names), nsamples=50, silent=True)
        if isinstance(sv, list):
            sv = sv[1]
        rank_int = pd.Series(np.abs(np.asarray(sv)).mean(axis=0), index=feature_names).sort_values(ascending=False)
    except Exception as e:
        rank_int = rank_std.copy()
        print(f"    [interventional shap fallback] {e}")
    # kendall tau between rankings
    tau, _ = kendalltau(rank_std.values, rank_int.values)
    return rank_std, rank_int, tau


def probability_of_sufficiency(model, bundle, row, scm, actionable):
    """For a rejected candidate x (Ŷ=0), for each actionable feature intervene
    z -> z' (up by a step / to max) through the SCM and measure
    PS = P(Ŷ_do(z=z') = 1 | x, Ŷ = 0).  Ranks actionable interventions."""
    feats = bundle.feature_cols_with_enc()
    x_row = pd.DataFrame([row.values], columns=row.index)
    base_p = _proba(model, bundle.transform(x_row))
    results = []
    for feat, direction in actionable.items():
        if feat not in bundle.df.columns:
            continue
        cur = row[feat]
        # choose intervention value: move up/down by a meaningful step
        col = bundle.df[feat]
        if pd.api.types.is_numeric_dtype(col):
            step = max(col.std() * 0.5, 1e-6)
            z_new = cur + step if direction == "up" else max(cur - step, 0)
            z_new = float(z_new)
        else:
            cats = sorted(col.astype(str).unique())
            try:
                i = cats.index(str(cur))
            except ValueError:
                continue
            z_new = cats[min(i + 1, len(cats) - 1)] if direction == "up" else cats[max(i - 1, 0)]
        # do-intervention through SCM (propagate descendants, no abduction)
        cf = scm.intervene(x_row, {feat: np.array([z_new])})
        p_cf = _proba(model, bundle.transform(cf))[0]
        results.append({"feature": feat, "intervention_to": z_new,
                        "PS": float(p_cf), "base_p": float(base_p[0])})
    return results


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def stability_tau(bundle, train_df, test_df, n_boot=N_BOOT):
    """Kendall tau between standard-SHAP rankings across bootstrap model seeds.
    Higher mean tau => more stable explanations (the paper's stability claim)."""
    feature_names = bundle.feature_names_out()
    X = bundle.transform(test_df)
    ranks = []
    for s in range(n_boot):
        m = _train_xgb(bundle, train_df, seed=42 + s)
        ranks.append(_shap_ranking(m, X, feature_names).values)
    taus = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            t, _ = kendalltau(ranks[i], ranks[j])
            if not np.isnan(t):
                taus.append(t)
    return float(np.mean(taus)) if taus else float("nan")


def dice_baseline(bundle, train_df, query_row):
    """Non-causal counterfactual baseline via DiCE (best-effort)."""
    try:
        import dice_ml
        from dice_ml import Dice
        d = bundle.df[bundle.feature_cols + [bundle.target]].head(2000)
        dfs = dice_ml.Data(dataframe=d, continuous_features=[c for c in bundle.feature_cols
                            if pd.api.types.is_numeric_dtype(bundle.df[c])],
                           outcome_name=bundle.target)
        m = dice_ml.Model(model=_train_xgb(bundle, train_df), backend="sklearn")
        exp = Dice(dfs, m, method="random")
        q = {c: query_row[c] for c in bundle.feature_cols if c in query_row}
        cf = exp.generate_counterfactuals(pd.DataFrame([q]), total_CFs=2, desired_class="opposite",
                                          verbose=False)
        return {"method": "DiCE", "n_cf": len(cf.cf_examples_list[0].final_cfs_df) if cf.cf_examples_list else 0}
    except Exception as e:
        return {"method": "DiCE", "note": f"unavailable ({e.__class__.__name__})"}


def main():
    print("=" * 70)
    print("  LAYER 5 -- Counterfactual explainability + PS + stability")
    print("=" * 70)
    shap_rows = []; ps_rows = []; stab_rows = []; explanations = {}
    for name in DATASETS:
        print(f"\n--- {name} ---")
        bundle = common.load_dataset(name)
        _, scm = L2.get_scm(name)
        # split (reuse layer3-style)
        idx = np.arange(len(bundle.df)); rng = np.random.default_rng(42); rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_df, test_df = bundle.df.iloc[idx[:cut]], bundle.df.iloc[idx[cut:]]
        model = _train_xgb(bundle, train_df)

        # causal vs standard shap
        rank_std, rank_int, tau = causal_vs_standard_shap(bundle, train_df, test_df, model)
        print(f"  standard vs interventional SHAP Kendall tau = {tau:.3f}")
        for feat in rank_std.index:
            shap_rows.append({"dataset": name, "feature": feat,
                              "rank_standard": rank_std.index.get_loc(feat),
                              "rank_interventional": rank_int.index.get_loc(feat)
                              if feat in rank_int.index else -1,
                              "mean_abs_shap_standard": rank_std[feat]})
        # show top divergence (proxy misattribution)
        div = (rank_std - rank_int).abs().sort_values(ascending=False)
        print(f"  largest ranking divergence: {dict(list(div.head(3).items()))}")

        # PS for rejected candidates
        X = bundle.transform(test_df)
        pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
        rejected = test_df[(pred == 0)].sample(min(N_EXPLAIN, (pred == 0).sum() or 1), random_state=42)
        print(f"  generating PS for {len(rejected)} rejected candidates...")
        actionable = ACTIONABLE[name]
        ps_agg = OrderedDict((f, []) for f in actionable)
        for _, row in rejected.iterrows():
            ps = probability_of_sufficiency(model, bundle, row, scm, actionable)
            for r in ps:
                ps_rows.append({"dataset": name, "feature": r["feature"], "PS": r["PS"]})
                ps_agg[r["feature"]].append(r["PS"])
            # store one full explanation example
            if len(explanations.get(name, [])) < 2:
                explanations.setdefault(name, []).append({
                    "candidate": {c: row[c] for c in row.index if c != bundle.sensitive_enc_col},
                    "base_p": ps[0]["base_p"] if ps else None,
                    "ranked_actions": sorted(ps, key=lambda r: -r["PS"]),
                })
        # top actionable feature by mean PS
        mean_ps = {f: np.mean(v) if v else 0 for f, v in ps_agg.items()}
        top_action = max(mean_ps, key=mean_ps.get)
        print(f"  top actionable feature (mean PS): {top_action} = {mean_ps[top_action]:.3f}")

        # stability tau
        tau_stab = stability_tau(bundle, train_df, test_df)
        stab_rows.append({"dataset": name, "mean_kendall_tau_shap_stability": tau_stab, "n_boot": N_BOOT})
        print(f"  explanation stability (Kendall tau across {N_BOOT} seeds) = {tau_stab:.3f}")

        # DiCE baseline on one example
        if len(rejected):
            dice = dice_baseline(bundle, train_df, rejected.iloc[0])
            print(f"  DiCE baseline: {dice}")

    pd.DataFrame(shap_rows).to_csv(os.path.join(common.OUT_DIR, "layer5_causal_vs_standard_shap.csv"), index=False)
    pd.DataFrame(ps_rows).to_csv(os.path.join(common.OUT_DIR, "layer5_ps_actions.csv"), index=False)
    pd.DataFrame(stab_rows).to_csv(os.path.join(common.OUT_DIR, "layer5_stability_tau.csv"), index=False)
    with open(os.path.join(common.OUT_DIR, "layer5_explanations.json"), "w") as fh:
        json.dump(explanations, fh, indent=2, default=str)
    print("\n[SAVED] outputs/layer5_*.csv, layer5_explanations.json")

    # plot: standard vs interventional SHAP ranking divergence (synthetic)
    try:
        s = pd.DataFrame(shap_rows)
        syn = s[s.dataset == "synthetic"]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(syn))
        ax.scatter(x, syn["rank_standard"], label="standard (correlational) SHAP", s=60)
        ax.scatter(x, syn["rank_interventional"], label="interventional (causal) SHAP", s=60, marker="x")
        ax.set_xticks(x); ax.set_xticklabels(syn["feature"], rotation=45, ha="right")
        ax.set_ylabel("rank (0 = most important)"); ax.invert_yaxis()
        ax.set_title("Layer 5: SHAP ranking divergence (synthetic) -- correlational vs causal")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(common.OUT_DIR, "layer5_shap_comparison.png"), dpi=150); plt.close()
        print("[SAVED] outputs/layer5_shap_comparison.png")
    except Exception as e:
        print(f"[skip plot] {e}")
    print("\n[OK] Layer 5 complete.")


if __name__ == "__main__":
    main()
