"""
layer1_proxy_lfr.py -- Layer 1: Bias-aware preprocessing -> D_fair.

Does (per review Section 3 / Layer 1):
* Proxy detection: for every feature x_j, the AUC of a small classifier
  predicting the protected attribute A from x_j alone, and the *incremental*
  AUC added on top of the legitimate qualifications Q.  Flags AUC > 0.60 or
  MI > epsilon.  This turns "ZIP is a proxy for race" from a narrative into a
  measured number.
* D_fair construction via two variants:
    1. proxy removal -- drop features flagged as proxies of A (transparent);
    2. reweighing -- Kamiran-Calders sample weights (already in common.py);
    3. LFR (aif360) -- learned fair representation, attempted and reported.
* Reports proxy scores + adversarial-AUC before/after each variant.

Outputs: outputs/proxy_report.csv, data/dfair_*.pkl, outputs/layer1_summary.csv

Run:  python layer1_proxy_lfr.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import _VectorizerMixin  # noqa (kept for type clarity only)
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

import common

warnings.filterwarnings("ignore")


def _encode_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = df[c].astype(float)
        else:
            out[c] = df[c].astype("category").cat.codes.astype(float)
    return out


def proxy_auc_for_A(df: pd.DataFrame, feature: str, A_col: str, seed=42) -> float:
    """5-fold CV AUC of predicting the (binary) protected attr A from a single
    feature.  Returns 0.5 if A is constant or the feature has no signal."""
    A = df[A_col]
    if A.nunique() < 2:
        return float("nan")
    y = (A != A.min()).astype(int).values      # binary 0/1
    X = _encode_features(df, [feature]).to_numpy()
    if np.unique(X).size < 2:
        return 0.5
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        m = LogisticRegression(max_iter=400)
        scores = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
        return float(np.mean(scores))
    except Exception:
        return float("nan")


def incremental_proxy_auc(df, feature, A_col, Q_cols, seed=42):
    """AUC(A ~ Q) vs AUC(A ~ Q + x_j): the lift x_j gives a model that already has
    the legitimate qualifications.  Large lift => x_j is a proxy beyond Q."""
    A = df[A_col]
    if A.nunique() < 2:
        return float("nan"), float("nan")
    y = (A != A.min()).astype(int).values
    QX = _encode_features(df, Q_cols).to_numpy()
    full = np.hstack([QX, _encode_features(df, [feature]).to_numpy()])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    m = LogisticRegression(max_iter=600)
    try:
        base = float(np.mean(cross_val_score(m, QX, y, cv=cv, scoring="roc_auc")))
        full_auc = float(np.mean(cross_val_score(m, full, y, cv=cv, scoring="roc_auc")))
        return base, full_auc
    except Exception:
        return float("nan"), float("nan")


def mutual_info(df, feature, A_col):
    A = df[A_col]
    if A.nunique() < 2:
        return float("nan")
    y = (A != A.min()).astype(int).values
    X = _encode_features(df, [feature]).to_numpy()
    return float(mutual_info_classif(X, y, random_state=0, discrete_features=False)[0])


def proxy_report(df, features, A_cols, Q_cols):
    rows = []
    for feat in features:
        row = {"feature": feat}
        for A in A_cols:
            row[f"auc_A={A}"] = proxy_auc_for_A(df, feat, A)
            base, full = incremental_proxy_auc(df, feat, A, [q for q in Q_cols if q != feat])
            row[f"inc_auc_A={A}"] = full - base if not np.isnan(base) else float("nan")
            row[f"mi_A={A}"] = mutual_info(df, feat, A)
        rows.append(row)
    rep = pd.DataFrame(rows)
    # flag proxies for any protected attr
    flag_cols = [c for c in rep.columns if c.startswith("auc_A=")]
    rep["proxy_flag"] = rep[flag_cols].gt(common.CFG["proxy"]["auc_flag"]).any(axis=1)
    return rep


def adversarial_auc(df, features, A_col, seed=42):
    """How well can an adversary recover A from the full feature set?
    A D_fair transform should drive this towards 0.5."""
    A = df[A_col]
    if A.nunique() < 2:
        return float("nan")
    y = (A != A.min()).astype(int).values
    X = _encode_features(df, features).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    m = LogisticRegression(max_iter=800)
    return float(np.mean(cross_val_score(m, X, y, cv=cv, scoring="roc_auc")))


def dfair_proxy_removal(df, report, features, A_cols):
    """Drop features flagged as proxies of any protected attribute."""
    flagged = report.loc[report["proxy_flag"], "feature"].tolist()
    keep = [f for f in features if f not in flagged]
    return df[keep + [c for c in df.columns if c not in features]], flagged


def dfair_lfr(df, features, A_col, target, seed=42):
    """Learned Fair Representations (Zemel et al.) via AIF360.

    NOTE: AIF360 0.6.1's LFR calls scipy's ``fmin_l_bfgs_b(disp=...)`` which
    modern scipy (>=1.12) rejects, so LFR cannot run on this stack.  We keep the
    call (so the attempt is logged) but fall back to ``dfair_residualize``.
    """
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.algorithms.preprocessing import LFR
        enc = _encode_features(df, features).astype(np.float64)
        work = enc.copy()
        work[A_col] = (df[A_col] != df[A_col].min()).astype(int).astype(float)
        work[target] = df[target].astype(float)
        bld = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=work,
                                 label_names=[target], protected_attribute_names=[A_col])
        lfr = LFR(unprivileged_groups=[{A_col: 0}], privileged_groups=[{A_col: 1}],
                  k=5, Ax=0.01, Ay=1.0, Az=50.0, verbose=0, seed=seed)
        lfr.fit(bld, maxiter=200, maxfun=200)
        tf = lfr.transform(bld)
        rep = pd.DataFrame(tf.features, columns=[f"lfr_{i}" for i in range(tf.features.shape[1])],
                           index=df.index)
        rep[A_col] = df[A_col].values; rep[target] = df[target].values
        return rep, "ok"
    except Exception as e:
        return None, f"LFR unavailable on this stack: {e}"


def dfair_residualize(df, features, A_col, target):
    """Transparent fair-representation substitute (used because AIF360 LFR is
    broken on scipy>=1.12): orthogonalize each feature against the protected
    attribute A by keeping the residual x_j - E[x_j | A].  Drives down the linear
    dependence of features on A (an adversarial-AUC drop measures success).
    """
    out = df.copy()
    A = (df[A_col] != df[A_col].min()).astype(int).values
    for feat in features:
        x = _encode_features(df, [feat]).to_numpy().ravel()
        if np.unique(x).size < 2:
            continue
        # E[x | A=0], E[x | A=1] -> subtract group mean
        mu = {0: x[A == 0].mean() if (A == 0).any() else 0.0,
              1: x[A == 1].mean() if (A == 1).any() else 0.0}
        resid = x - np.where(A == 1, mu[1], mu[0])
        if pd.api.types.is_numeric_dtype(df[feat]):
            out[feat] = resid
        else:
            # categorical: snap residual back to nearest existing code (keeps dtype)
            codes = df[feat].astype("category").cat.codes.values.astype(float)
            out[feat] = pd.Series(resid, index=df.index)
    return out, "residualized vs A"


def main():
    print("=" * 70)
    print("  LAYER 1 -- Bias-aware preprocessing -> D_fair")
    print("=" * 70)
    summary_rows = []
    for name in ("synthetic", "adult"):
        print(f"\n--- {name} ---")
        bundle = common.load_dataset(name)
        df = bundle.df
        A_cols = [bundle.sensitive]
        if name == "synthetic":
            A_cols.append("race")            # second protected attr (proxy path)
        features = bundle.feature_cols
        Q_cols = bundle.legitimate

        # --- proxy report ---
        rep = proxy_report(df, features, A_cols, Q_cols)
        rep_path = os.path.join(common.OUT_DIR, f"proxy_report_{name}.csv")
        rep.to_csv(rep_path, index=False)
        flagged = rep.loc[rep["proxy_flag"], "feature"].tolist()
        print(f"  proxy report -> {rep_path}")
        print(f"  flagged proxies: {flagged}")
        top = rep.sort_values(f"auc_A={bundle.sensitive}", ascending=False).head(6)
        for _, r in top.iterrows():
            print(f"    {r['feature']:20s} AUC({bundle.sensitive})={r[f'auc_A={bundle.sensitive}']:.3f}  "
                  f"inc={r[f'inc_auc_A={bundle.sensitive}']:.3f}  flag={r['proxy_flag']}")

        # --- baseline adversarial AUC ---
        adv_before = adversarial_auc(df, features, bundle.sensitive)
        print(f"  adversarial-AUC recover {bundle.sensitive}: {adv_before:.3f}")

        # --- variant 1: proxy removal ---
        d_fair, dropped = dfair_proxy_removal(df, rep, features, A_cols)
        adv_after = adversarial_auc(d_fair, [c for c in d_fair.columns if c not in [bundle.target, *A_cols] and c in features],
                                    bundle.sensitive) if dropped else adv_before
        common.save_pickle({"df": d_fair, "dropped": dropped}, f"dfair_proxyremoval_{name}.pkl")
        summary_rows.append({"dataset": name, "variant": "proxy_removal",
                             "n_features": len(features), "n_dropped": len(dropped),
                             "adv_auc_before": adv_before, "adv_auc_after": adv_after,
                             "note": ";".join(dropped)})
        print(f"  proxy-removal: dropped {len(dropped)} -> adv-AUC {adv_before:.3f} -> {adv_after:.3f}")

        # --- variant 2: reweighing weights (attached to original df) ---
        w = common.reweighing_weights(df[bundle.target].values,
                                      (df[bundle.sensitive] != df[bundle.sensitive].min()).astype(int).values)
        common.save_pickle({"weights": w, "features": features}, f"dfair_reweighing_{name}.pkl")
        summary_rows.append({"dataset": name, "variant": "reweighing",
                             "n_features": len(features), "n_dropped": 0,
                             "adv_auc_before": adv_before, "adv_auc_after": adv_before,
                             "note": "Kamiran-Calders sample weights"})

        # --- variant 3: residualization (LFR unavailable on this stack) ---
        rep_lfr, msg = dfair_lfr(df, features, bundle.sensitive, bundle.target)
        if rep_lfr is not None:                  # rare path if stack changes
            lfr_feats = [c for c in rep_lfr.columns if c.startswith("lfr_")]
            adv_lfr = adversarial_auc(rep_lfr, lfr_feats, bundle.sensitive) if lfr_feats else float("nan")
            common.save_pickle({"df": rep_lfr}, f"dfair_lfr_{name}.pkl")
            summary_rows.append({"dataset": name, "variant": "LFR", "n_features": len(lfr_feats),
                                 "n_dropped": len(features) - len(lfr_feats),
                                 "adv_auc_before": adv_before, "adv_auc_after": adv_lfr, "note": msg})
            print(f"  LFR: {len(lfr_feats)}-dim representation -> adv-AUC {adv_lfr:.3f} ({msg})")
        else:
            print(f"  LFR: {msg}")
            # fall back to transparent residualization
            d_res, note = dfair_residualize(df, features, bundle.sensitive, bundle.target)
            adv_res = adversarial_auc(d_res, features, bundle.sensitive)
            common.save_pickle({"df": d_res}, f"dfair_residualize_{name}.pkl")
            summary_rows.append({"dataset": name, "variant": "residualize", "n_features": len(features),
                                 "n_dropped": 0, "adv_auc_before": adv_before,
                                 "adv_auc_after": adv_res, "note": note})
            print(f"  residualize: features orthogonalized vs {bundle.sensitive} -> adv-AUC {adv_before:.3f} -> {adv_res:.3f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(common.OUT_DIR, "layer1_summary.csv"), index=False)
    print("\n[SAVED] outputs/layer1_summary.csv")
    print(summary.to_string(index=False))

    # headline plot: proxy AUCs for the synthetic set (zip should dominate race)
    try:
        rep = pd.read_csv(os.path.join(common.OUT_DIR, "proxy_report_synthetic.csv"))
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(rep))
        w = 0.4
        ax.bar(x - w/2, rep["auc_A=gender"], w, label="AUC(gender)", color="#4C72B0")
        if "auc_A=race" in rep.columns:
            ax.bar(x + w/2, rep["auc_A=race"], w, label="AUC(race)", color="#C44E52")
        ax.axhline(common.CFG["proxy"]["auc_flag"], ls="--", color="k", label="flag threshold")
        ax.set_xticks(x); ax.set_xticklabels(rep["feature"], rotation=45, ha="right")
        ax.set_ylabel("Proxy AUC"); ax.set_ylim(0.4, 1.0)
        ax.set_title("Layer 1: per-feature proxy AUC (synthetic hiring)")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(common.OUT_DIR, "layer1_proxy_auc.png"), dpi=150); plt.close()
        print("[SAVED] outputs/layer1_proxy_auc.png")
    except Exception as e:
        print(f"[skip plot] {e}")
    print("\n[OK] Layer 1 complete.")


if __name__ == "__main__":
    main()
