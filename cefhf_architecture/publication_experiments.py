"""Leakage-free, repeated evaluation used for publication tables.

Real datasets support statistical fairness claims.  SCM counterfactual metrics
are reported only for the synthetic dataset, whose causal graph is known.
Every fold fits preprocessing, reweighing, models, and the SCM on training rows.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import common
import layer2_dag_scm as L2


@dataclass
class Spec:
    name: str
    df: pd.DataFrame
    target: str
    sensitive: str
    positive_group: object
    source: str


def load_synthetic() -> Spec:
    df = pd.read_csv(os.path.join(common.DATA_DIR, "synthetic_hiring.csv"))
    return Spec("synthetic", df, "shortlisted", "gender", "Male", "known synthetic SCM")


def load_adult() -> Spec:
    b = common.load_adult()
    return Spec("adult", b.df.drop(columns=[b.sensitive_enc_col]), b.target,
                b.sensitive, "Male", "UCI Adult (1994 Census extract)")


def load_acs(cache_dir: str, max_rows: int, seed: int) -> Spec:
    try:
        from folktables import ACSDataSource, ACSEmployment
    except ImportError as exc:
        raise RuntimeError("Install folktables==0.0.12 before running ACS") from exc
    source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person",
                           root_dir=cache_dir)
    raw = source.get_data(states=["CA"], download=True)
    features, label, group = ACSEmployment.df_to_pandas(raw)
    df = features.copy()
    df["employment"] = np.asarray(label).reshape(-1).astype(int)
    # RAC1P: 1 is White alone. Keep the original multi-category attribute for
    # intersectional auditing but use a transparent binary primary comparison.
    race = np.asarray(group).reshape(-1)
    df["race_group"] = np.where(race == 1, "White", "Non-White")
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return Spec("acs_employment_ca_2018", df, "employment", "race_group", "White",
                "Folktables ACSEmployment, California 2018 1-Year ACS PUMS")


def profile(specs: list[Spec]) -> pd.DataFrame:
    rows = []
    for s in specs:
        d = s.df
        rows.append({
            "dataset": s.name, "rows": len(d), "features": len(d.columns) - 1,
            "target_rate": d[s.target].mean(), "sensitive_groups": d[s.sensitive].nunique(),
            "missing_cells_rate": d.isna().sum().sum() / d.size,
            "duplicate_rows_rate": d.duplicated().mean(), "source": s.source,
        })
    return pd.DataFrame(rows)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def reweigh(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    w = np.ones(len(y), dtype=float)
    for g in np.unique(a):
        for label in np.unique(y):
            m = (a == g) & (y == label)
            if m.any():
                w[m] = ((a == g).mean() * (y == label).mean()) / m.mean()
    return w / w.mean()


def group_metrics(y: np.ndarray, pred: np.ndarray, a: np.ndarray, pos) -> dict:
    privileged = a == pos
    rates, tprs, fprs = [], [], []
    for mask in (privileged, ~privileged):
        rates.append(pred[mask].mean() if mask.any() else np.nan)
        yp, pp = y[mask], pred[mask]
        tprs.append(pp[yp == 1].mean() if (yp == 1).any() else np.nan)
        fprs.append(pp[yp == 0].mean() if (yp == 0).any() else np.nan)
    return {
        "SPD": abs(rates[0] - rates[1]),
        "DIR": min(rates) / max(rates) if max(rates) > 0 else np.nan,
        "EOD": np.nanmax([abs(tprs[0] - tprs[1]), abs(fprs[0] - fprs[1])]),
    }


def estimator(name: str, seed: int):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1500, random_state=seed)
    return XGBClassifier(n_estimators=75, learning_rate=.07, max_depth=4,
                         subsample=.9, colsample_bytree=.9, eval_metric="logloss",
                         n_jobs=-1, random_state=seed)


def bootstrap_mean_ci(values, seed=2024, n=2000):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, (n, len(arr)), replace=True).mean(1)
    return arr.mean(), np.quantile(means, .025), np.quantile(means, .975)


def evaluate(spec: Spec, folds: int, repeats: int, seed: int) -> pd.DataFrame:
    y = spec.df[spec.target].to_numpy().astype(int)
    X = spec.df.drop(columns=[spec.target])
    a = spec.df[spec.sensitive].astype(str).to_numpy()
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, y)):
        for model_name, use_rw in [("LogisticRegression", False), ("XGBoost", False),
                                   ("XGBoost+RW", True)]:
            pre = make_preprocessor(X.iloc[tr])
            Xt = pre.fit_transform(X.iloc[tr])
            Xv = pre.transform(X.iloc[te])
            model = estimator("XGBoost" if use_rw else model_name, seed + fold)
            weights = reweigh(y[tr], a[tr]) if use_rw else None
            model.fit(Xt, y[tr], sample_weight=weights)
            prob = model.predict_proba(Xv)[:, 1]
            pred = (prob >= .5).astype(int)
            result = {
                "dataset": spec.name, "fold": fold, "model": model_name,
                "Accuracy": accuracy_score(y[te], pred), "F1": f1_score(y[te], pred),
                "ROC-AUC": roc_auc_score(y[te], prob),
                "PR-AUC": average_precision_score(y[te], prob),
                **group_metrics(y[te], pred, a[te], str(spec.positive_group)),
            }
            # A-only sensitivity is not called counterfactual fairness.
            flipped = X.iloc[te].copy()
            vals = sorted(X.iloc[tr][spec.sensitive].astype(str).unique())
            if len(vals) == 2:
                flipped[spec.sensitive] = np.where(
                    flipped[spec.sensitive].astype(str) == vals[0], vals[1], vals[0])
                p_flip = model.predict_proba(pre.transform(flipped))[:, 1]
                result["A_flip_prob_rate"] = (np.abs(p_flip - prob) > .10).mean()
                result["A_flip_class_rate"] = ((p_flip >= .5) != pred).mean()
            rows.append(result)
    return pd.DataFrame(rows)


def evaluate_synthetic_scm(spec: Spec, folds: int, repeats: int, seed: int) -> pd.DataFrame:
    """Training-fold SCM comparison: unconstrained XGB vs CF augmentation."""
    df, y = spec.df, spec.df[spec.target].to_numpy().astype(int)
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(df, y)):
        train, test = df.iloc[tr].copy(), df.iloc[te].copy()
        # Bundle preprocessing and structural equations see training rows only.
        b = common._finalize("synthetic", df, "shortlisted", "gender",
            [c for c in df.columns if c not in ("shortlisted", "u_confounder", "gender")],
            ["education_level", "experience_years", "screening_score", "skills"], fit_df=train)
        scm = L2.build_scm(b, L2.expert_dag_synthetic(), fit_df=train)
        if len(test) > 3000:
            test = test.sample(3000, random_state=seed + fold)
        Xtr, ytr = b.transform(train), train.shortlisted.to_numpy()
        Xcf = b.transform(scm.flip_protected(train, "gender"))
        for lam in (0.0, 1.0):
            model = estimator("XGBoost", seed + fold)
            if lam:
                model.fit(np.vstack([Xtr, Xcf]), np.r_[ytr, ytr],
                          sample_weight=np.r_[np.ones(len(ytr)), np.ones(len(ytr))])
            else:
                model.fit(Xtr, ytr)
            p = model.predict_proba(b.transform(test))[:, 1]
            pcf = model.predict_proba(b.transform(scm.flip_protected(test, "gender")))[:, 1]
            pred = (p >= .5).astype(int)
            sf = group_metrics(test.shortlisted.to_numpy(), pred,
                               test.gender.astype(str).to_numpy(), "Male")
            rows.append({"dataset": "synthetic", "fold": fold,
                         "method": "CF-augmentation" if lam else "Unconstrained",
                         "ROC-AUC": roc_auc_score(test.shortlisted, p), **sf,
                         "CF_prob_rate": (np.abs(pcf - p) > .10).mean(),
                         "CF_class_rate": ((pcf >= .5) != pred).mean()})
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [c for c in raw.columns if c not in group_cols + ["fold"]]
    rows = []
    for keys, part in raw.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys)); row["n_folds"] = part.fold.nunique()
        for metric in metrics:
            if pd.api.types.is_numeric_dtype(part[metric]):
                mean, lo, hi = bootstrap_mean_ci(part[metric])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acs-max-rows", type=int, default=30000)
    ap.add_argument("--skip-acs", action="store_true")
    ap.add_argument("--causal-only", action="store_true",
                    help="reuse existing main results and run only synthetic SCM evaluation")
    args = ap.parse_args()
    specs = [load_synthetic(), load_adult()]
    if not args.skip_acs:
        specs.append(load_acs(os.path.join(common.DATA_DIR, "folktables"),
                              args.acs_max_rows, args.seed))
    os.makedirs(common.OUT_DIR, exist_ok=True)
    prof = profile(specs)
    prof.to_csv(os.path.join(common.OUT_DIR, "publication_dataset_quality.csv"), index=False)
    if not args.causal_only:
        raw = pd.concat([evaluate(s, args.folds, args.repeats, args.seed) for s in specs],
                        ignore_index=True)
        raw.to_csv(os.path.join(common.OUT_DIR, "publication_fold_results.csv"), index=False)
        summary = summarize(raw, ["dataset", "model"])
        summary.to_csv(os.path.join(common.OUT_DIR, "publication_main_results.csv"), index=False)
    else:
        summary = pd.read_csv(os.path.join(common.OUT_DIR, "publication_main_results.csv"))
    causal = evaluate_synthetic_scm(specs[0], args.folds, args.repeats, args.seed)
    causal.to_csv(os.path.join(common.OUT_DIR, "publication_causal_fold_results.csv"), index=False)
    causal_summary = summarize(causal, ["dataset", "method"])
    causal_summary.to_csv(os.path.join(common.OUT_DIR, "publication_causal_results.csv"), index=False)
    manifest = {"folds": args.folds, "repeats": args.repeats, "seed": args.seed,
                "acs_max_rows": args.acs_max_rows, "cf_threshold": .10,
                "main_results_folds": int(summary["n_folds"].min()),
                "causal_results_folds": int(causal_summary["n_folds"].min()),
                "note": "All fitted artifacts are training-fold only."}
    with open(os.path.join(common.OUT_DIR, "publication_run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(summary.to_string(index=False))
    print(causal_summary.to_string(index=False))


if __name__ == "__main__":
    main()
