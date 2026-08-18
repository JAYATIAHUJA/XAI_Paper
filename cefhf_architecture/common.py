"""
common.py -- shared utilities for the CEFHF six-layer architecture.

Single source of data loading, preprocessing, predictive metrics, the CFVR-flip
baseline (direct-effect sensitivity, NOT counterfactual fairness), Kamiran-Calders
reweighing, and bootstrap confidence intervals.  All layer scripts import from here
so that interfaces stay consistent.

Conventions mirror XAI_Paper/12_full_robust_pipeline.py (sensitive feature is
label-encoded numeric and placed first in the transformed feature space, so its
index is always 0).
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
CFG_PATH = os.path.join(ROOT, "configs", "seeds.yaml")
REPO_ROOT = os.path.dirname(ROOT)                       # .../research
# The architecture now lives directly below the repository root.  Older copies
# were nested in an ``XAI_Paper`` directory, so retain that layout only as a
# backwards-compatible fallback.
_legacy_root = os.path.join(REPO_ROOT, "XAI_Paper")
XAI_PAPER = _legacy_root if os.path.isdir(_legacy_root) else REPO_ROOT
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    with open(CFG_PATH, "r") as fh:
        return yaml.safe_load(fh)


CFG = load_config()


# --------------------------------------------------------------------------- #
# Dataset bundle
# --------------------------------------------------------------------------- #
@dataclass
class DatasetBundle:
    """One prepared dataset.

    `df` holds the cleaned, *raw* (human-readable) dataframe -- the SCM in Layer 2
    operates on these original columns.  `preprocessor` + `sensitive_idx` give the
    numeric matrix the classifiers in Layer 3/4 consume.
    """
    name: str
    df: pd.DataFrame
    target: str
    sensitive: str                       # protected attribute A (raw column name)
    feature_cols: list                   # columns used as model features (raw)
    numeric_cols: list
    categorical_cols: list
    legitimate: list                     # Q: legitimate qualifications (for proxy | Q)
    preprocessor: ColumnTransformer
    sensitive_enc_col: str               # name of the encoded-sensitive column added to df
    sensitive_idx: int = 0               # position of A in the transformed matrix

    # -- transformed matrix helpers -----------------------------------------
    def transform(self, df_subset: pd.DataFrame) -> np.ndarray:
        prepared = df_subset.copy()
        if self.sensitive_enc_col not in prepared:
            classes = sorted(self.df[self.sensitive].astype(str).str.strip().unique())
            mapping = {value: idx for idx, value in enumerate(classes)}
            prepared[self.sensitive_enc_col] = (
                prepared[self.sensitive].astype(str).str.strip().map(mapping)
            )
        X_raw = prepared[self.feature_cols_with_enc()]
        return self.preprocessor.transform(X_raw)

    def feature_cols_with_enc(self) -> list:
        return [self.sensitive_enc_col] + self.numeric_cols + self.categorical_cols

    def feature_names_out(self) -> list:
        """Names of columns in the *transformed* matrix (after one-hot encoding).
        Built manually from the fitted transformer so it never mismatches X.shape[1]
        (sklearn's get_feature_names_out can raise on some pipelines)."""
        try:
            names = []
            num_cols = self.preprocessor.transformers_[0][2]      # [enc] + numeric
            names += [f"num__{c}" for c in num_cols]
            ohe = self.preprocessor.named_transformers_["cat"]
            cat_cols = self.preprocessor.transformers_[1][2]
            for col, cats in zip(cat_cols, ohe.categories_):
                for cat in cats:
                    names.append(f"cat__{col}_{cat}")
            return names
        except Exception:
            X = self.preprocessor.transform(self.df[self.feature_cols_with_enc()].head(4))
            return [f"f{i}" for i in range(X.shape[1])]

    def X_y(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_matrix, y, A_encoded) over the whole df."""
        X = self.preprocessor.transform(self.df[self.feature_cols_with_enc()])
        y = self.df[self.target].values
        A = self.df[self.sensitive_enc_col].values
        return X, y, A


def _build_preprocessor(numeric_with_sensitive: list, categorical: list) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_with_sensitive),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def _split_feature_types(df, feature_cols, sensitive):
    numeric, categorical = [], []
    for col in feature_cols:
        if col == sensitive:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # low-cardinality numeric (e.g. education-num) treated as categorical
            (categorical if df[col].nunique() < 10 else numeric).append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def load_adult() -> DatasetBundle:
    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
    ]
    path = os.path.join(XAI_PAPER, "data", "adult", "adult.data")
    df = pd.read_csv(path, names=cols, skipinitialspace=True, na_values="?").dropna()
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    # fnlwgt is a sample weight, not a candidate feature -> drop
    feature_cols = [c for c in cols if c not in ("income", "fnlwgt")]
    return _finalize("adult", df, target="income", sensitive="sex", feature_cols=feature_cols,
                     legitimate=["education-num", "hours-per-week", "capital-gain",
                                 "capital-loss", "age"])


def load_recruitment() -> DatasetBundle:
    path = os.path.join(XAI_PAPER, "data", "archive", "Dataset.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna()
    feature_cols = [c for c in df.columns if c not in ("shortlisted",)]
    return _finalize("recruitment", df, target="shortlisted", sensitive="gender",
                     feature_cols=feature_cols,
                     legitimate=["education_level", "experience_years", "screening_score"])


def load_synthetic() -> DatasetBundle:
    path = os.path.join(DATA_DIR, "synthetic_hiring.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    feature_cols = [c for c in df.columns if c not in ("shortlisted", "u_confounder")]
    return _finalize("synthetic", df, target="shortlisted", sensitive="gender",
                     feature_cols=feature_cols,
                     legitimate=["education_level", "experience_years", "screening_score",
                                 "skills"])


def _finalize(name, df, target, sensitive, feature_cols, legitimate,
              fit_df=None) -> DatasetBundle:
    df = df.copy()
    # label-encode the protected attribute (placed first in the transformed space)
    le = LabelEncoder()
    enc_col = f"_{sensitive}_enc"
    df[enc_col] = le.fit_transform(df[sensitive].astype(str).str.strip())
    # restrict feature_cols to ones present
    feature_cols = [c for c in feature_cols if c in df.columns and c != sensitive]
    numeric, categorical = _split_feature_types(df, feature_cols, sensitive)
    pre = _build_preprocessor([enc_col] + numeric, categorical)
    # Callers running an experiment must pass the training partition as fit_df.
    # Loading a standalone bundle still fits all rows for exploratory use only.
    source = df if fit_df is None else fit_df.copy()
    if enc_col not in source:
        source[enc_col] = le.transform(source[sensitive].astype(str).str.strip())
    X_raw = source[[enc_col] + numeric + categorical]
    pre.fit(X_raw)
    return DatasetBundle(
        name=name, df=df, target=target, sensitive=sensitive,
        feature_cols=feature_cols, numeric_cols=numeric, categorical_cols=categorical,
        legitimate=legitimate, preprocessor=pre, sensitive_enc_col=enc_col, sensitive_idx=0,
    )


def load_dataset(name: str) -> DatasetBundle:
    return {"adult": load_adult, "recruitment": load_recruitment,
            "synthetic": load_synthetic}[name]()


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
def make_xgb(random_state=42, **kwargs) -> XGBClassifier:
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                  eval_metric="logloss", verbosity=0, random_state=random_state)
    params.update(kwargs)
    return XGBClassifier(**params)


def make_baselines(random_state=42) -> dict:
    return {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=random_state),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "XGBoost": lambda: make_xgb(random_state),
    }


# --------------------------------------------------------------------------- #
# CFVR-flip baseline  (direct-effect / unawareness-violation sensitivity)
# --------------------------------------------------------------------------- #
def compute_cfvr_flip(model, X, sensitive_idx, tau=0.10) -> tuple[float, float]:
    """CFVR-flip: flip the (encoded) sensitive column and re-predict.

    This is the Level-2 intervention on A *alone* -- it is NOT Kusner-style
    counterfactual fairness (which needs an SCM to propagate A -> descendants).
    Kept as the honest `direct-effect / unawareness-violation` baseline so the
    paper can show CFVR-SCM != CFVR-flip.  Returns (cfvr_prob, cfvr_class).
    """
    p_orig = model.predict_proba(X)[:, 1]
    Xcf = X.copy()
    uniq = np.unique(Xcf[:, sensitive_idx])
    if len(uniq) == 2:
        a, b = uniq[0], uniq[1]
        ma, mb = Xcf[:, sensitive_idx] == a, Xcf[:, sensitive_idx] == b
        Xcf[ma, sensitive_idx] = b
        Xcf[mb, sensitive_idx] = a
    else:
        Xcf[:, sensitive_idx] = -Xcf[:, sensitive_idx]
    p_cf = model.predict_proba(Xcf)[:, 1]
    cfvr_prob = (np.abs(p_cf - p_orig) > tau).mean()
    pred_o = (p_orig >= 0.5).astype(int)
    pred_c = (p_cf >= 0.5).astype(int)
    cfvr_class = (pred_o != pred_c).mean()
    return float(cfvr_prob), float(cfvr_class)


# --------------------------------------------------------------------------- #
# Reweighing (Kamiran-Calders) -- reuse from 12_full_robust_pipeline
# --------------------------------------------------------------------------- #
def reweighing_weights(y, A) -> np.ndarray:
    n = len(y)
    w = np.ones(n)
    for g in np.unique(A):
        for lab in np.unique(y):
            mask = (A == g) & (y == lab)
            p_g = (A == g).mean()
            p_y = (y == lab).mean()
            p_gy = mask.mean()
            if p_gy > 0:
                w[mask] = (p_g * p_y) / p_gy
    return w * (n / w.sum())


# --------------------------------------------------------------------------- #
# Statistical fairness metrics
# --------------------------------------------------------------------------- #
def stat_fairness(y_true, y_pred, A) -> dict:
    spd = abs(demographic_parity_difference(y_true, y_pred, sensitive_features=A))
    try:
        dir_ = demographic_parity_ratio(y_true, y_pred, sensitive_features=A)
    except Exception:
        dir_ = float("nan")
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=A)
    return {"SPD": spd, "DIR": dir_, "EOD": eod}


def perf_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Differential Fairness epsilon (Foulds et al.) -- intersectional
# --------------------------------------------------------------------------- #
def differential_fairness_epsilon(y_pred, groups) -> float:
    """DF epsilon = max over subgroup pairs of |log P(Yhat=1|g) - log P(Yhat=1|g')|.

    `groups` is an array of subgroup labels (e.g. gender x age-band tuples).
    """
    rates = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() == 0:
            continue
        p = y_pred[m].mean()
        p = np.clip(p, 1e-6, 1 - 1e-6)
        rates[g] = np.log(p)
    if len(rates) < 2:
        return 0.0
    vals = np.array(list(rates.values()))
    return float(np.max(vals) - np.min(vals))


def intersectional_groups(A, secondary) -> np.ndarray:
    """Combine the primary protected attr with a secondary one (e.g. age band)."""
    return np.array([f"{a}|{s}" for a, s in zip(A, secondary)])


def age_bands(ages: np.ndarray) -> np.ndarray:
    bins = [-np.inf, 25, 35, 45, 55, np.inf]
    labels = ["<25", "25-34", "35-44", "45-54", "55+"]
    return np.asarray(pd.cut(ages, bins=bins, labels=labels).astype(str).to_numpy(), dtype=object)


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Iterable, n_boot=200, alpha=0.05, seed=2024) -> tuple[float, float, float]:
    """Return (mean, lo, hi) bootstrap CI for a vector of fold-level values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# --------------------------------------------------------------------------- #
# CV split helper
# --------------------------------------------------------------------------- #
def cv_splits(X, y, n_splits=5, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def train_val_split(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_pickle(obj, name):
    path = os.path.join(DATA_DIR, name)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return path


def load_pickle(name):
    with open(os.path.join(DATA_DIR, name), "rb") as fh:
        return pickle.load(fh)
