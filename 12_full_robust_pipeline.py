"""
=============================================================
12_FULL_ROBUST_PIPELINE.py -- Master Research Pipeline
=============================================================
Implementation of the full robust pipeline for ML fairness:
  1. 5-Fold Cross Validation on both datasets.
  2. RandomizedSearchCV for XGBoost tuning.
  3. CFVR-Prob and CFVR-Class metrics.
  4. Preprocessing Reweighing mitigation (fit on train folds, eval on val).
  5. Outputs robust CSV tables and comparison plots.
=============================================================
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from fairlearn.metrics import (demographic_parity_difference,
                                demographic_parity_ratio,
                                equalized_odds_difference)

# ─────────────────────────────────────────────────────────
# CFVR computation
# ─────────────────────────────────────────────────────────
def compute_cfvr_both(model, X_test, gender_idx, threshold=0.10):
    """
    Returns both CFVR-Prob and CFVR-Class.
    - CFVR-Prob: probability shift |p_cf - p_orig| > threshold
    - CFVR-Class: binary prediction label flipped (prediction changed)
    """
    p_orig = model.predict_proba(X_test)[:, 1]
    X_cf = X_test.copy()
    uniq = np.unique(X_cf[:, gender_idx])
    if len(uniq) == 2:
        a, b = uniq[0], uniq[1]
        ma, mb = X_cf[:, gender_idx] == a, X_cf[:, gender_idx] == b
        X_cf[ma, gender_idx] = b
        X_cf[mb, gender_idx] = a
    else:
        X_cf[:, gender_idx] = -X_cf[:, gender_idx]

    p_cf = model.predict_proba(X_cf)[:, 1]
    delta = np.abs(p_cf - p_orig)

    cfvr_prob = (delta > threshold).mean()

    pred_orig = (p_orig >= 0.5).astype(int)
    pred_cf   = (p_cf   >= 0.5).astype(int)
    cfvr_class = (pred_orig != pred_cf).mean()

    return cfvr_prob, cfvr_class

# ─────────────────────────────────────────────────────────
# Reweighing Weights Computation (Kamiran-Calders Formula)
# ─────────────────────────────────────────────────────────
def compute_reweighing_weights(y_train, gender_train):
    """
    Computes sample weights using the Kamiran-Calders formula:
    W(x) = [P(Y=y) * P(A=a)] / P(Y=y, A=a)
    """
    n = len(y_train)
    weights = np.ones(n)
    unique_genders = np.unique(gender_train)
    unique_labels  = np.unique(y_train)

    for g in unique_genders:
        for y in unique_labels:
            mask = (gender_train == g) & (y_train == y)
            p_g = (gender_train == g).mean()
            p_y = (y_train == y).mean()
            p_gy = mask.mean()
            if p_gy > 0:
                w = (p_g * p_y) / p_gy
                weights[mask] = w
    
    # Normalize to sum to n
    weights = weights * (n / weights.sum())
    return weights

# ─────────────────────────────────────────────────────────
# Dataset Preprocessing Function
# ─────────────────────────────────────────────────────────
def prepare_data(df, target_col, sensitive_col, exclude_cols):
    df = df.dropna().copy()
    df.columns = df.columns.str.strip()
    
    # Label encode sensitive feature
    le = LabelEncoder()
    df["sensitive_enc"] = le.fit_transform(df[sensitive_col].astype(str).str.strip())
    
    # Identify numerical vs categorical
    categorical_cols = []
    numeric_cols     = []
    for col in df.columns:
        if col in exclude_cols or col == "sensitive_enc":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 10:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    # Features matrix building
    feature_cols = ["sensitive_enc"] + numeric_cols + categorical_cols
    X_raw = df[feature_cols].copy()
    y = df[target_col].copy().values
    gender_raw = df["sensitive_enc"].values
    
    # We will build the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["sensitive_enc"] + numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )
    
    # Determine the sensitive feature index in the preprocessed feature space
    # Since "sensitive_enc" is the first numeric column, its index will be 0
    sensitive_idx = 0
    
    return X_raw, y, gender_raw, preprocessor, sensitive_idx

# ─────────────────────────────────────────────────────────
# Hyperparameter Tuning for XGBoost
# ─────────────────────────────────────────────────────────
def tune_xgboost(X_train, y_train, random_state=42):
    print("    [Tuning] Running RandomizedSearchCV on XGBoost...")
    param_dist = {
        "n_estimators"     : [100, 200, 300, 400],
        "max_depth"        : [3, 4, 5, 6, 7],
        "learning_rate"    : [0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample"        : [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree" : [0.7, 0.8, 0.9, 1.0],
        "min_child_weight" : [1, 3, 5],
    }
    
    xgb = XGBClassifier(random_state=random_state, eval_metric="logloss", verbosity=0)
    
    # Optimize tuning time by subsampling if data is large
    if len(y_train) > 5000:
        # Subsample for tuning to keep it fast (e.g. 5000 samples, 3 folds)
        idx = np.random.choice(len(y_train), 5000, replace=False)
        X_tune = X_train[idx]
        y_tune = y_train[idx]
        cv_folds = 3
        n_iter = 15
    else:
        X_tune = X_train
        y_tune = y_train
        cv_folds = 5
        n_iter = 25

    rscv = RandomizedSearchCV(
        xgb, param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    rscv.fit(X_tune, y_tune)
    print(f"    [Tuning] Best parameters found: {rscv.best_params_}")
    return rscv.best_params_

# ─────────────────────────────────────────────────────────
# Main 5-Fold Cross Validation Pipeline
# ─────────────────────────────────────────────────────────
def run_cv_pipeline(X_raw, y, gender_raw, preprocessor, sensitive_idx, best_xgb_params, random_state=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    models_to_eval = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest"      : lambda: RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "XGBoost (Base)"     : lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                 random_state=random_state, eval_metric="logloss", verbosity=0),
        "XGBoost (Tuned)"    : lambda: XGBClassifier(**best_xgb_params, random_state=random_state, 
                                                 eval_metric="logloss", verbosity=0),
        "XGBoost (Tuned + RW)": lambda: XGBClassifier(**best_xgb_params, random_state=random_state, 
                                                  eval_metric="logloss", verbosity=0)
    }
    
    cv_metrics = {name: [] for name in models_to_eval}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_raw, y), 1):
        print(f"  Fold {fold}/5...", end=" ", flush=True)
        
        # Preprocess splits to prevent leakage
        X_train_raw = X_raw.iloc[train_idx]
        X_val_raw   = X_raw.iloc[val_idx]
        
        X_train_fold = preprocessor.fit_transform(X_train_raw)
        X_val_fold   = preprocessor.transform(X_val_raw)
        
        y_train_fold = y[train_idx]
        y_val_fold   = y[val_idx]
        
        gender_train_fold = gender_raw[train_idx]
        gender_val_fold   = gender_raw[val_idx]
        
        # Compute reweighing weights
        sample_weights = compute_reweighing_weights(y_train_fold, gender_train_fold)
        
        for name, model_creator in models_to_eval.items():
            model = model_creator()
            
            # Train model
            if name == "XGBoost (Tuned + RW)":
                model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)
            else:
                model.fit(X_train_fold, y_train_fold)
                
            # Predict
            y_pred = model.predict(X_val_fold)
            y_proba = model.predict_proba(X_val_fold)[:, 1]
            
            # Performance metrics
            acc = accuracy_score(y_val_fold, y_pred)
            f1  = f1_score(y_val_fold, y_pred, zero_division=0)
            auc = roc_auc_score(y_val_fold, y_proba)
            
            # Group fairness metrics
            spd = abs(demographic_parity_difference(y_val_fold, y_pred, sensitive_features=gender_val_fold))
            try:
                dir_ = demographic_parity_ratio(y_val_fold, y_pred, sensitive_features=gender_val_fold)
            except Exception:
                dir_ = float("nan")
            eod = equalized_odds_difference(y_val_fold, y_pred, sensitive_features=gender_val_fold)
            
            # Counterfactual sensitivity metrics
            cfvr_prob, cfvr_class = compute_cfvr_both(model, X_val_fold, sensitive_idx)
            
            cv_metrics[name].append({
                "Accuracy"  : acc,
                "F1"        : f1,
                "ROC-AUC"   : auc,
                "SPD"       : spd,
                "DIR"       : dir_,
                "EOD"       : eod,
                "CFVR-Prob" : cfvr_prob,
                "CFVR-Class": cfvr_class
            })
            
    print("done!")
    
    # Aggregate metrics
    results = {}
    for name in models_to_eval:
        metrics_list = cv_metrics[name]
        results[name] = {}
        for m in ["Accuracy", "F1", "ROC-AUC", "SPD", "DIR", "EOD", "CFVR-Prob", "CFVR-Class"]:
            vals = [fold_m[m] for fold_m in metrics_list if not np.isnan(fold_m[m])]
            results[name][f"{m}_mean"] = np.mean(vals)
            results[name][f"{m}_std"]  = np.std(vals)
            
    return results

# ─────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()
    
    # ── 1. RECRUITMENT DATASET ───────────────────────────────
    print("\n" + "=" * 80)
    print("  PHASE 1: RUNNING PIPELINE ON RECRUITMENT BIAS DATASET")
    print("=" * 80)
    
    rec_df = pd.read_csv("data/archive/Dataset.csv")
    rec_exclude = ["shortlisted", "gender", "Gender_enc"]
    rec_X_raw, rec_y, rec_gender_raw, rec_preprocessor, rec_sensitive_idx = prepare_data(
        rec_df, target_col="shortlisted", sensitive_col="gender", exclude_cols=rec_exclude
    )
    
    # Fit once on train split to tune parameters
    rec_X_train_raw, _, rec_y_train, _ = train_test_split(rec_X_raw, rec_y, test_size=0.2, random_state=42, stratify=rec_y)
    rec_X_train = rec_preprocessor.fit_transform(rec_X_train_raw)
    
    rec_best_params = tune_xgboost(rec_X_train, rec_y_train)
    
    print("\n  [CV] Running Stratified 5-Fold CV...")
    rec_results = run_cv_pipeline(
        rec_X_raw, rec_y, rec_gender_raw, rec_preprocessor, rec_sensitive_idx, rec_best_params
    )
    
    # ── 2. ADULT UCI DATASET ─────────────────────────────────
    print("\n" + "=" * 80)
    print("  PHASE 2: RUNNING PIPELINE ON ADULT UCI BENCHMARK")
    print("=" * 80)
    
    cols = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"
    ]
    adult_path = "data/adult/adult.data"
    adult_df = pd.read_csv(adult_path, names=cols, skipinitialspace=True, na_values="?")
    adult_df["income"] = adult_df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    
    adult_exclude = ["income", "sex", "sex_enc"]
    adult_X_raw, adult_y, adult_gender_raw, adult_preprocessor, adult_sensitive_idx = prepare_data(
        adult_df, target_col="income", sensitive_col="sex", exclude_cols=adult_exclude
    )
    
    # Fit once on train split to tune parameters
    adult_X_train_raw, _, adult_y_train, _ = train_test_split(adult_X_raw, adult_y, test_size=0.2, random_state=42, stratify=adult_y)
    adult_X_train = adult_preprocessor.fit_transform(adult_X_train_raw)
    
    adult_best_params = tune_xgboost(adult_X_train, adult_y_train)
    
    print("\n  [CV] Running Stratified 5-Fold CV...")
    adult_results = run_cv_pipeline(
        adult_X_raw, adult_y, adult_gender_raw, adult_preprocessor, adult_sensitive_idx, adult_best_params
    )
    
    # ── 3. FORMAT TABLES ─────────────────────────────────────
    def build_results_df(results_dict):
        rows = []
        metrics_cols = ["Accuracy", "F1", "ROC-AUC", "DIR", "SPD", "EOD", "CFVR-Prob", "CFVR-Class"]
        for model_name, metrics in results_dict.items():
            row = {"Model": model_name}
            for m in metrics_cols:
                mean = metrics[f"{m}_mean"]
                std  = metrics[f"{m}_std"]
                row[m] = f"{mean:.4f} ± {std:.4f}"
            rows.append(row)
        return pd.DataFrame(rows)

    rec_df_results = build_results_df(rec_results)
    adult_df_results = build_results_df(adult_results)
    
    print("\n" + "=" * 100)
    print("  FINAL 5-FOLD CV RESULTS: RECRUITMENT BIAS DATASET (Mean ± Std)")
    print("=" * 100)
    print(rec_df_results.to_markdown(index=False))
    
    print("\n" + "=" * 100)
    print("  FINAL 5-FOLD CV RESULTS: ADULT UCI BENCHMARK (Mean ± Std)")
    print("=" * 100)
    print(adult_df_results.to_markdown(index=False))
    
    # Save CSVs
    rec_df_results.to_csv("outputs/robust_results_recruitment.csv", index=False)
    adult_df_results.to_csv("outputs/robust_results_adult.csv", index=False)
    print("\n[SAVED] outputs/robust_results_recruitment.csv")
    print("[SAVED] outputs/robust_results_adult.csv")
    
    # Save raw pickle for final walkthrough
    cv_pipeline_bundle = {
        "rec_results"      : rec_results,
        "adult_results"    : adult_results,
        "rec_best_params"  : rec_best_params,
        "adult_best_params": adult_best_params
    }
    with open("data/cv_pipeline_results.pkl", "wb") as f:
        pickle.dump(cv_pipeline_bundle, f)
    print("[SAVED] data/cv_pipeline_results.pkl")
    
    # ── 4. VISUALIZATIONS ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Robust CV Evaluations: Performance & Fairness Trade-offs", fontsize=15, fontweight="bold")
    
    model_names = list(rec_results.keys())
    x = np.arange(len(model_names))
    width = 0.35
    
    # 1. Accuracy vs ROC-AUC (Recruitment)
    rec_acc_mean = [rec_results[m]["Accuracy_mean"] for m in model_names]
    rec_acc_std  = [rec_results[m]["Accuracy_std"]  for m in model_names]
    rec_auc_mean = [rec_results[m]["ROC-AUC_mean"]  for m in model_names]
    rec_auc_std  = [rec_results[m]["ROC-AUC_std"]   for m in model_names]
    
    axes[0, 0].bar(x - width/2, rec_acc_mean, width, yerr=rec_acc_std, label="Accuracy", color="#4C72B0", alpha=0.85, capsize=4)
    axes[0, 0].bar(x + width/2, rec_auc_mean, width, yerr=rec_auc_std, label="ROC-AUC", color="#55A868", alpha=0.85, capsize=4)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha="right")
    axes[0, 0].set_ylim(0.4, 0.85)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Performance Metrics (Recruitment Dataset)")
    axes[0, 0].legend()
    
    # 2. SPD vs CFVR (Recruitment)
    rec_spd_mean = [rec_results[m]["SPD_mean"] for m in model_names]
    rec_spd_std  = [rec_results[m]["SPD_std"]  for m in model_names]
    rec_cfvr_p_mean = [rec_results[m]["CFVR-Prob_mean"] for m in model_names]
    rec_cfvr_p_std  = [rec_results[m]["CFVR-Prob_std"]  for m in model_names]
    
    axes[0, 1].bar(x - width/2, rec_spd_mean, width, yerr=rec_spd_std, label="|SPD|", color="#C44E52", alpha=0.85, capsize=4)
    axes[0, 1].bar(x + width/2, rec_cfvr_p_mean, width, yerr=rec_cfvr_p_std, label="CFVR-Prob", color="#8172B3", alpha=0.85, capsize=4)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha="right")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Fairness & Sensitivity Metrics (Recruitment Dataset)")
    axes[0, 1].legend()
    
    # 3. Accuracy vs ROC-AUC (Adult UCI)
    adult_acc_mean = [adult_results[m]["Accuracy_mean"] for m in model_names]
    adult_acc_std  = [adult_results[m]["Accuracy_std"]  for m in model_names]
    adult_auc_mean = [adult_results[m]["ROC-AUC_mean"]  for m in model_names]
    adult_auc_std  = [adult_results[m]["ROC-AUC_std"]   for m in model_names]
    
    axes[1, 0].bar(x - width/2, adult_acc_mean, width, yerr=adult_acc_std, label="Accuracy", color="#4C72B0", alpha=0.85, capsize=4)
    axes[1, 0].bar(x + width/2, adult_auc_mean, width, yerr=adult_auc_std, label="ROC-AUC", color="#55A868", alpha=0.85, capsize=4)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha="right")
    axes[1, 0].set_ylim(0.4, 0.95)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Performance Metrics (Adult UCI)")
    axes[1, 0].legend()
    
    # 4. SPD vs CFVR (Adult UCI)
    adult_spd_mean = [adult_results[m]["SPD_mean"] for m in model_names]
    adult_spd_std  = [adult_results[m]["SPD_std"]  for m in model_names]
    adult_cfvr_p_mean = [adult_results[m]["CFVR-Prob_mean"] for m in model_names]
    adult_cfvr_p_std  = [adult_results[m]["CFVR-Prob_std"]  for m in model_names]
    
    axes[1, 1].bar(x - width/2, adult_spd_mean, width, yerr=adult_spd_std, label="|SPD|", color="#C44E52", alpha=0.85, capsize=4)
    axes[1, 1].bar(x + width/2, adult_cfvr_p_mean, width, yerr=adult_cfvr_p_std, label="CFVR-Prob", color="#8172B3", alpha=0.85, capsize=4)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha="right")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Fairness & Sensitivity Metrics (Adult UCI)")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("outputs/robust_cv_plots.png", dpi=150)
    plt.close()
    print("[SAVED] outputs/robust_cv_plots.png")
    
    print(f"\n[OK] Pipeline completed successfully in {time.time() - t_start:.1f}s")
    print("=" * 80 + "\n")
