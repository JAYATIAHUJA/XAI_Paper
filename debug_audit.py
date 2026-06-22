"""Debug script — diagnose LR SPD anomaly and CFVR bug"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
with open("data/models.pkl", "rb") as f:
    mdata = pickle.load(f)

y_test      = data["y_test"]
gender_test = data["gender_test"]
X_test      = data["X_test"]
feature_names = data["feature_names"]

print("=== FEATURE NAMES IN PREPROCESSED DATA ===")
for i, n in enumerate(feature_names):
    tag = " <-- GENDER" if "gender" in n.lower() else ""
    print(f"  [{i}] {n}{tag}")

print("\n=== GROUP-WISE SELECTION RATES ===")
for name, preds in mdata["predictions"].items():
    y_pred  = preds["y_pred"]
    female_mask = gender_test == 0
    male_mask   = gender_test == 1
    female_pos  = y_pred[female_mask].mean() * 100
    male_pos    = y_pred[male_mask].mean()   * 100
    spd_check   = female_pos - male_pos
    total_pos   = y_pred.sum()
    print(f"  {name}:")
    print(f"    Female: {female_mask.sum()} samples, predicted shortlisted={female_pos:.1f}%")
    print(f"    Male  : {male_mask.sum()} samples, predicted shortlisted={male_pos:.1f}%")
    print(f"    SPD check (F-M): {spd_check:+.1f}pp  | total predicted 1s: {total_pos}/{len(y_test)}")

print("\n=== CONFUSION MATRICES ===")
for name, preds in mdata["predictions"].items():
    y_pred = preds["y_pred"]
    cm = confusion_matrix(y_test, y_pred)
    total_pred_1 = y_pred.sum()
    print(f"  {name}: TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"    Total predicted Shortlisted=1: {total_pred_1} ({total_pred_1/len(y_test)*100:.1f}%)")

print("\n=== ROOT CAUSE: REDUNDANT GENDER FEATURES ===")
gender_features = [(i, n) for i, n in enumerate(feature_names) if "gender" in n.lower()]
print(f"  Found {len(gender_features)} gender-related features:")
for i, n in gender_features:
    print(f"    [{i}] {n}")
print()
print("  PROBLEM 1: Gender_enc AND gender_Female/gender_Male are BOTH in the feature set.")
print("  This is redundant/collinear. Logistic Regression can fit one and ignore the other.")
print()
print("  PROBLEM 2: CFVR only flips index [0] = Gender_enc.")
print("  But LR mainly uses gender_Female (idx 4) and gender_Male (idx 5).")
print("  Flipping only Gender_enc has NO effect on LR predictions -> CFVR=0 is a BUG.")
print()
print("  FIX: Remove 'gender' from categorical columns in preprocessing.")
print("  Keep only Gender_enc (numeric 0/1). This eliminates redundancy.")
print("  Then CFVR correctly flips the single gender index.")
