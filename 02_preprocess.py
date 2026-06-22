"""
=============================================================
STEP 2 -- Preprocessing
=============================================================
Input  : data/recruitment_bias.csv
Output : data/processed.pkl  (train/test splits + encoders)
=============================================================
Run:
    python 02_preprocess.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

os.makedirs("data", exist_ok=True)

print("\n" + "=" * 60)
print("  STEP 2 -- PREPROCESSING")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv("data/archive/Dataset.csv")
print(f"\n[INFO] Loaded dataset: {df.shape}")

# ── Drop nulls ────────────────────────────────────────────
df = df.dropna()
print(f"[INFO] After dropping nulls: {df.shape}")

# ── Detect column names ───────────────────────────────────
# Normalize column names (strip spaces, lowercase for comparison)
df.columns = df.columns.str.strip()
print(f"[INFO] Columns: {list(df.columns)}")

TARGET = "shortlisted"
SENSITIVE_COL = "gender"   # protected attribute for fairness

# ── Label encode Gender for AIF360 compatibility ─────────
# AIF360 needs numeric sensitive attribute
# Male=1 (privileged), Female=0 (unprivileged)
le_gender = LabelEncoder()
df["Gender_enc"] = le_gender.fit_transform(df["gender"])  # keep original gender too
gender_mapping = dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))
print(f"\n[INFO] Gender encoding: {gender_mapping}")

# ── Identify column types ─────────────────────────────────
# CRITICAL: exclude 'gender' (string) from categorical_cols.
# We already encoded it as Gender_enc (0/1 numeric).
# Keeping both Gender_enc AND OHE'd gender_Female/gender_Male creates:
#   - Redundant/collinear features
#   - LR overfits on OHE columns -> SPD anomaly
#   - CFVR fails because flipping Gender_enc doesn't affect OHE columns
EXCLUDE_COLS = [TARGET, "gender", "Gender_enc"]

categorical_cols = []
numeric_cols     = []

for col in df.columns:
    if col in EXCLUDE_COLS:
        continue
    if df[col].dtype == "object" or df[col].nunique() < 10:
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)

print(f"[INFO] Numeric features    : {numeric_cols}")
print(f"[INFO] Categorical features: {categorical_cols}")
print(f"[INFO] Excluded from features (handled separately): {EXCLUDE_COLS}")

# ── Build features ────────────────────────────────────────
# Keep Gender_enc (numeric 0/1) as a feature
feature_cols = ["Gender_enc"] + numeric_cols + categorical_cols
X_raw = df[feature_cols].copy()
y     = df[TARGET].copy()

print(f"\n[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")
print(f"[INFO] Target distribution:\n{y.value_counts().to_string()}")

# ── Build sklearn ColumnTransformer ───────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Gender_enc"] + numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder="drop"
)

# ── Train / Test split ────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[INFO] Train size: {X_train_raw.shape[0]}")
print(f"[INFO] Test size : {X_test_raw.shape[0]}")

# ── Fit & transform ───────────────────────────────────────
X_train = preprocessor.fit_transform(X_train_raw)
X_test  = preprocessor.transform(X_test_raw)

# Get feature names after OneHotEncoding
ohe_feature_names = []
if categorical_cols:
    ohe = preprocessor.named_transformers_["cat"]
    ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols))

feature_names = ["Gender_enc"] + numeric_cols + ohe_feature_names
print(f"\n[INFO] Total features after encoding: {len(feature_names)}")
print(f"[INFO] Feature names: {feature_names}")

# ── Also keep raw Gender column for fairness auditing ─────
gender_test = X_test_raw["Gender_enc"].values  # 0=Female, 1=Male

# ── Save everything ───────────────────────────────────────
bundle = {
    "X_train"      : X_train,
    "X_test"       : X_test,
    "y_train"      : y_train.values,
    "y_test"       : y_test.values,
    "feature_names": feature_names,
    "gender_test"  : gender_test,
    "gender_train" : X_train_raw["Gender_enc"].values,
    "preprocessor" : preprocessor,
    "le_gender"    : le_gender,
    "gender_mapping": gender_mapping,
    "df_raw"       : df,
    "X_test_raw"   : X_test_raw,
}

with open("data/processed.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n[SUCCESS] Saved data/processed.pkl")
print("\nContents saved:")
for k, v in bundle.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}")
    elif isinstance(v, (list, dict)):
        print(f"  {k}: len={len(v)}")
    else:
        print(f"  {k}: {type(v).__name__}")

print("\n[NEXT STEP] Run: python 03_train_baseline.py")
print("=" * 60 + "\n")
