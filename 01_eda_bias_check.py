"""
=============================================================
STEP 1 -- EDA + Bias Detection
=============================================================
Input  : data/recruitment_bias.csv
Output : outputs/eda_*.png  +  console bias report
=============================================================
Run:
    python 01_eda_bias_check.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = {"Male": "#4C72B0", "Female": "#DD8452"}

# ── Load ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1 -- EXPLORATORY DATA ANALYSIS & BIAS CHECK")
print("=" * 60)

CSV_PATH = "data/archive/Dataset.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] {CSV_PATH} not found.\n"
        "Run 00_download_dataset.py first."
    )

df = pd.read_csv(CSV_PATH)
print(f"\n[INFO] Shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}\n")
print(df.info())
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe().to_string())

# ── 1. Target distribution ────────────────────────────────
print("\n" + "-" * 50)
print("TARGET DISTRIBUTION -- Shortlisted")
print("-" * 50)
vc = df["shortlisted"].value_counts()
print(vc)
print(f"  Shortlisted rate: {df['shortlisted'].mean():.2%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Target Variable -- Shortlisted", fontsize=14, fontweight="bold")
vc.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"], edgecolor="white")
axes[0].set_title("Count")
axes[0].set_xlabel("Shortlisted")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)
axes[1].pie(vc.values, labels=["Not Shortlisted", "Shortlisted"],
            autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"], startangle=140)
axes[1].set_title("Proportion")
plt.tight_layout()
plt.savefig("outputs/eda_01_target_distribution.png", dpi=150)
plt.close()
print("[SAVED] outputs/eda_01_target_distribution.png")

# ── 2. GENDER BIAS -- the critical check ──────────────────
print("\n" + "=" * 50)
print("BIAS CHECK -- GENDER vs SHORTLISTED")
print("=" * 50)

gender_table = pd.crosstab(df["gender"], df["shortlisted"], normalize="index") * 100
print("\nShortlisting rate by Gender (%):")
print(gender_table.rename(columns={0: "Not Shortlisted %", 1: "Shortlisted %"}).to_string())

shortlist_by_gender = df.groupby("gender")["shortlisted"].mean() * 100
male_rate   = shortlist_by_gender.get("Male", 0)
female_rate = shortlist_by_gender.get("Female", 0)
gap = male_rate - female_rate

print(f"\n  Male shortlisting rate   : {male_rate:.1f}%")
print(f"  Female shortlisting rate : {female_rate:.1f}%")
print(f"  GAP (M - F)              : {gap:+.1f}%")
if abs(gap) > 5:
    print(f"\n  [WARN]  POTENTIAL GENDER BIAS DETECTED (gap > 5%)")
else:
    print(f"\n  [OK]  No strong gender bias detected (gap <= 5%)")

fig, ax = plt.subplots(figsize=(8, 5))
colors = [PALETTE.get(g, "#999") for g in shortlist_by_gender.index]
bars = ax.bar(shortlist_by_gender.index, shortlist_by_gender.values,
              color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, shortlist_by_gender.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_title("Shortlisting Rate by Gender", fontsize=14, fontweight="bold")
ax.set_ylabel("Shortlisted (%)")
ax.set_xlabel("Gender")
ax.set_ylim(0, 100)
ax.axhline(df["shortlisted"].mean() * 100, color="red", linestyle="--",
           linewidth=1.5, label=f"Overall avg: {df['shortlisted'].mean()*100:.1f}%")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda_02_gender_bias.png", dpi=150)
plt.close()
print("[SAVED] outputs/eda_02_gender_bias.png")

# ── 3. Education bias ─────────────────────────────────────
print("\n" + "-" * 50)
print("BIAS CHECK -- EDUCATION vs SHORTLISTED")
print("-" * 50)
edu_rate = df.groupby("education_level")["shortlisted"].mean() * 100
print(edu_rate.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
edu_rate.sort_values(ascending=False).plot(kind="bar", ax=ax,
    color=sns.color_palette("Blues_d", len(edu_rate)), edgecolor="white")
ax.set_title("Shortlisting Rate by Education Level", fontsize=14, fontweight="bold")
ax.set_ylabel("Shortlisted (%)")
ax.set_xlabel("Education")
ax.tick_params(axis="x", rotation=30)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/eda_03_education_bias.png", dpi=150)
plt.close()
print("[SAVED] outputs/eda_03_education_bias.png")

# ── 4. Age distribution ───────────────────────────────────
if "age" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Age Distribution", fontsize=13, fontweight="bold")
    for g, grp in df.groupby("gender"):
        axes[0].hist(grp["age"], bins=20, alpha=0.6, label=g,
                     color=PALETTE.get(g, "#aaa"), edgecolor="white")
    axes[0].set_title("Age by Gender")
    axes[0].set_xlabel("age")
    axes[0].legend()
    # Age vs Shortlisted
    for sl, grp in df.groupby("shortlisted"):
        axes[1].hist(grp["age"], bins=20, alpha=0.6,
                     label=f"Shortlisted={sl}", edgecolor="white")
    axes[1].set_title("Age by Shortlisting")
    axes[1].set_xlabel("age")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/eda_04_age_distribution.png", dpi=150)
    plt.close()
    print("[SAVED] outputs/eda_04_age_distribution.png")

# ── 5. Feature correlations ───────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/eda_05_correlation_heatmap.png", dpi=150)
    plt.close()
    print("[SAVED] outputs/eda_05_correlation_heatmap.png")

# ── 6. Experience vs Shortlisted ─────────────────────────
if "experience_years" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    for sl, grp in df.groupby("shortlisted"):
        ax.hist(grp["experience_years"], bins=15, alpha=0.6,
                label=f"Shortlisted={sl}", edgecolor="white")
    ax.set_title("Years of Experience by Shortlisting Outcome", fontsize=13, fontweight="bold")
    ax.set_xlabel("experience_years")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/eda_06_experience_shortlisted.png", dpi=150)
    plt.close()
    print("[SAVED] outputs/eda_06_experience_shortlisted.png")

print("\n" + "=" * 60)
print("  EDA COMPLETE -- All plots saved to outputs/")
print("  NEXT: python 02_preprocess.py")
print("=" * 60 + "\n")
