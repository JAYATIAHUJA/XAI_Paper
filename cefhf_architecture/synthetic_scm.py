"""
synthetic_scm.py -- Layer-2 ground truth: a synthetic hiring SCM with a KNOWN DAG.

Why this exists (review P1 + Section 4.1)
-----------------------------------------
No public *hiring* dataset exposes the true causal structure, so no fairness
metric computed on one can be validated.  We therefore generate a hiring dataset
whose data-generating process we control.  This is the only dataset on which we
can prove:
  * Layer 2 recovers the right DAG / edge labels,
  * CFVR-SCM (counterfactual via the SCM) differs from CFVR-flip (A-only),
  * gamma-sensitivity bounds are honest (we know the true confounder strength),
  * CEFHF's causal explanations beat correlational SHAP/LIME.

DAG (edges tagged for evaluation)
  gender(A) ──discriminatory──> gap_years, negotiation, screening_score
  race       ──proxy────────> zip ──proxy──> university_quality
  ses        ──legitimate────> university_quality ──legitimate──> skills
  skills     ──legitimate────> screening_score, Y
  experience ──legitimate────> screening_score, Y
  U (unobserved) ──confounded──> gender, Y          (Gamma controls its strength)

The deliberate `gender -> screening_score` edge is the in-scoring bias we want
detected and removed; `race -> zip -> university_quality` is the residential
proxy path; `U` is the confounder the Gamma-sweep probes.

Run:  python synthetic_scm.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import networkx as nx

import common

SEED = common.CFG["seeds"]["synthetic"]
N = 20_000
GAMMA = 1.0                       # baseline confounding strength

DATA_PATH = os.path.join(common.DATA_DIR, "synthetic_hiring.csv")
DAG_JSON = os.path.join(common.DATA_DIR, "synthetic_dag.json")
DAG_GML = os.path.join(common.DATA_DIR, "synthetic_dag.graphml")

# Edge label taxonomy
LEGIT = "legitimate"
PROXY = "proxy"
DISCR = "discriminatory"
CONFD = "confounded"
STRUCT = "structural"            # downstream of A but not a direct scoring bias

# (source, target, label)
EDGES = [
    ("gender", "gap_years", DISCR),
    ("gender", "negotiation", DISCR),
    ("gender", "screening_score", DISCR),
    ("race", "zip", PROXY),
    ("zip", "university_quality", PROXY),
    ("ses", "university_quality", LEGIT),
    ("university_quality", "skills", LEGIT),
    ("skills", "screening_score", LEGIT),
    ("experience_years", "screening_score", LEGIT),
    ("skills", "shortlisted", LEGIT),
    ("experience_years", "shortlisted", LEGIT),
    ("screening_score", "shortlisted", STRUCT),
    ("gap_years", "shortlisted", STRUCT),
    ("negotiation", "shortlisted", STRUCT),
    ("education_level", "shortlisted", LEGIT),
    ("u_confounder", "gender", CONFD),
    ("u_confounder", "shortlisted", CONFD),
]


def build_ground_truth_dag() -> nx.DiGraph:
    G = nx.DiGraph()
    for s, t, lab in EDGES:
        G.add_edge(s, t, label=lab)
    return G


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = N, gamma: float = GAMMA, seed: int = SEED) -> pd.DataFrame:
    """Generate `n` hiring candidates from the SCM.

    `gamma` scales the strength of the unobserved confounder U on both the
    protected attribute A and the outcome Y.  gamma=1 is the baseline;
    larger gamma => stronger confounding => wider counterfactual-interval
    bounds (probed by gamma_sensitivity.py).
    """
    rng = np.random.default_rng(seed)

    # --- exogenous / confounder ------------------------------------------------
    U = rng.normal(0, 1, n)                                  # unobserved confounder

    # --- protected attributes --------------------------------------------------
    # gender (1 = female) influenced by U (confounding)
    p_female = _sigmoid(0.4 * gamma * U)
    gender = rng.binomial(1, p_female)                      # 0 male, 1 female
    # race minority (0/1); independent of U here, drives the proxy path
    race = rng.binomial(1, 0.30, n)

    # --- legitimate drivers ----------------------------------------------------
    ses = rng.normal(0, 1, n)                               # socioeconomic status
    age = rng.integers(22, 56, n)                           # 22..55

    # --- structural mediators of A (discriminatory path) -----------------------
    gap_years = np.clip(0.6 * gender + rng.normal(0, 0.5, n), 0, None)      # women: more career gaps
    negotiation = -0.5 * gender + rng.normal(0, 0.7, n)                      # women: negotiate less

    # --- proxy path: race -> zip -> university_quality -------------------------
    # ZIP encodes race (residential segregation). Higher ZIP => minority =>
    # LOWER university quality (the proxy-discrimination mechanism we want flagged).
    zip_code = 1.0 * race + rng.normal(0, 0.3, n)
    university_quality = (0.6 * ses - 0.6 * zip_code       # SES up, segregation down
                          + rng.normal(0, 0.5, n))
    # standardise for stable coefficients
    university_quality = (university_quality - university_quality.mean()) / university_quality.std()

    # --- legitimate mediators -------------------------------------------------
    skills = 0.7 * university_quality + rng.normal(0, 0.5, n)
    skills = (skills - skills.mean()) / skills.std()
    experience_years = np.clip(age - 22 - gap_years, 0, None).astype(float)

    # education level derived from ses + university_quality (legitimate)
    edu_score = 0.5 * ses + 0.6 * university_quality + rng.normal(0, 0.5, n)
    edu_bins = np.quantile(edu_score, [0.25, 0.55, 0.80])
    education_level = np.where(edu_score < edu_bins[0], "HighSchool",
                      np.where(edu_score < edu_bins[1], "Bachelors",
                      np.where(edu_score < edu_bins[2], "Masters", "PhD")))

    # --- screening score: legitimate drivers + DIRECT gender bias --------------
    # skills, experience legitimate; the +(-0.35*gender) term is the
    # in-scoring discrimination (the bias Layer 3 must flag).
    score = (0.6 * skills
             + 0.4 * (experience_years / 10.0)
             - 0.35 * gender                       # <-- discriminatory direct edge
             + rng.normal(0, 0.5, n))
    screening_score = (score - score.mean()) / score.std()

    # --- outcome Y: shortlisted ------------------------------------------------
    # legitimate drivers + contaminated screening_score + structural mediators
    # + confounder U (scaled by gamma).
    logit = (-1.2
             + 1.1 * skills
             + 0.8 * screening_score               # carries gender bias downstream
             + 0.5 * (experience_years / 10.0)
             + 0.3 * education_level_to_num(education_level)
             - 0.2 * gap_years
             + 0.2 * negotiation
             + 0.5 * gamma * U)                    # <-- confounder -> Y
    p_y = _sigmoid(logit)
    shortlisted = rng.binomial(1, p_y)

    df = pd.DataFrame({
        "gender": np.where(gender == 1, "Female", "Male"),
        "race": np.where(race == 1, "Minority", "Majority"),
        "age": age,
        "ses": ses,
        "gap_years": gap_years,
        "negotiation": negotiation,
        "zip": np.round(zip_code, 3),
        "university_quality": university_quality,
        "skills": skills,
        "experience_years": experience_years,
        "education_level": education_level,
        "screening_score": screening_score,
        "u_confounder": U,                         # kept ONLY for evaluation/gamma-sweep
        "shortlisted": shortlisted,
    })
    return df


def education_level_to_num(arr):
    mapping = {"HighSchool": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    return np.vectorize(mapping.get)(arr).astype(float)


def save_ground_truth_dag():
    G = build_ground_truth_dag()
    nx.write_graphml(G, DAG_GML)
    edges = [{"source": s, "target": t, "label": d["label"]} for s, t, d in G.edges(data=True)]
    nodes = list(G.nodes())
    with open(DAG_JSON, "w") as fh:
        json.dump({"nodes": nodes, "edges": edges,
                   "protected": ["gender", "race"],
                   "target": "shortlisted",
                   "confounder": "u_confounder"}, fh, indent=2)
    return G


def main():
    print(f"Generating {N} synthetic hiring candidates (gamma={GAMMA}, seed={SEED})...")
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"[SAVED] {DATA_PATH}  shape={df.shape}")

    G = save_ground_truth_dag()
    print(f"[SAVED] {DAG_JSON}")
    print(f"[SAVED] {DAG_GML}")
    print(f"\nGround-truth DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    by_label = {}
    for _, _, d in G.edges(data=True):
        by_label[d["label"]] = by_label.get(d["label"], 0) + 1
    for k, v in by_label.items():
        print(f"  {k:14s}: {v} edges")

    # quick signal check: a model must be able to predict Y (unlike recruitment)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    feat = ["age", "ses", "gap_years", "negotiation", "zip", "university_quality",
            "skills", "experience_years", "screening_score"]
    Xtr, Xte, ytr, yte = train_test_split(df[feat], df["shortlisted"], test_size=0.2,
                                          random_state=42, stratify=df["shortlisted"])
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    auc = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    print(f"\nSignal check: LogisticRegression ROC-AUC = {auc:.3f}  (recruitment dataset was ~0.50)")
    print(f"Shortlist rate by gender:  Male={df[df.gender=='Male'].shortlisted.mean():.3f}  "
          f"Female={df[df.gender=='Female'].shortlisted.mean():.3f}")
    print(f"Shortlist rate by race:    Majority={df[df.race=='Majority'].shortlisted.mean():.3f}  "
          f"Minority={df[df.race=='Minority'].shortlisted.mean():.3f}")


if __name__ == "__main__":
    main()
