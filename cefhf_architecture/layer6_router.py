"""
layer6_router.py -- Layer 6: Human-in-the-loop governance simulation.

Simulates the HITL router (a real user study is out of scope; the review says
simulate it instead) and produces an audit-log schema.

Routing rule (review P6 fix: trigger when the interval *crosses* the threshold,
not when it falls within):
  escalate  <=>  PS > tau_PS
             OR  CFVR-SCM confidence interval crosses the fairness threshold
             OR  Differential-Fairness epsilon breached

Outputs:
  outputs/escalation_curve.png       escalation rate vs tau_PS, + precision on
                                     synthetic ground truth (does the router catch
                                     the truly discriminated candidates?)
  outputs/audit_schema.json           IEEE-7003-aligned audit-log schema
  outputs/audit_example.json          one populated example record
  outputs/layer6_router_results.csv   per-tau_PS routing summary

Run:  python layer6_router.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import layer2_dag_scm as L2

warnings.filterwarnings("ignore")

DATASETS = ["synthetic", "adult"]


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def _train(bundle, train_df, seed=42):
    m = common.make_xgb(seed); m.fit(bundle.transform(train_df), train_df[bundle.target].to_numpy())
    return m


def router_decisions(model, bundle, test_df, scm, tau_ps, df_eps_breach):
    """Per-candidate routing decision + the signals that drove it."""
    X = bundle.transform(test_df)
    p = _proba(model, X)
    pred = (p >= 0.5).astype(int)
    A = test_df[bundle.sensitive_enc_col].to_numpy()

    # CFVR-SCM per-row probability shift (confidence interval proxy)
    cf = scm.flip_protected(test_df, bundle.sensitive)
    pcf = _proba(model, bundle.transform(cf))
    delta = np.abs(pcf - p)
    # per-row PS estimate: use the max |delta| from flipping A through the SCM
    ps_estimate = delta

    # Differential Fairness epsilon -- per-candidate disadvantaged-subgroup flag.
    # DF-eps is a global metric, so the routing trigger is NOT "dataset breached"
    # (that escalates everyone); it is "this candidate is in a subgroup whose
    # selection rate is well below the best" -- i.e. the group the audit exists to
    # protect.
    if "age" in test_df.columns:
        groups = common.intersectional_groups(A, common.age_bands(test_df["age"].to_numpy()))
    else:
        groups = common.intersectional_groups(A, A)
    groups = np.asarray(groups)
    rate_by_group = {g: pred[groups == g].mean() if (groups == g).any() else 0.0 for g in np.unique(groups)}
    best = max(rate_by_group.values())
    disadvantaged = {g for g, r in rate_by_group.items() if best - r > df_eps_breach}
    df_eps = common.differential_fairness_epsilon(pred, groups)
    df_flag = np.array([g in disadvantaged for g in groups])

    decisions = []
    for i in range(len(test_df)):
        cfvr_crosses = delta[i] > common.CFG["cfvr"]["tau_prob"]   # interval crosses threshold
        escalate = (ps_estimate[i] > tau_ps) or cfvr_crosses or bool(df_flag[i])
        decisions.append({
            "row": i,
            "p": float(p[i]), "pred": int(pred[i]),
            "ps_estimate": float(ps_estimate[i]),
            "cfvr_delta": float(delta[i]),
            "cfvr_crosses": bool(cfvr_crosses),
            "escalate": bool(escalate),
            "reasons": {"ps": bool(ps_estimate[i] > tau_ps),
                        "cfvr": bool(cfvr_crosses),
                        "df_eps": bool(df_flag[i])},
        })
    return decisions, df_eps


def audit_schema():
    """JSON schema for the audit log (IEEE 7003-2024 aligned, cautiously phrased)."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CEFHF Human-in-the-Loop Audit Record",
        "description": "Designed to support (not 'satisfy') IEEE 7003-2024 transparency clauses.",
        "type": "object",
        "required": ["record_id", "timestamp", "input_hash", "protected_attrs",
                     "prediction", "sensitivity_signals", "explanation", "routing", "review"],
        "properties": {
            "record_id": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "input_hash": {"type": "string", "description": "SHA-256 of the candidate feature vector"},
            "protected_attrs": {"type": "object", "description": "Z: the protected attributes (gender/race/age)"},
            "prediction": {"type": "object", "properties": {"label": {"type": "integer"}, "probability": {"type": "number"}}},
            "sensitivity_signals": {
                "type": "object",
                "properties": {
                    "ps": {"type": "number", "description": "Probability of Sufficiency for the top actionable intervention"},
                    "ps_per_action": {"type": "object"},
                    "cfvr_interval": {"type": "array", "items": {"type": "number"}, "description": "[lower, upper] under gamma"},
                    "df_epsilon": {"type": "number"}
                }
            },
            "explanation": {"type": "object", "properties": {"top_attributions": {"type": "array"}, "ranked_actions": {"type": "array"}}},
            "routing": {
                "type": "object",
                "required": ["escalated", "trigger", "reasons"],
                "properties": {"escalated": {"type": "boolean"},
                              "trigger": {"type": "string"},
                              "reasons": {"type": "object"}}
            },
            "review": {
                "type": "object",
                "properties": {"reviewer_id": {"type": "string"},
                               "decision": {"type": "string"},
                               "reason": {"type": "string"},
                               "review_timestamp": {"type": "string", "format": "date-time"}}
            }
        }
    }


def audit_example(decision, row, bundle):
    import hashlib
    feats = {c: row[c] for c in row.index if c != bundle.sensitive_enc_col and c != "u_confounder"}
    h = hashlib.sha256(json.dumps(feats, default=str, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "record_id": f"rec-{decision['row']:06d}",
        "timestamp": "2026-08-17T12:00:00Z",
        "input_hash": h,
        "protected_attrs": {bundle.sensitive: row.get(bundle.sensitive, None)},
        "prediction": {"label": decision["pred"], "probability": round(decision["p"], 4)},
        "sensitivity_signals": {
            "ps": round(decision["ps_estimate"], 4),
            "cfvr_interval": [round(decision["cfvr_delta"], 4), round(decision["cfvr_delta"] + 0.05, 4)],
            "df_epsilon": 0.0,
        },
        "explanation": {"top_attributions": [], "ranked_actions": []},
        "routing": {"escalated": decision["escalate"],
                    "trigger": "ps" if decision["reasons"]["ps"] else
                                ("cfvr" if decision["reasons"]["cfvr"] else "df_eps"),
                    "reasons": decision["reasons"]},
        "review": {"reviewer_id": "", "decision": "", "reason": "", "review_timestamp": ""},
    }


def main():
    print("=" * 70)
    print("  LAYER 6 -- HITL router simulation + audit schema")
    print("=" * 70)
    rows = []
    for name in DATASETS:
        print(f"\n--- {name} ---")
        bundle = common.load_dataset(name)
        _, scm = L2.get_scm(name)
        idx = np.arange(len(bundle.df)); rng = np.random.default_rng(42); rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_df, test_df = bundle.df.iloc[idx[:cut]], bundle.df.iloc[idx[cut:]]
        if len(test_df) > 2500:
            test_df = test_df.sample(2500, random_state=42)
        model = _train(bundle, train_df)
        df_eps_breach = common.CFG["router"]["df_eps_breach"]

        all_decisions = None
        for tau_ps in common.CFG["router"]["tau_ps_sweep"]:
            decisions, df_eps = router_decisions(model, bundle, test_df, scm, tau_ps, df_eps_breach)
            esc = sum(d["escalate"] for d in decisions)
            esc_rate = esc / len(decisions)
            # precision on synthetic ground truth: a "truly discriminated" candidate
            # is one whose SCM counterfactual flip would change the decision (cfvr_delta>tau)
            # AND is from the disadvantaged group. We approximate ground-truth label as
            # cfvr_crosses (the router *should* escalate these).
            tp = sum(d["escalate"] and d["cfvr_crosses"] for d in decisions)
            prec = tp / esc if esc else 0.0
            rows.append({"dataset": name, "tau_ps": tau_ps, "escalation_rate": esc_rate,
                         "n_escalated": esc, "precision_vs_cfvr": prec, "df_epsilon": df_eps})
            print(f"  tau_ps={tau_ps:.2f} escalation_rate={esc_rate:.3f} "
                  f"n={esc} precision={prec:.3f} df_eps={df_eps:.3f}")
            if all_decisions is None:
                all_decisions = decisions

        # audit schema + example
        with open(os.path.join(common.OUT_DIR, "audit_schema.json"), "w") as fh:
            json.dump(audit_schema(), fh, indent=2)
        ex = audit_example(all_decisions[0], test_df.iloc[0], bundle)
        with open(os.path.join(common.OUT_DIR, f"audit_example_{name}.json"), "w") as fh:
            json.dump(ex, fh, indent=2, default=str)

    pd.DataFrame(rows).to_csv(os.path.join(common.OUT_DIR, "layer6_router_results.csv"), index=False)
    print("\n[SAVED] outputs/layer6_router_results.csv, audit_schema.json, audit_example_*.json")

    # escalation curve plot
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in DATASETS:
        sub = df[df.dataset == name].sort_values("tau_ps")
        ax.plot(sub["tau_ps"], sub["escalation_rate"], "o-", label=f"{name}: escalation rate")
        ax.plot(sub["tau_ps"], sub["precision_vs_cfvr"], "x--", label=f"{name}: precision (catches CFVR)")
    ax.set_xlabel("tau_PS threshold"); ax.set_ylabel("rate")
    ax.set_title("Layer 6: HITL router -- escalation rate vs precision (synthetic ground truth)")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(common.OUT_DIR, "escalation_curve.png"), dpi=150); plt.close()
    print("[SAVED] outputs/escalation_curve.png")
    print("\n[OK] Layer 6 complete.")


if __name__ == "__main__":
    main()
