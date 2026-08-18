# Leakage-free publication experiment summary

Run design: 5-fold stratified cross-validation repeated three times (15 held-out
folds), seed 42. All preprocessing, reweighing, model fitting, and synthetic SCM
fitting occurs inside the training fold. Values below are fold means; full 95%
bootstrap confidence intervals are in `publication_main_results.csv` and
`publication_causal_results.csv`.

## Main real/synthetic benchmark

| Dataset | Model | ROC-AUC | PR-AUC | SPD | EOD |
|---|---:|---:|---:|---:|---:|
| Synthetic | Logistic Regression | 0.877 | 0.870 | 0.067 | 0.045 |
| Synthetic | XGBoost | 0.874 | 0.866 | 0.060 | 0.040 |
| Synthetic | XGBoost + reweighing | 0.874 | 0.866 | 0.039 | 0.027 |
| Adult | Logistic Regression | 0.905 | 0.771 | 0.184 | 0.111 |
| Adult | XGBoost | 0.921 | 0.821 | 0.178 | 0.107 |
| Adult | XGBoost + reweighing | 0.916 | 0.807 | 0.082 | 0.169 |
| ACS Employment (CA 2018) | Logistic Regression | 0.844 | 0.769 | 0.014 | 0.021 |
| ACS Employment (CA 2018) | XGBoost | 0.897 | 0.851 | 0.017 | 0.026 |
| ACS Employment (CA 2018) | XGBoost + reweighing | 0.897 | 0.851 | 0.018 | 0.026 |

Reweighing is not uniformly beneficial: it strongly reduces Adult SPD but
worsens Adult EOD. This is a metric trade-off and must not be described as
"ensuring fairness." On ACS, reweighing changes little because the White versus
Non-White group disparities under these metrics are already small.

## Synthetic causal evaluation

| Method | ROC-AUC | SPD | EOD | CF probability violation | CF class flip |
|---|---:|---:|---:|---:|---:|
| Unconstrained XGBoost | 0.863 | 0.083 | 0.069 | 0.143 | 0.073 |
| Counterfactual augmentation | 0.863 | 0.033 | 0.029 | 0.001 | 0.022 |

On the known synthetic SCM, counterfactual augmentation reduces the probability
violation rate from 14.3% to 0.14% with essentially unchanged ROC-AUC. This claim
is limited to the specified synthetic data-generating process; Adult and ACS do
not provide known causal ground truth.

## Dataset-quality notes

- Synthetic: 20,000 rows, no missing or duplicate rows.
- Adult: 30,162 complete-case rows after the documented missing-value removal;
  0.076% feature-identical rows.
- ACS Employment: reproducible 30,000-row sample from California 2018 1-Year ACS
  PUMS; 16.7% feature-identical rows. These are not deduplicated because the
  benchmark features contain no respondent identifier and equal feature vectors
  can represent different survey respondents.
