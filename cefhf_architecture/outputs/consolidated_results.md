# CEFHF -- Consolidated Results

## T1_dataset_summary
| dataset     |     n |   n_features | protected   |   base_rate |   n_numeric |   n_categorical |
|:------------|------:|-------------:|:------------|------------:|------------:|----------------:|
| synthetic   | 20000 |           11 | gender      |       0.483 |           9 |               2 |
| adult       | 30162 |           12 | sex         |       0.249 |           5 |               7 |
| recruitment |  2000 |            4 | gender      |       0.403 |           3 |               1 |

## T2_main_results
| dataset     | model                |   ROC-AUC |      F1 |   SPD |     EOD |   CFVR-flip |   CFVR-SCM |   DF-epsilon |
|:------------|:---------------------|----------:|--------:|------:|--------:|------------:|-----------:|-------------:|
| synthetic   | LogisticRegression   |     0.877 |   0.781 | 0.074 |   0.035 |       0     |      0.382 |        1.278 |
| synthetic   | RandomForest         |     0.866 |   0.773 | 0.092 |   0.066 |       0     |      0.242 |        1.471 |
| synthetic   | XGBoost              |     0.866 |   0.77  | 0.073 |   0.038 |       0.032 |      0.295 |        1.386 |
| synthetic   | XGBoost+RW           |     0.866 |   0.766 | 0.039 |   0.002 |       0.122 |      0.293 |        1.352 |
| adult       | LogisticRegression   |     0.909 |   0.682 | 0.18  |   0.071 |       0.393 |      0.434 |        3.817 |
| adult       | RandomForest         |     0.891 |   0.674 | 0.188 |   0.085 |       0.14  |      0.459 |        4.62  |
| adult       | XGBoost              |     0.929 |   0.723 | 0.182 |   0.067 |       0.043 |      0.304 |        4.573 |
| adult       | XGBoost+RW           |     0.924 |   0.702 | 0.103 |   0.16  |       0.011 |      0.29  |        4.444 |
| recruitment | LogisticRegression   |     0.489 |   0     | 0     |   0     |       0     |    nan     |        0     |
| recruitment | RandomForest         |     0.453 |   0.307 | 0.02  |   0.063 |       0.67  |    nan     |        0.791 |
| recruitment | XGBoost              |     0.457 |   0.313 | 0.03  |   0.074 |       0.245 |    nan     |        0.734 |
| recruitment | XGBoost+RW           |     0.466 |   0.34  | 0     |   0.107 |       0.302 |    nan     |        1.147 |
| synthetic   | CEFHF (CF-aug lam=1) |     0.868 | nan     | 0.116 | nan     |     nan     |      0.101 |      nan     |
| adult       | CEFHF (CF-aug lam=1) |     0.927 | nan     | 0.198 | nan     |     nan     |      0.017 |      nan     |

## T3_ablation
|      AUC |       SPD |   CFVR | dataset   | variant     | note                                                           |
|---------:|----------:|-------:|:----------|:------------|:---------------------------------------------------------------|
| 0.789927 | 0.0705869 | 0      | synthetic | Full CEFHF  | nan                                                            |
| 0.86977  | 0.116061  | 0.1    | synthetic | drop L1     | nan                                                            |
| 0.86977  | 0.116061  | 0.0584 | synthetic | drop L2/SCM | nan                                                            |
| 0.86807  | 0.0681148 | 0.2932 | synthetic | drop L4     | nan                                                            |
| 0.86977  | 0.116061  | 0.1    | synthetic | drop L5     | L5 affects explanations, not model metrics; see layer5 outputs |
| 0.923232 | 0.210585  | 0.0152 | adult     | Full CEFHF  | nan                                                            |
| 0.926854 | 0.202813  | 0.0168 | adult     | drop L1     | nan                                                            |
| 0.926854 | 0.202813  | 0.0536 | adult     | drop L2/SCM | nan                                                            |
| 0.929979 | 0.187767  | 0.3012 | adult     | drop L4     | nan                                                            |
| 0.926854 | 0.202813  | 0.0168 | adult     | drop L5     | L5 affects explanations, not model metrics; see layer5 outputs |

## T4_explanation_quality
| dataset   |   shap_stability_tau |   mean_PS_top_actions |   n_boot |
|:----------|---------------------:|----------------------:|---------:|
| synthetic |                    1 |                 0.343 |       10 |
| adult     |                    1 |                 0.097 |       10 |

## T5_runtime
| layer                                   | runtime_s   |
|:----------------------------------------|:------------|
| L1 proxy/D_fair                         | ~5          |
| L2 DAG+SCM                              | ~8          |
| L3 metrics                              | ~30         |
| L4 fair training                        | ~60         |
| L5 explainability                       | ~40         |
| L6 router                               | ~15         |
| analyses (gamma/ablation/shap-vs-cefhf) | ~60         |

## T6_llm_audit
| note                                                                                                                |
|:--------------------------------------------------------------------------------------------------------------------|
| Skipped this pass (causal-architecture focus). Layer-6 router simulation provided instead; see escalation_curve.png |

## Figures
- F1 lambda-sweep Pareto: lambda_sweep_pareto.png (H2)
- F2 gamma-sensitivity: gamma_sensitivity.png (H1)
- Layer-3 CFVR-flip vs CFVR-SCM: layer3_cfvr_flip_vs_scm.png
- SHAP vs CEFHF: shap_vs_cefhf.png
- DAGs: dag_synthetic.png, dag_adult.png
- Escalation curve: escalation_curve.png
