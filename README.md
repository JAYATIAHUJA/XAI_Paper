# XAI_Paper: ML Fairness and Counterfactual Sensitivity Audit Pipeline

This repository contains the codebase and implementation for a research paper on machine learning fairness, explainable AI (XAI), and counterfactual sensitivity auditing.

## Key Features

1. **Bias Detection**: Pre-training Exploratory Data Analysis (EDA) checking for disparities in selection rates across demographic groups.
2. **Standard Group Fairness Metrics**: Evaluates models using demographic parity difference (SPD), demographic parity ratio (DIR), and equalized odds difference (EOD).
3. **Novel Metric - CFVR**: Computes the **Counterfactual Violation Rate (CFVR)** to measure how model predictions change under counterfactual input perturbations (e.g., flipping gender/sex).
   - **CFVR-Prob**: Measure of absolute probability shift ($|\Delta P| > 0.10$).
   - **CFVR-Class**: Measure of binary classification label flip.
4. **Bias Mitigation**: Implements pre-processing mitigation via **AIF360 Reweighing** (Kamiran & Calders, 2012) to rebalance sample weights and train fairer classifiers.
5. **Cross-Validation & Tuning**: Evaluates model performance and fairness robustness using 5-Fold Stratified Cross Validation and Randomized Hyperparameter Tuning (XGBoost).
6. **Generalization Benchmark**: Runs validation on the **Adult UCI** dataset as an external domain benchmark.

## Pipeline Structure

- `00_download_dataset.py`: Setup script to download target datasets.
- `01_eda_bias_check.py`: Exploratory analysis and selection rate disparity check.
- `02_preprocess.py`: Feature scaling, encoding, and train-test splitting.
- `03_train_baseline.py`: Trains Logistic Regression, Random Forest, and XGBoost baselines.
- `04_shap_analysis.py`: SHAP feature importances and local explanations.
- `05_fairness_metrics.py`: Standard fairness evaluations (SPD, DIR, EOD).
- `06_cfvr.py`: Novel CFVR-Prob and CFVR-Class metric evaluation.
- `07_debiasing_reweighing.py`: AIF360 Reweighing mitigation.
- `08_final_table.py`: Generates the consolidated comparison tables.
- `09_adult_uci_validation.py`: Generalization study on the Adult UCI dataset.
- `10_bias_vs_instability.py`: Analysis to separate model instability from systematic group bias.
- `11_improved_training.py`: Initial cross-validated and tuned training script.
- `run_all.py`: Master script to run baseline pipeline steps.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full baseline pipeline:
   ```bash
   python run_all.py
   ```
