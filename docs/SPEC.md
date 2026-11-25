# StrokeSense: implementation specification

**Title:** *Does AI know more than your birthday? A leakage-tested study of stroke prediction.*
**Thesis:** On this dataset most apparent ML gain over age came from how missing BMI was filled.

## Locked decisions
| Topic | Decision | Evidence |
|---|---|---|
| Split | Stratified 80/20, seed 42 (same rows as the ChatGPT bundle) | 4,088 / 1,022 rows; 199 / 50 strokes |
| Rows | Keep all 5,110 | no duplicates; glucose is bimodal, not erroneous |
| BMI | Removed from the final model; kept for EDA and the leak experiment | PR-AUC of pairwise boosting 0.214 -> 0.174 without it |
| Imbalance | No SMOTE, no reweighting; move the threshold | SMOTE: mean risk 31% vs 4.9% real |
| Frozen model | L2 logistic, 4 features (age, glucose, hypertension, heart disease) | PR-AUC 0.190; age-matched AUC gain +0.061 [0.015, 0.104]; nested selection gives PR-AUC 0.174, so 0.190 is optimistic; PR-AUC gain over age not established |
| Companion | Integer scorecard, refitted without BMI | BMI-free PR-AUC 0.165; communication aid only |
| Reference models | Age-only, Balanced RF, XGBoost, CatBoost, HGB, EBM | BMI-free: EBM monotone 0.182, EBM 0.179, CatBoost 0.169 |
| Threshold | Out-of-fold predictions, never the test split | |
| Final test | Scored once, 2026-11-25, after app approval | `artifacts/results/test_results.json` |
| Metrics | PR-AUC, age-matched AUC, Brier/calibration, recall and precision at threshold, net benefit; ROC-AUC secondary | |

## Distinctive components
1. Missing-data artifact test (app page: Leak Detective): median vs random draw vs removal. Done.
2. Age-matched AUC. Done.
3. Confounding plot: "ever married" OR 4.14 -> 0.67 after age. Done.
4. Pocket scorecard: whole points added by hand. Prototype done.
5. Decision curve. Done on out-of-fold predictions.
6. Conformal prediction: considered, then dropped. Too few strokes for honest class coverage.
7. Feature stability: selection frequency across 200 bootstraps. Done.

## App (Streamlit)
Overview | Patient simulator (model + scorecard + age-only side by side) | Leak detective | Research lab (benchmarks, disagreement) | Threshold studio | Drift observatory.
Every result page: "Educational prototype. Not a diagnosis." No causal wording in what-if.

## Docker and monitoring: what each part is for
- **Dockerfile**: the instructor runs one command with the exact same versions.
- **compose `app`**: Streamlit with its built-in `/_stcore/health` healthcheck and restart policy.
- **compose `monitor`**: scheduled job reading the prediction log; writes PSI / KS / chi-square drift results.
- **JSON logs**: timestamp, model version, latency, risk band. Never raw inputs.
- **Monitor signal**: PSI of logged risk bands against development predictions. Inputs are never logged, so input drift is only demonstrated on synthetic batches.
- **MLflow (SQLite)**: training runs, local only, not in the app image.
- **Evidently**: dropped; PSI/KS/chi-square already cover it and the image stays small.
- **GitHub Actions**: tests, then Docker build, on every push.

## Order of work
| Day | Work | Done when |
|---|---|---|
| 1 | Audit, benchmark, leak test (done); All done: model frozen in `models/`. Test split scored once, after the app threshold and conformal decisions are final | models/metadata.json |
| 2 | Streamlit app, tests, logging | all pages work from a saved model |
| 3 | Docker, MLflow, drift, CI, GitHub, hosting, report | clean clone runs with one command |

Rubric: analysis = nb 1-3; features = nb 3, 5 and stability; training = nb 4-5; evaluation = final test section; report = 3-5 page PDF; bonuses = README/requirements, GitHub + hosting.
