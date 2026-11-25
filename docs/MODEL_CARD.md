# Model card: StrokeSense logistic regression

**Version** `1b83916f`, frozen 25 November 2026. L2 logistic regression (C = 0.1) on standardised age, average glucose, hypertension, heart disease. Threshold 5.175%.

**Task.** Binary classification of the recorded stroke label (0/1). Output: probability of class 1, and a class at the threshold. Not a prospective risk.

**Intended use.** Teaching and research on evaluation, leakage and fairness. **Not for** diagnosis, triage, screening, treatment or reassurance.

**Data.** Supplied by the university (original source not verified), 5,110 people, 249 strokes (4.87%). Stratified 80/20 split, seed 42. How it was collected, and the time between inputs and stroke, are unknown: the model describes association in this dataset, not future risk.

**Evaluation.** Development: 4×5-fold cross-validation with thresholds chosen inside training folds. Test: scored once.

| Metric | Development | Test (95% interval) | Age only, same test |
|---|---|---|---|
| PR-AUC | 0.190 | 0.259 (0.168–0.372) | 0.204 (0.140–0.297) |
| PR-AUC, selection repeated inside folds | 0.174 | – | – |
| ROC-AUC | 0.842 | 0.840 (0.777–0.893) | 0.834 (0.772–0.887) |
| Brier score | 0.043 | 0.041 (0.031–0.052) | 0.042 (0.032–0.053) |
| Sensitivity | 0.793 | 0.800 (0.670–0.888) | 0.840 (0.715–0.917) |
| Specificity | 0.733 | 0.740 (0.711–0.766) | 0.719 (0.690–0.746) |
| Precision | 0.132 | 0.137 (0.102–0.181) | 0.133 (0.100–0.175) |
| F1 | 0.226 | 0.233 | 0.230 |
| False alarms / missed strokes | 1036.5 / 41.2 (of 4,088) | 253 / 10 (of 1,022) | 273 / 8 |

Test PR-AUC difference from age alone: +0.055 (95% interval -0.012 to +0.130) (not established).

**Subgroups, test.** Strokes caught by age: 40 or under: 0 of 5; 41–60: 2 of 7; 61–70: 7 of 7; 71+: 31 of 31. Women and men, people with and without BMI: counts in `artifacts/results/test_results.json`. Most groups have too few strokes for firm conclusions.

**Why no BMI.** Blank BMI is far more common among strokes (18.6% vs 3.4% of development records). Models used the blanks as a signal. We do not know why they are blank, so we cannot assume the signal exists at prediction time. Group-level cost: among 170 development records with no BMI the model expected 14 strokes in total and 37 occurred. This is not a statement about any individual.

**Limits.** Threshold works almost as an age cut-off. The four features were chosen after exploring development data, so 0.190 is probably optimistic (0.174 with selection repeated inside folds). Early integrity and target summaries covered all rows, including test labels; no model used them. 50 test strokes give wide intervals. No external validation. The app accepts every training age (slider 1–82). Under 18, development data has 1 stroke in 674 people, so output there cannot be checked.

Full table with reasons and needed data: [LIMITATIONS.md](LIMITATIONS.md). Reporting self-check: [TRIPOD_AI.md](TRIPOD_AI.md).

**Robustness (development, after the freeze).** Heart disease gives no consistent improvement. Smoking stays unselected under every treatment of Unknown (at most 1 of 100 selections). Repeating selection and training in 500 bootstraps gives a same-age ranking gain of +0.042 (-0.019 to +0.101).

**Privacy.** Logs hold time, model version, coarse risk band and latency. Never inputs or exact risk.
