"""Fill README/model card tables and build reports/report.html from saved artifacts.  python -m src.build_report
PDF: msedge --headless --no-pdf-header-footer --print-to-pdf=reports/StrokeSense_Report.pdf file:///.../reports/report.html"""
import json

import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts/results"
T = json.loads((RES / "test_results.json").read_text(encoding="utf-8"))
R = pd.read_csv(RES / "FINAL_RESULTS.csv", index_col=0)
S = pd.read_csv(RES / "subgroups_verified.csv").set_index("group")
L = pd.read_csv(RES / "leak_cv.csv").pivot_table(index="label", columns="mode", values="pr_auc")
m, (a, b) = R.loc["Logistic 4 (selected)"], T["models"].values()
D = T["difference"]
iv = lambda q: f"{q['value']:.3f} ({max(0, q['ci'][0]):.3f}–{q['ci'][1]:.3f})"
rows = [(n, f"{m[c]:.3f}", iv(a[k]), iv(b[k])) for n, c, k in (("PR-AUC", "pr", "pr_auc"), ("ROC-AUC", "roc", "roc_auc"), ("Brier score", "brier", "brier"), ("Sensitivity", "sens", "sensitivity"), ("Specificity", "spec", "specificity"), ("Precision", "prec", "precision"))]
rows += [("F1", f"{m.f1:.3f}", f"{a['f1']:.3f}", f"{b['f1']:.3f}"), ("False alarms / missed strokes", f"{m.fp:.1f} / {199 - m.tp:.1f} (of 4,088)", f"{a['fp']} / {a['fn']} (of 1,022)", f"{b['fp']} / {b['fn']}")]
md = "\n".join("| " + " | ".join(r) + " |" for r in rows)
diff = f"{D['pr_auc']['value']:+.3f} (95% interval {D['pr_auc']['ci'][0]:+.3f} to {D['pr_auc']['ci'][1]:+.3f})"
age = "; ".join(f"{g['group']}: {g['caught']} of {g['strokes']}" for g in T["subgroups"][:4])


def table(head, body):
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</table>"


def main():
    for f in ("README.md", "docs/MODEL_CARD.md"):
        p = ROOT / f
        p.write_text(p.read_text(encoding="utf-8").replace("@@RESULTS@@", md).replace("@@DIFF@@", diff).replace("@@AGE@@", age), encoding="utf-8")
    fig = lambda n, c, w=100: f'<figure><img src="figures/{n}.png" style="width:{w}%"><figcaption>{c}</figcaption></figure>'
    bench = table(["Model", "PR-AUC", "ROC-AUC", "Sensitivity", "Specificity", "Precision", "PR gap (train − unseen)"], [(k.replace(" (selected)", ", selected"), f"{r.pr:.3f}", f"{r.roc:.3f}", f"{r.sens:.1%}", f"{r.spec:.1%}", f"{r.prec:.1%}", f"{r.gap_pr:+.3f}") for k, r in R.sort_values("pr", ascending=False).iterrows()])
    leak = table(["Model", "Blank = median", "Blank = random draw", "BMI removed"], [(n, *(f"{L.loc[k, c]:.3f}" for c in ("median", "draw", "drop"))) for k, n in (("HGB mono pairs", "Constrained boosting"), ("XGBoost", "XGBoost"), ("CatBoost", "CatBoost"), ("Logistic", "Logistic, 9 inputs"), ("Age only (logistic)", "Age only"))])
    sub = table(["Age", "Development strokes caught", "Test strokes caught", "Test people flagged"], [(g["group"], f"{S.caught[k]:.1f} of {int(S.strokes[k])}", f"{g['caught']} of {g['strokes']}", f"{g['flagged']} of {g['n']}") for k, g in zip(["0-40", "41-60", "61-70", "71+"], T["subgroups"])])
    nb, e = S.loc["No BMI recorded"], S.loc["BMI recorded"]
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>StrokeSense report</title><style>
@page{{size:A4;margin:17mm 17mm}}body{{font:10pt/1.5 'Segoe UI',Arial,sans-serif;color:#14222F}}h1{{font-size:23pt;line-height:1.1;margin:0 0 2mm;letter-spacing:-.02em}}h2{{font-size:13pt;margin:6mm 0 1.5mm;color:#0B6F65;break-after:avoid}}p{{margin:0 0 2.2mm}}
.by{{color:#465868;margin-bottom:4mm}}.key{{border-left:3px solid #0B8578;background:#F0F5F4;padding:3mm 4mm;margin:3mm 0}}table{{border-collapse:collapse;width:100%;font-size:8.6pt;margin:2mm 0 3mm;break-inside:avoid}}th{{text-align:left;border-bottom:1.4px solid #14222F;padding:1.2mm 2mm 1.2mm 0}}td{{border-bottom:.5px solid #C9D3DA;padding:1mm 2mm 1mm 0}}
figure{{margin:2mm 0 3mm;text-align:center;break-inside:avoid}}img{{max-height:72mm;object-fit:contain}}figcaption{{font-size:8.4pt;color:#465868;text-align:left}}ul{{margin:0 0 2mm;padding-left:5mm}}</style>
<h1>Does AI know more than your birthday?</h1><p class="by">StrokeSense AI: a leakage-tested study of stroke prediction. Academic project for Hanna Abi Akl, November 2026. Author: maryembrik.</p>
<div class="key"><b>Summary.</b> Age carries most of the signal in this dataset. The strongest first model gained its lead from how blank BMI values were filled. The final four-feature logistic model caught {a['tp']} of 50 strokes in a test set scored once, with {a['fp']} false alarms among 972 people without stroke. Its test PR-AUC differs from age alone by {diff}, so no advantage is established.</div>
<h2>1. Data analysis</h2><p>5,110 people, 249 strokes (4.87%). No duplicate rows; all rows kept. BMI is blank for 201 people. Accuracy is uninformative at this prevalence: predicting "no stroke" for everyone scores 95.1%, so PR-AUC is primary. Data were split 80/20, stratified, seed 42: 4,088 development (199 strokes) and 1,022 test (50). All choices used development data only. Early integrity and target summaries covered all rows, including test labels; no model used them.</p>{fig('rates', 'Figure 1. Stroke rates by variable.')}
<p>"Ever married" looks like a strong risk factor (odds ratio 4.14), but the sign reverses once age is included (0.67). It is mostly a proxy for age.</p>{fig('confounding', 'Figure 2. Unadjusted and age-adjusted odds ratios.', 72)}
<h2>2. The missing-BMI shortcut</h2><p>In development data BMI is blank for 37 of 199 people with stroke (18.6%) and 133 of 3,889 without (3.4%). Filling blanks with a single value lets a tree isolate them. Random plausible filling, or removing BMI, removes the gain (mean PR-AUC, fixed settings):</p>{leak}
<p>This is a missing-data artifact. It does not prove why values are missing, nor conventional target leakage. We removed BMI. The price: people with no BMI are under-predicted ({nb.oof_risk:.1%} expected, {nb.observed:.1%} observed).</p>
<h2>3. Feature selection</h2><p>Final features: age, average glucose, hypertension, heart disease. Repeating stability selection inside 20 outer folds chose age 20 times, glucose 13, hypertension 12, heart disease 3. Heart disease stays partly for clinical plausibility. Nested selection gives PR-AUC 0.174, against 0.190 for the four features chosen after exploration, so 0.190 is probably optimistic.</p>
<h2>4. Model training</h2><p>Ten models, 4 × 5-fold cross-validation, with thresholds chosen inside training folds (target sensitivity 80%). Tuned models also choose settings and feature set there; the three fixed rows (selected, Logistic 9, age only) are not tuned. No resampling: SMOTE raised the mean predicted risk to about 32% against a true 4.9%. Differences use Nadeau-Bengio corrected errors; every interval against the selected model includes zero. Logistic regression was chosen for simplicity and calibration, not superiority.</p>{bench}
<h2>5. Evaluation</h2><p>The model and its 5.175% threshold were frozen before the test set was scored. The columns are separate evaluations. Test ranking intervals use 4,000 bootstrap resamples, rates use Wilson intervals.</p>{table(['Metric', 'Development CV', 'Test (95%)', 'Age only, same test (95%)'], rows)}
<p>Test PR-AUC is above the development figure, but that figure lies inside the test interval: with 50 strokes this is consistent with sampling variation. Each model uses its own threshold from the same rule, so age-only catches {b['tp']} with {b['fp']} alarms while the four-feature model catches {a['tp']} with {a['fp']}. The model expected {a['expected_strokes']:.1f} test strokes; 50 occurred (calibration slope {a['calibration_slope']:.2f}). On development data, among people within three years of the same age, the model ranks stroke pairs better than age by 0.061 (95% 0.015 to 0.104).</p>
<h2>6. The main limitation: one threshold acts like an age cut-off</h2>{sub}<p>Overall sensitivity of 80% hides that strokes at 60 or under are mostly missed while nearly everyone older is flagged. Age-specific thresholds were not added: they would be another unvalidated layer. Other limits: unknown data collection and follow-up, no external validation, few strokes per subgroup.</p>
<h2>7. Reproducibility and deployment</h2><ul><li>Pinned versions; SHA-256 of every model and result file, verified by tests; a test scorer that refuses to run twice.</li><li>Streamlit app with six pages that reads saved artifacts and never trains.</li><li>Docker Compose: the app with a health check, and a monitor that checks logged risk bands every five minutes. Logs never hold inputs.</li><li>GitHub Actions runs the tests, then builds and health-checks the containers. MLflow records the frozen results.</li></ul><p>This is an educational simulation, not a medical diagnosis or a validated risk calculator.</p></html>"""
    (ROOT / "reports/report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
