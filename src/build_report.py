"""Build reports/report.html from saved artifacts.  python -m src.build_report
PDF: msedge --headless --no-pdf-header-footer --print-to-pdf=reports/StrokeSense_Report.pdf file:///.../reports/report.html"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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



RJ = json.loads((ROOT / "artifacts/app/research.json").read_text(encoding="utf-8"))
from src.limitations import HEAD, NEEDS_DATA, NOT_RESOLVABLE, markdown  # noqa: E402
RBT = pd.read_csv(RES / "retraining_bootstrap.csv").set_index(["metric", "model"])
NB = len(pd.read_csv(RES / "retraining_resamples.csv"))
SM = pd.read_csv(RES / "smoking_sensitivity.csv")
H3 = pd.read_csv(RES / "three_vs_four.csv").query("metric == 'pr'").iloc[-1]
rb = lambda k, n: f"{RBT.loc[(k, n), 'mean']:+.3f} ({RBT.loc[(k, n), 'lo']:+.3f} to {RBT.loc[(k, n), 'hi']:+.3f})"
NS = pd.read_csv(RES / "operating_point_nested.csv", index_col=0).loc["Logistic nested-selection"]
AD = pd.read_csv(RES / "adult_training_check.csv", index_col=0)
CS = pd.read_csv(RES / "calibration_subgroups.csv").set_index("group")
YC = pd.read_csv(RES / "younger_group_cost.csv").iloc[-1]
cal = "; ".join(f"{k} {CS.expected[k]:.0f} vs {CS.strokes[k]}" for k in ("41-60", "61-70", "71+", "Women", "Men", "No BMI recorded"))
AB = pd.read_csv(RES / "ablation_cv.csv").groupby("label").pr_auc.mean()
MC = pd.read_csv(RES / "models_cv.csv").groupby("label").mean_pred.mean()
PV = pd.read_csv(RES / "paired_vs_logistic.csv", index_col=0)
TEAL, AMBER, GREY, INK = "#0B8578", "#B0701A", "#7A8B99", "#14222F"
FIG = ROOT / "reports/figures"


def save(f, name):
    f.savefig(FIG / f"{name}.png", dpi=190, bbox_inches="tight")
    plt.close(f)


def figures():
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": "#465868", "ytick.color": "#465868", "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    f, (x, z) = plt.subplots(1, 2, figsize=(7.4, 2.5))
    e, n, k = RJ["age"]["edges"], np.array(RJ["age"]["people"]), np.array(RJ["age"]["strokes"])
    x.bar([v + 2.5 for v in e[:-1]], 100 * k / n, width=4.3, color=TEAL)
    x.set(title="Stroke rate rises steeply with age", xlabel="age (five-year bands)", ylabel="% with stroke")
    mb = RJ["missing_bmi"]
    for j, (q, l, c) in enumerate([("no_stroke", "No stroke", TEAL), ("stroke", "Stroke", AMBER)]):
        kk, nn = mb[q]
        lo, hi = mb["ci"][q]
        z.barh(j, 100 * kk / nn, height=.5, color=c, xerr=[[100 * (kk / nn - lo)], [100 * (hi - kk / nn)]], error_kw=dict(ecolor=INK, capsize=3))
        z.text(100 * hi + .8, j, f"{kk / nn:.1%}  ({kk} of {nn:,})", va="center", fontsize=8.5)
    z.set(title="Blank BMI is not random", yticks=[0, 1], yticklabels=["No stroke", "Stroke"], xlim=(0, 42), xlabel="% with no BMI recorded (95% interval)")
    f.tight_layout(w_pad=2)
    save(f, "r1_data")

    f, (x, z) = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    end = []
    for k, l in (("HGB mono pairs", "Constrained boosting"), ("XGBoost", "XGBoost"), ("CatBoost", "CatBoost"), ("Logistic", "Logistic, 9 inputs"), ("Age only (logistic)", "Age only")):
        v, tree = L.loc[k, ["median", "draw", "drop"]].values, k not in ("Logistic", "Age only (logistic)")
        x.plot(range(3), v, color=AMBER if tree else TEAL, marker="D" if tree else "o", ls=":" if "Age" in k else "-", lw=1.8, ms=5)
        end.append([v[2], l])
    y = 0
    for v, l in sorted(end):
        y = max(v, y + .0035) if y else v
        x.text(2.08, y, f"{l} {v:.3f}", va="center", fontsize=8)
    x.set(title="Fill the blanks differently and the lead vanishes", xticks=range(3), xticklabels=["median", "random", "BMI removed"], ylabel="PR-AUC, CV mean", xlim=(-.1, 3.25))
    nm = [("Logistic, BMI median", "A raw + median BMI | Logistic"), ("+ 'BMI is blank' flag", "C + BMI-missing flag | Logistic")]
    b_ = z.bar([0, 1], [AB[q] for _, q in nm], width=.5, color=[TEAL, AMBER])
    z.bar_label(b_, fmt="%.3f", fontsize=9, padding=2)
    z.set(title="One flag is enough", xticks=[0, 1], xticklabels=["logistic,\nmedian BMI", "+ flag:\nBMI is blank"], ylim=(0, .27), ylabel="PR-AUC, CV mean")
    f.tight_layout(w_pad=2)
    save(f, "r2_leak")

    f, (x, z) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    Q = R[R.index != "Logistic 4 (selected)"].sort_values("dpr")
    x.errorbar(Q.dpr, range(len(Q)), xerr=Q.dpr_2se, fmt="o", color=TEAL, ms=5, capsize=3, lw=1.4)
    x.axvline(0, color=AMBER, lw=1.5)
    x.set(title="PR-AUC minus the selected model", yticks=range(len(Q)), yticklabels=[i.replace("Logistic", "Logistic, tuned") if i == "Logistic" else i for i in Q.index], xlabel="difference ± 2 corrected standard errors")
    for key, l, c, mk in (("Logistic 4 (selected)", "Logistic, 4 features", TEAL, "o"), ("CatBoost", "CatBoost", AMBER, "D")):
        c_ = RJ["calibration"][key]
        z.plot([100 * v for v in c_["pred"]], [100 * v for v in c_["obs"]], marker=mk, color=c, ms=4.5, lw=1.5, label=l)
    z.plot([0, 22], [0, 22], ls="--", color=GREY, lw=1, label="perfect")
    z.set(title="Calibration (development)", xlabel="predicted probability, %", ylabel="observed strokes, %", xlim=(0, 22), ylim=(0, 22))
    z.legend(frameon=False, fontsize=8)
    f.tight_layout(w_pad=2)
    save(f, "r3_models")

    f, (x, z) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ks = [("PR-AUC", "pr", "pr_auc"), ("Precision", "prec", "precision"), ("Specificity", "spec", "specificity"), ("Sensitivity", "sens", "sensitivity"), ("ROC-AUC", "roc", "roc_auc")]
    for j, (l, dv, t) in enumerate(ks):
        for q, off, c, mk, lab in ((a, .0, TEAL, "o", "test, 4 features (95%)"), (b, -.27, AMBER, "D", "test, age only (95%)")):
            v, (lo, hi) = q[t]["value"], q[t]["ci"]
            x.errorbar(v, j + off, xerr=[[v - lo], [hi - v]], fmt=mk, color=c, ms=5, capsize=2.5, lw=1.3, label=lab if not j else None)
        x.plot(m[dv], j + .27, "|", color=INK, ms=9, mew=2, label="development, 4 features" if not j else None)
    x.set(title="Final test, scored once (50 strokes)", yticks=range(5), yticklabels=[k[0] for k in ks], xlim=(0, 1))
    x.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(.42, -.12), ncols=3, columnspacing=1, handletextpad=.3)
    ag = T["subgroups"][:4]
    sc = [g["caught"] / g["strokes"] for g in ag]
    fl = [g["flagged"] / g["n"] for g in ag]
    w = np.arange(4)
    b1, b2 = z.bar(w - .2, [100 * v for v in sc], .36, color=TEAL, label="strokes caught"), z.bar(w + .2, [100 * v for v in fl], .36, color=AMBER, label="people flagged")
    for j, g in enumerate(ag):
        z.text(j - (.2 if j else .28), 100 * sc[j] + 2, f"{g['caught']}/{g['strokes']}", ha="center", fontsize=7.5)
        z.text(j + (.2 if j else .28), 100 * max(sc[j], fl[j]) + 2 + 10 * (j > 0), f"{g['flagged']}/{g['n']}", ha="center", fontsize=7.5)
    z.set(title="Same threshold, four ages (test)", xticks=w, xticklabels=[g["group"] for g in ag], ylim=(0, 145), yticks=[0, 50, 100], ylabel="%")
    z.legend(frameon=False, fontsize=7.5, loc="upper left", ncols=2)
    f.tight_layout(w_pad=1.5)
    save(f, "r4_test")


pair = {n: f"{D[k]['value']:+.3f} ({D[k]['ci'][0]:+.3f} to {D[k]['ci'][1]:+.3f})" for n, k in (("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"), ("Brier score", "brier"))}


def main():
    figures()
    fig = lambda n, c: f'<figure><img src="figures/{n}.png"><figcaption>{c}</figcaption></figure>'
    bench = table(["Model", "PR-AUC", "ROC-AUC", "Sens.", "Spec.", "Prec.", "F1", "Train − unseen PR"], [(k.replace(" (selected)", ", selected"), f"{r.pr:.3f}", f"{r.roc:.3f}", f"{r.sens:.1%}", f"{r.spec:.1%}", f"{r.prec:.1%}", f"{r.f1:.3f}", f"{r.gap_pr:+.3f}") for k, r in R.sort_values("pr", ascending=False).iterrows()] + [("Logistic, selection inside folds*", f"{NS.pr_fold:.3f}", f"{NS.roc_fold:.3f}", f"{NS.sens:.1%}", f"{NS.spec:.1%}", f"{NS.precision:.1%}", f"{NS.f1:.3f}", "–")])
    nb, ao = S.loc["No BMI recorded"], PV.loc["Age only"]
    caps = [("overview", "Overview: every amber point is a stroke, distance from the centre is age"), ("simulator", "Patient Simulator: the predicted class first, then its probability"), ("detective", "Leak Detective: the missing-BMI investigation"),
            ("lab7", "Research Lab: the final test, kept separate from development"), ("threshold", "Threshold Studio: illustrative thresholds; the frozen one never changes"), ("drift", "Drift Observatory: synthetic batches only, no real patients")]
    grid = '<div class="grid">' + "".join(f'<figure><img src="figures/app_{n}.jpg"><figcaption>{c}</figcaption></figure>' for n, c in caps) + "</div>"
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>StrokeSense AI report</title><style>
@page{{size:A4;margin:13mm 16mm}}body{{font:9.2pt/1.38 'Segoe UI',Arial,sans-serif;color:#14222F;margin:0}}h1{{font-size:23pt;line-height:1.1;margin:0 0 1.5mm;letter-spacing:-.02em}}h2{{font-size:12.5pt;margin:4.5mm 0 1.2mm;color:#0B6F65;break-after:avoid}}p{{margin:0 0 1.8mm}}a{{color:#0B6F65}}
.by{{color:#465868;margin-bottom:3mm}}.key{{border-left:3px solid #0B8578;background:#F0F5F4;padding:2.5mm 4mm;margin:2mm 0 3mm}}table{{border-collapse:collapse;width:100%;font-size:8.4pt;margin:1.5mm 0 2.5mm;break-inside:avoid;font-variant-numeric:tabular-nums}}th{{text-align:left;border-bottom:1.4px solid #14222F;padding:1mm 2mm 1mm 0}}td{{border-bottom:.5px solid #C9D3DA;padding:.8mm 2mm .8mm 0}}
figure{{margin:1.5mm 0 2.5mm;break-inside:avoid}}figure img{{width:84%;margin:0 auto;display:block}}.grid img{{width:66% !important}}figcaption{{font-size:8.2pt;color:#465868;margin-top:1mm}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1mm 4mm}}.grid figure{{margin:0}}.grid img{{border-radius:2mm}}ul{{margin:0 0 2mm;padding-left:5mm}}.lim table{{font-size:7.6pt;break-inside:auto}}.lim tr{{break-inside:avoid}}.lim td{{vertical-align:top;padding-right:3mm}}.lim td:first-child{{font-weight:600;width:17%}}</style>
<h1>Does AI know more than your birthday?</h1><p class="by">StrokeSense AI, a leakage-tested study of classifying the recorded stroke label. Academic project, instructor Hanna Abi Akl, November 2026. Author: Maryem Brik.<br>App: <a href="https://strokesense-ai.streamlit.app">strokesense-ai.streamlit.app</a> &nbsp; Code: <a href="https://github.com/maryembrik/StrokeSense-AI">github.com/maryembrik/StrokeSense-AI</a></p>
<p><b>Task.</b> Binary classification: 1 means the patient had a stroke, 0 means not. The model estimates the probability of class 1; a threshold turns it into a class. The dataset has no dates, follow-up or horizon, so "prediction" here means predicting the label, <b>not a future stroke</b>.</p><div class="key"><b>Answer.</b> Possibly a little more, but this study does not establish it. The first winning model owed its lead to blank BMI values. After removing that shortcut, a four-feature logistic model caught {a['tp']} of 50 strokes in a test set scored once, with {a['fp']} false alarms among 972 people without stroke. Its PR-AUC exceeds age alone by {diff}: the interval includes zero. A same-age ranking gain seen in development also becomes uncertain once feature selection and training are repeated.</div>
<h2>1. Data analysis</h2><p>5,110 people, 249 strokes (4.87%), no duplicates. BMI is blank for 201 people, smoking is "Unknown" for 1,544, and 856 are under 18. All rows were kept. Predicting "no stroke" for everyone gives 95.1% accuracy, so accuracy is useless here and PR-AUC is primary. The data were split 80/20, stratified, seed 42: 4,088 development rows (199 strokes) and 1,022 test rows (50).</p>
{fig('r1_data', 'Figure 1. Development data. Left: stroke rate by age. Right: share with no BMI recorded.')}
<p>Age confounds other variables. "Ever married" has an odds ratio of 4.14, but 0.67 after adjusting for age: it mainly marks older people.</p>
<h2>2. The shortcut: missing BMI</h2><p>Filling every blank with one median lets a tree isolate the blanks. Filling them at random, or removing BMI, erases boosting's lead. A direct check agrees: adding only a "BMI is blank" flag to logistic regression raises PR-AUC from {AB['A raw + median BMI | Logistic']:.3f} to {AB['C + BMI-missing flag | Logistic']:.3f}.</p>
{fig('r2_leak', 'Figure 2. Cross-validated PR-AUC, development data, fixed settings. Amber: tree boosting (left), flag added (right).')}
<p>We excluded BMI because we do not know why it is blank, or whether that would hold for new people. This does not prove target leakage. <b>The cost, at group level:</b> among the {nb.n:.0f} development records with no BMI the model expected {CS.expected["No BMI recorded"]:.0f} strokes in total and {nb.strokes:.0f} occurred (ratio {CS.oe["No BMI recorded"]:.1f}, approximate 95% {CS.oe_lo["No BMI recorded"]:.1f}–{CS.oe_hi["No BMI recorded"]:.1f}). This is a property of the group total. It does not show that each such person is underestimated, and a blank BMI is no evidence about an individual: {nb.n - nb.strokes:.0f} of the {nb.n:.0f} had no stroke.</p>
<h2>3. Feature selection</h2><p>Final features: age, average glucose, hypertension, heart disease. Stability selection repeated inside 20 outer folds chose them 20, 13, 12 and 3 times; gender, smoking, work, residence and marriage were never chosen. Heart disease was kept for clinical plausibility, which is not evidence of predictive value (see Robustness). Selection inside the folds gives PR-AUC 0.174, against {m.pr:.3f} for the four features fixed after exploration, so <b>{m.pr:.3f} is exploration-informed and probably optimistic; 0.174 is the conservative, selection-aware estimate</b>. All nine BMI-free variables give {R.loc['Logistic 9'].pr:.3f}.</p>
<h2>4. Model training</h2><p>Ten models, 4 × 5-fold stratified cross-validation. Scaling and encoding are fitted inside folds. Seven models are tuned by random search in an inner 5-fold loop over deliberately regularised spaces (for example trees of depth 1–3, strong L2), which also chooses 4 or 9 features. Three are fixed: the selected model (L2, C = 0.1), Logistic 9 and age only (a spline). Every threshold is the highest that reaches 80% sensitivity on inner predictions. No resampling: logistic with SMOTE raised the mean predicted probability to {MC['Logistic + SMOTE']:.1%} against a true 4.9%.</p>
{bench}<p>* Conservative estimate from a separate run with the same outer folds; the threshold comes from different inner folds.</p>{fig('r3_models', 'Figure 3. Above, PR-AUC and ROC-AUC are 20-fold means and other rates use counts averaged over four repeats. Left: no difference exceeds its corrected (Nadeau-Bengio) uncertainty. Right: ten groups by predicted probability.')}
<p>Logistic regression was chosen for simplicity and calibration (slope 1.09, 95% 0.95–1.24), not superiority. A patient bootstrap of fixed predictions is less neutral: it favours the selected model over XGBoost on ROC-AUC, and gives age alone {ao.d_fp:.0f} more false alarms (95% 43–99) for {ao.d_tp:.1f} more strokes caught (−4.8 to 12.5), with no clear net-benefit difference. It ignores retraining variation, so we do not lead with it.</p>
<p><b>Checks.</b> The selected model shows no positive train-to-unseen PR-AUC gap in this validation ({m.gap_pr:+.3f}), while Random Forest shows a larger one ({R.loc['Random Forest'].gap_pr:+.3f}). This does not rule out overfitting or selection-related optimism. With shuffled labels ROC-AUC falls to 0.50 over 200 fits. No identical feature rows between development and test. Splines or interactions on the four features add nothing (PR-AUC 0.187–0.191).</p>
<h2>5. Evaluation</h2><p>The model and its 5.175% threshold were frozen before the test rows were scored. The development column is a 20-fold mean (pooling folds instead gives a lower PR-AUC, 0.172). Test ranking intervals use 4,000 bootstrap resamples, rates use Wilson intervals. Paired differences compare both models on the same resampled people.</p>
{table(['Metric', 'Development, 4 features', 'Test, 4 features (95%)', 'Test, age only (95%)', 'Paired difference (95%)'], [r + (pair.get(r[0], '–'),) for r in rows])}
{fig('r4_test', 'Figure 4. Left: development and test. Right: test counts by age; several groups have fewer than 10 strokes.')}
<p>The model expected {a['expected_strokes']:.1f} test strokes and 50 occurred. Test PR-AUC is higher than in development, but the development value is inside its interval: this is compatible with chance. At its own threshold, age alone caught {b['tp']} strokes with {b['fp']} alarms; our model caught {a['tp']} with {a['fp']}. <b>Same-age ranking:</b> on development data, the model ranks 59.0% of pairs (one stroke, one not, within three years of age, same held-out fold) correctly, against 52.9% for age: +0.061 (95% 0.015–0.104; every pair equal, 150 patient resamples with predictions fixed). Counting every stroke equally gives +0.057 (0.017–0.102). This is a concordance gain, different from the PR-AUC difference. It keeps features and predictions fixed; repeating selection and training gives {rb('match', 'selected minus age')} (see Robustness). The two answer different questions and do not conflict: the gain is sensitive to which features get selected.</p><p><b>Calibration by group</b> (development, expected vs observed strokes): {cal}. Within age 71+ the slope is {CS.slope["71+"]:.2f} ({CS.slope_lo["71+"]:.2f}–{CS.slope_hi["71+"]:.2f}): the average is right, the ranking inside the group is weak.</p><p><b>Adults only?</b> Training on adults only and on all ages gives the same adult PR-AUC ({AD.pr["adults only"]:.3f} and {AD.pr["all ages"]:.3f}, difference {AD.pr["difference"]:+.4f} ± {AD.pr_2se["difference"]:.4f}), so no row was removed. This check was run after the freeze and changed nothing.</p>
<p><b>Robustness</b> (development only, after the freeze; nothing was changed). <i>Heart disease:</i> four inputs minus three gives PR-AUC {H3.difference:+.4f} ± {H3.two_se:.4f} over 100 folds (better in {H3.folds_better:.0f}). There is no consistent improvement, so it is not justified on predictive grounds; it stays because the model was already frozen and tested with it. <i>Smoking:</i> 30.5% is Unknown. Stability selection was repeated with Unknown as a category, drawn at random within age band (assumes Unknown people resemble known people of their age), removed, pooled as ever/never, and set to either extreme as a bound. Smoking was selected in at most {SM.sel_smoking_status.max()} of 100 draws in one treatment and never in the others. This concerns added prediction in this dataset, not medical effect.</p>
<p><i>Retraining-aware intervals:</i> {NB} stratified resamples; each repeats stability selection, refits and scores the people left out (about 73 strokes), with no convergence warnings.</p>
{table(['Gain over age alone', 'Four features fixed, retrained', 'Features selected again, retrained'], [(l, rb(k, 'four minus age'), rb(k, 'selected minus age')) for l, k in (('PR-AUC', 'pr'), ('ROC-AUC', 'roc'), ('Same-age ranking', 'match'))])}
<p>With selection repeated, every interval includes zero: the evidence that the extra features help beyond age is <b>suggestive, not established</b>. Out-of-bag training uses about 63% of distinct people, so scores are slightly pessimistic.</p>
<h2>6. Limitations</h2><p><b>Need new data.</b></p><div class="lim">{table(HEAD, NEEDS_DATA)}</div><p><b>Cannot be resolved from the current dataset.</b></p><div class="lim">{table(HEAD, NOT_RESOLVABLE)}</div>
<h2>7. The application</h2>{grid}
<h2>8. Reproducibility and hosting</h2><ul><li><b>Reproducible:</b> pinned versions, fixed seeds, SHA-256 of every frozen model and result file checked by tests, and a test scorer that refuses to run twice.</li><li><b>Automated:</b> all 44 tests pass locally with the dataset. In GitHub Actions the 4 that need the unpublished dataset are skipped, so 40 run. Actions also builds and health-checks the Docker containers.</li><li><b>Monitored:</b> a Compose service checks logged probability bands and latency every five minutes; inputs are never logged. It does not run on Streamlit Cloud. MLflow records the frozen results.</li><li><b>Hosted:</b> Streamlit Community Cloud. The app reads saved files and never trains.</li></ul>
<h2>References</h2><p><b>Dataset provenance.</b> Dataset provided by the university for the course project. Its original provenance was not independently verified.</p><p><b>Methods.</b> Nadeau, Bengio. Inference for the generalization error. Mach Learn 2003;52:239–281, doi:10.1023/A:1024068626366 (corrected standard errors). Cawley, Talbot. On over-fitting in model selection. JMLR 2010;11:2079–2107 (nested selection). Meinshausen, Bühlmann. Stability selection. JRSS B 2010;72:417–473, doi:10.1111/j.1467-9868.2010.00740.x. Wilson. Probable inference. JASA 1927;22:209–212 (rate intervals). van den Goorbergh et al. The harm of class imbalance corrections. JAMIA 2022;29:1525–1534, doi:10.1093/jamia/ocac093 (no resampling). Kapoor, Narayanan. Leakage and the reproducibility crisis. Patterns 2023;4:100804, doi:10.1016/j.patter.2023.100804 (shortcut check).</p><p><b>Evaluation and reporting.</b> Saito, Rehmsmeier. The precision-recall plot is more informative. PLoS ONE 2015;10:e0118432, doi:10.1371/journal.pone.0118432. Van Calster et al. Calibration: the Achilles heel. BMC Med 2019;17:230, doi:10.1186/s12916-019-1466-7. TRIPOD+AI. BMJ 2024;385:e078378 (self-check in docs/TRIPOD_AI.md).</p><p>This educational research application classifies observations according to the stroke label available in the dataset. It has not been validated as a prospective future-stroke prediction tool or a clinical risk calculator.</p></html>"""
    (ROOT / "docs/LIMITATIONS.md").write_text(markdown(), encoding="utf-8")
    (ROOT / "reports/report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
