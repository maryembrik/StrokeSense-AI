import nbformat as nbf
N = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); py = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
md("""# StrokeSense - Part 1: forensic audit and model evidence
**Question.** Once age is accounted for, how much does machine learning add to stroke prediction on this dataset?

Every modelling decision here uses the 80% development split. The 20% test split is not evaluated in this notebook.""")
py('''
import sys, warnings; sys.path.insert(0, ".."); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
BLUE, ORANGE, INK, MUTE = "#2a78d6", "#eb6834", "#222", "#898781"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
  "axes.axisbelow": True, "font.size": 10, "figure.dpi": 110, "axes.titleweight": "bold", "axes.titlelocation": "left"})
df = pd.read_csv("../data/raw/stroke_data.csv"); df.head()''')
md("## 1. Structure and integrity")
py('''
pd.Series({"rows": len(df), "columns": df.shape[1], "duplicate ids": df.id.duplicated().sum(),
 "duplicate rows ignoring id": df.drop(columns="id").duplicated().sum(), "missing bmi": df.bmi.isna().sum(),
 "stroke cases": df.stroke.sum(), "stroke %": round(100*df.stroke.mean(), 2),
 "label changes down the file": int((df.stroke.diff().fillna(0) != 0).sum()),
 "Unknown smoking": (df.smoking_status=="Unknown").sum(), "Unknown smoking under 18": ((df.smoking_status=="Unknown")&(df.age<18)).sum()})''')
md("""The file is sorted by the target, so any unshuffled split would be invalid. 682 of the 1,544 `Unknown` smokers are under 18: a structural unknown, not a missing answer. We shuffle, stratify, and delete no rows.""")
py('''
from src.experiments import load_split
X, _, y, _ = load_split()      # second and fourth objects are the locked test split
D = X.assign(stroke=y); print(D.shape, int(D.stroke.sum()), "stroke cases in development data")''')
md("## 2. Exploration (development data)")
py('''
def wilson(k, n): 
    ci = stats.binomtest(int(k), int(n)).proportion_ci(method="wilson"); return ci.low, ci.high
def rate_plot(ax, g, title):
    t = D.groupby(g, observed=True).stroke.agg(["sum", "size"]); t = t[t["size"] >= 15]
    r = t["sum"]/t["size"]; ci = np.array([wilson(a, b) for a, b in t.values])
    ax.barh(range(len(t)), 100*r, color=BLUE, height=.55)
    ax.errorbar(100*r, range(len(t)), xerr=[100*(r-ci[:,0]).clip(lower=0), 100*(ci[:,1]-r).clip(lower=0)], fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.set_yticks(range(len(t))); ax.set_yticklabels([f"{i} (n={n})" for i, n in zip(t.index, t["size"])]); ax.set_title(title); ax.set_xlabel("stroke %, 95% CI"); ax.grid(axis="y", visible=False)
fig, ax = plt.subplots(3, 2, figsize=(13, 12)); ax = ax.ravel()
rate_plot(ax[0], pd.cut(D.age, [-1,18,40,50,60,70,83], labels=["0-18","19-40","41-50","51-60","61-70","71+"]), "Age")
rate_plot(ax[1], D.smoking_status, "Smoking"); rate_plot(ax[2], D.work_type, "Work type"); rate_plot(ax[3], D.ever_married, "Ever married")
rate_plot(ax[4], D.hypertension.map({0:"no",1:"yes"}), "Hypertension"); rate_plot(ax[5], D.heart_disease.map({0:"no",1:"yes"}), "Heart disease")
plt.tight_layout(); plt.savefig("../reports/figures/rates.png"); plt.show()''')
py('''
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, c, t in zip(ax, ["age","avg_glucose_level","bmi"], ["Age (years)","Average glucose (mg/dL)","BMI"]):
    for v, col, l in [(0, BLUE, "no stroke"), (1, ORANGE, "stroke")]:
        s = D.loc[D.stroke==v, c].dropna(); g = np.linspace(s.min(), s.max(), 200); a.plot(g, stats.gaussian_kde(s)(g), color=col, lw=2, label=l)
    a.set_title(t); a.set_yticks([]); a.legend(frameon=False)
plt.tight_layout(); plt.savefig("../reports/figures/distributions.png"); plt.show()''')
md("""**Glucose has two humps.** High readings form a second subpopulation around 200 mg/dL. They look like a plausible group rather than errors, so trimming them as outliers would delete patients with a higher stroke rate.

### Confounding: marriage does not cause strokes
Raw odds ratios against odds ratios adjusted for a smooth age effect.""")
py('''
rows = []
A = np.column_stack([(D.age-45)/20, ((D.age-45)/20)**2, np.maximum(D.age-60,0)/20])
for n, v in {"ever married": D.ever_married=="Yes", "self-employed": D.work_type=="Self-employed", "hypertension": D.hypertension==1, "heart disease": D.heart_disease==1,
             "glucose >= 160": D.avg_glucose_level>=160, "formerly smoked": D.smoking_status=="formerly smoked", "urban": D.Residence_type=="Urban", "BMI missing": D.bmi.isna()}.items():
    v = v.astype(float).values[:, None]; f = lambda Z: np.exp(LogisticRegression(C=1e6, max_iter=5000).fit(Z, D.stroke).coef_[0][0])
    rows.append((n, f(v), f(np.hstack([v, A]))))
o = pd.DataFrame(rows, columns=["factor","raw OR","age-adjusted OR"]).set_index("factor").round(2)
o2 = o.sort_values("raw OR"); fig, a = plt.subplots(figsize=(8, 4))
a.hlines(range(len(o2)), o2["age-adjusted OR"], o2["raw OR"], color="#c3c2b7", lw=2)
a.scatter(o2["raw OR"], range(len(o2)), color=BLUE, s=60, label="raw", zorder=3); a.scatter(o2["age-adjusted OR"], range(len(o2)), color=ORANGE, marker="D", s=50, label="age-adjusted", zorder=3)
a.set_yticks(range(len(o2))); a.set_yticklabels(o2.index); a.axvline(1, color=INK, lw=1); a.set_xscale("log"); a.set_xticks([.5,1,2,4,6]); a.set_xticklabels([.5,1,2,4,6]); a.minorticks_off(); a.set_title("Odds ratio before and after adjusting for age"); a.legend(frameon=False); plt.tight_layout(); plt.savefig("../reports/figures/confounding.png"); plt.show(); o''')
md("""## 3. The BMI trap
Is a *blank* BMI a stronger signal than BMI itself? A value that is blank more often for stroke patients, even at the same age, suggests recording happened after or because of the outcome.""")
py('''
b = pd.cut(D.age, [40, 60, 70, 83]); t = D.groupby([b, D.bmi.isna()], observed=True).stroke.agg(["size","sum","mean"]).round(3); print(t)
ab = pd.read_csv("../artifacts/results/ablation_cv.csv"); ab.groupby("label", sort=False)[["pr_auc","roc_auc","age_matched_auc","brier"]].mean().round(4)''')
md("""The flag is the only added feature that clearly raises scores, and it is the one we refuse. Nothing tells us *why* BMI is blank; if it reflects incomplete notes for stroke patients, the flag will not exist at prediction time. We report it as a sensitivity analysis and keep it out of the final model. The cost is real and visible in the table.

The other engineered ideas (peer-relative deviation, age-conditional imputation, smoking split, burden count) did not help. They stay documented as negative results and out of the model.

## 4. Benchmark: 4 x 5-fold cross-validation, development data only""")
py('''
m = pd.concat([pd.read_csv("../artifacts/results/models_cv.csv"), pd.read_csv("../artifacts/results/shape_cv.csv")])
cols = ["pr_auc","roc_auc","age_matched_auc","brier","mean_pred","recall_top20","net_benefit_10"]
T = m.groupby("label")[cols].mean().round(4); T.insert(1, "pr_auc_sd", m.groupby("label").pr_auc.std().round(3)); T.sort_values("pr_auc", ascending=False)''')
py('''
P = m.pivot(index="fold", columns="label", values="pr_auc"); keep = ["HGB mono pairs","HistGradientBoosting","XGBoost","CatBoost","Logistic","EBM","EBM monotone","Balanced RF","Integer scorecard","Random Forest + SMOTE"]
d = P[keep].sub(P["Age only (logistic)"], axis=0); mu = d.mean(); ci = 2*d.std()*np.sqrt(1/20+.25); s = mu.sort_values().index
fig, a = plt.subplots(figsize=(8, 4.5)); a.errorbar(mu[s], range(len(s)), xerr=ci[s], fmt="o", color=BLUE, ecolor="#9ec5f4", elinewidth=3, ms=8); a.axvline(0, color=INK, lw=1)
a.set_yticks(range(len(s))); a.set_yticklabels(s); a.set_title("Gain in PR-AUC over a model that knows only age"); a.set_xlabel("difference, +/- 2 corrected SE (Nadeau-Bengio)"); plt.tight_layout(); plt.savefig("../reports/figures/gain_over_age.png"); plt.show()''')
md("""At this point boosting looks like a clear winner. Section 5 shows why that reading is wrong.

## 5. The missing-data artifact: constant imputation
All blank BMIs were filled with the *same* number (the median). Trees can isolate that exact value and so rebuild the missing-BMI flag we refused. Test: refit with (a) an age-conditional random draw instead of a constant, and (b) no BMI at all.""")
py('''
L = pd.read_csv("../artifacts/results/leak_cv.csv")
Q = L.pivot_table(index="label", columns="mode", values="pr_auc")[["median","draw","drop"]].sort_values("median"); 
fig, a = plt.subplots(figsize=(8, 4.8)); n = range(len(Q))
a.hlines(n, Q["drop"], Q["median"], color="#c3c2b7", lw=2)
a.scatter(Q["median"], n, color=ORANGE, marker="D", s=50, label="BMI filled with median", zorder=3); a.scatter(Q["drop"], n, color=BLUE, s=60, label="BMI removed", zorder=3)
a.set_yticks(n); a.set_yticklabels(Q.index); a.set_xlabel("PR-AUC (4 x 5-fold CV)"); a.set_title("Boosting's advantage disappears without the BMI spike"); a.legend(frameon=False, loc="lower right"); plt.tight_layout(); plt.savefig("../reports/figures/leak.png"); plt.show(); Q.round(4)''')
py('''
P = L[L["mode"]=="drop"].pivot(index="fold", columns="label", values="pr_auc"); d = P.sub(P["Age only (logistic)"], axis=0).drop(columns="Age only (logistic)")
pd.DataFrame({"gain over age-only": d.mean(), "+/- 2 corrected SE": 2*d.std()*np.sqrt(1/20+.25), "folds won /20": (d>0).sum()}).round(4)''')
md("""### Reading section 5
The apparent advantage of boosting under median-imputed BMI shrinks strongly when BMI is excluded or imputed by random draw. This suggests the models exploited the missingness pattern rather than measured BMI. It does **not** prove target leakage: blank BMI may be a recording artifact, a proxy for something unmeasured, or real information. We exclude it because we cannot know that it would exist at prediction time. SMOTE and balanced forests also inflate predicted risk (mean about 32% against a true 4.9%).

## 6. Final selection
**Rule, written in `src/day1.py` before the run:** among BMI-free models take the highest mean PR-AUC, then the simplest model within one corrected standard error. Every model uses the same 4 x stratified 5-fold split (20 held-out folds).

**Age-matched AUC:** share of case/control pairs within +/-3 years of age ranked correctly (ties count half), using pairs scored by the same outer-fold model. Section 8 gives sensitivity and uncertainty. The bootstrap table below pools scores across folds and is superseded by section 8 for age-matched AUC.""")
py('''pd.read_csv("../artifacts/results/final_benchmark.csv", index_col=0).round(4)''')
py('''display(pd.read_csv("../artifacts/results/bootstrap_ci.csv", header=[0,1], index_col=0)); display(pd.read_csv("../artifacts/results/persistence.csv").round(4)); display(pd.read_csv("../artifacts/results/calibration.csv", index_col=0).round(3))''')
py('''
Z = np.load("../artifacts/results/oof.npz"); yy = Z["y"]; fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
p4, pa, ps = Z["Logistic (4 clinical)"], Z["Age spline only"], Z["Logistic + SMOTE ~ drop"]
for p, c, mk, l in [(p4, BLUE, "o", "4-feature logistic"), (ps, ORANGE, "D", "logistic + SMOTE")]:
    g = pd.DataFrame(dict(y=yy, p=p)).groupby(pd.qcut(p, 10, duplicates="drop"), observed=True).mean(); ax[0].plot(g.p, g.y, marker=mk, color=c, lw=2, label=l)
ax[0].plot([0, 1], [0, 1], color=MUTE, lw=1, ls="--"); ax[0].set_title("Calibration (deciles)"); ax[0].set_xlabel("predicted risk"); ax[0].set_ylabel("observed rate"); ax[0].legend(frameon=False)
t = np.linspace(.01, .25, 60); nb = lambda p: [((p>=k)&(yy==1)).mean()-((p>=k)&(yy==0)).mean()*k/(1-k) for k in t]
ax[1].plot(t, nb(p4), color=BLUE, lw=2, label="4-feature logistic"); ax[1].plot(t, nb(pa), color=ORANGE, lw=2, label="age only"); ax[1].plot(t, nb(np.ones(len(yy))), color=MUTE, lw=1, ls="--", label="flag everyone"); ax[1].axhline(0, color=INK, lw=1, label="flag nobody")
ax[1].set_ylim(-.01, .045); ax[1].set_title("Decision curve"); ax[1].set_xlabel("risk threshold"); ax[1].set_ylabel("net benefit"); ax[1].legend(frameon=False)
st = pd.read_csv("../artifacts/results/stability.csv").head(10)[::-1]; ax[2].barh(st.feature, 100*st.selected, color=BLUE, height=.6); ax[2].set_title("L1 selection stability, 200 bootstraps"); ax[2].set_xlabel("% of bootstraps selected"); ax[2].grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig("../reports/figures/reliability.png"); plt.show()''')
py('''import json; display(json.load(open("../models/metadata.json"))); json.load(open("../models/scorecard.json"))''')
md("""### Conclusions
- **Selected:** L2 logistic regression on age, average glucose, hypertension and heart disease; no BMI. It had the highest mean PR-AUC in the section 6 benchmark. Once every algorithm is tuned and given the same four features (section 7), CatBoost ties it (0.194 vs 0.191, difference 0.003 +/- 0.018), so logistic is chosen for simplicity, not superiority. Tuning C barely changes PR-AUC, so the default 0.1 was kept.
- **Beyond age?** A modest yes for ranking people of similar age; corrected estimate in section 8. The PR-AUC gain over age is not established.
- **Calibration** is close to ideal (slope 1.10, ECE 0.010), with no resampling or extra calibration step.
- **Scorecard:** convenient, but its PR-AUC is lower (0.165) and outside the selection rule. It is a communication aid, not the model. Its formula gives 'age' most of the weight.
- **Limits.** The four-feature set was defined from clinical reasoning after seeing exploratory odds ratios; stability was computed afterwards on the same development data. Bootstrap intervals resample fixed out-of-fold predictions, so they omit retraining variation. Our first audit summarised all 5,110 rows, including test labels, before the split was isolated; no model has been scored on the test split.""")
md("""## 7. Adversarial review
**Fairness of the benchmark.** Section 6 used fixed hyperparameters for tree models and gave the four-feature set only to logistic regression. Here every algorithm gets nested tuning (3-fold inner, 10 outer folds) on both feature sets.""")
py('''display(pd.read_csv("../artifacts/results/nested_tuning.csv").groupby(["F","m"])[["pr","roc","am"]].mean().round(4)); pd.read_csv("../artifacts/results/bmi_mechanism.csv", index_col=0)''')
md("""**Mechanism of the BMI artifact (second table, boosting).** Any constant fill gives PR-AUC near 0.21; adding random jitter to the fill removes the gain (0.161, below the 0.173 of no BMI). Among patients with *recorded* BMI, the median-filled model is worse than no BMI (0.140 vs 0.151). The gain therefore comes from identifying blanks, not from BMI values. Why BMI is blank remains unknown.

**Sanity.** Permuted labels give ROC-AUC 0.487 and PR-AUC 0.050 (chance is 0.5 and 0.049). An independent re-implementation reproduced the headline numbers exactly.

**A weakness.** Restricted to age 40 and over, ROC-AUC is 0.734, against 0.719 for age alone. Much of the 0.84 comes from separating young people from old ones.""")
md("""## 8. Corrections after review (these supersede earlier operating-point and age-matched numbers)
**Protocol.** Fixed settings, outer 4 x stratified 5-fold. Each threshold comes from inner out-of-fold predictions of the outer training part only and is then frozen. Counts are averages over the four repeats (each repeat scores every patient once), hence fractions. `pr_fold` = mean over 20 folds (primary); `pr_pooled` = mean over repeats of pooled PR-AUC; they are never compared with each other. Intervals resample patients with predictions held fixed, so they omit retraining variation.""")
py('''T = pd.read_csv("../artifacts/results/operating_point_nested.csv"); display(T.round(3)); display(pd.read_csv("../artifacts/results/paired_vs_logistic.csv").round(2)); display(pd.read_csv("../artifacts/results/nested_selection.csv", index_col=0)); display(pd.read_csv("../artifacts/results/calibration_outer.csv").round(3)); pd.read_csv("../artifacts/results/age_matched_sensitivity.csv").round(3)''')
py('''
S = pd.read_csv("../artifacts/results/age_matched_sensitivity.csv"); fig, a = plt.subplots(figsize=(8, 4))
for k, (w, c, m) in enumerate([("pair", BLUE, "o"), ("case", ORANGE, "D")]):
    q = S[S.weighting==w]; a.errorbar(np.arange(4)+.12*(k-.5), q.gain, yerr=[q.gain-q.gain_lo, q.gain_hi-q.gain], fmt=m, color=c, capsize=4, label=w+"-weighted")
a.set_xticks(range(4)); a.set_xticklabels(["same age", "+/-1 y", "+/-3 y", "+/-5 y"]); a.axhline(0, color=INK, lw=1); a.legend(frameon=False); a.set_title("Within-age ranking gain over age only (95% interval)"); plt.tight_layout(); plt.savefig("../reports/figures/age_matched.png"); plt.show()''')
md("""**The 0.061 versus 0.093 discrepancy.** 0.061 compares pairs inside the same fold. 0.093 came from pooled scores averaged over repeats: different fold models put near-identical ages on slightly different scales, which pushed age-only below chance (0.489) and inflated the gap. **0.093 is withdrawn; the estimate is 0.061 (95% interval 0.015 to 0.104).** The gain stays positive at every window; at +/-5 years, case-weighted, the interval touches zero.

**Operating point.** With honest thresholds the logistic model reaches 79.3% sensitivity (not the 80.4% seen when the threshold was chosen on the same data). The earlier 33-alarm XGBoost advantage disappears: XGBoost catches 6 more strokes with 82 more false alarms, and net benefit at 5% does not differ clearly for any model.

**Feature selection.** Repeating a stability selection inside every outer fold gives PR-AUC 0.174, against 0.190 for the four features chosen after exploration, so 0.190 is probably optimistic. All nine BMI-free variables, with no selection step, give 0.186. Age is always selected, glucose and hypertension in 13 and 12 of 20 folds, heart disease in 3. Selection depends on the penalty: section 6 used a weaker one and reported 86% for heart disease. Heart disease is kept on clinical grounds, not on stability.

**Conclusion.** Keep the four-feature logistic model for parsimony, calibration (slope 1.10, interval 0.95 to 1.24) and lack of a robust advantage elsewhere. It is not proven to be the best classifier. The test split remains unscored; it was not used for fitting, tuning, thresholds or evaluation, although initial integrity and target summaries covered all rows.""")
md("""## 9. Final fair benchmark (reference table; supersedes the operating-point table of section 8)
Every tuned model has a regularised search space; hyperparameters, feature set (4 or 9, no BMI) and threshold are chosen inside each outer training fold. `Logistic` is the tuned version; the fixed rows have no tuning. Counts average 4 repeats.""")
py('''display(pd.read_csv("../artifacts/results/FINAL_RESULTS.csv", index_col=0).round(3)); pd.read_csv("../artifacts/results/subgroups.csv").round(3)''')
md("""**Overfitting.** The selected logistic model shows no train-to-unseen loss (PR gap -0.008, ROC gap 0.003). Regularised tuning reduced the trees' gaps strongly (Random Forest PR gap is 0.049) without making any of them better. No model's PR-AUC differs from the selected one by more than 2 corrected SE.

**The main limitation: the threshold behaves almost like an age cut-off.** At age 71+ it flags everyone; at 61-70, 92%; at 41-60 it finds only 33% of strokes. Within age bands ROC-AUC is 0.56 to 0.71. The overall 79% sensitivity must always be shown with this breakdown.

**Other subgroups.** Sensitivity is 75% for women and 85% for men; intervals overlap (66-82% vs 76-91%). Patients with no recorded BMI are under-predicted (8.5% vs 22% observed), the direct cost of excluding that signal. Wilson intervals treat the average over repeats as one sample and are approximate.""")
md("""### 10. Final test (scored once, after the freeze; supersedes "remains unscored" above)
| | Logistic, 4 features | Age only |
|---|---|---|
| PR-AUC | 0.259 (0.168 to 0.372) | 0.204 (0.140 to 0.297) |
| ROC-AUC | 0.840 (0.777 to 0.893) | 0.834 (0.772 to 0.887) |
| Brier | 0.041 (0.031 to 0.052) | 0.042 (0.032 to 0.053) |
| Sensitivity | 0.800 (0.670 to 0.888) | 0.840 (0.715 to 0.917) |
| Specificity | 0.740 (0.711 to 0.766) | 0.719 (0.690 to 0.746) |
| Precision | 0.137 (0.102 to 0.181) | 0.133 (0.100 to 0.175) |
| False alarms / missed strokes | 253 / 10 | 273 / 8 |

1,022 people, 50 strokes. PR-AUC difference +0.055 (95% -0.012 to +0.130): includes zero, so no advantage over age is established. Produced by `python -m src.score_test --final`; no change was made afterwards.""")
N.cells = C; N.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nbf.write(N, "notebooks/StrokeSense_Research.ipynb")
