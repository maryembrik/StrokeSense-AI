"""Verify section-9 subgroup claims against the frozen model, then write aggregate-only data for the app. Never scores the test split."""
import hashlib, json, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from src.train import ROOT, FEATURES, development_data
if json.load(open(ROOT / "models/FREEZE.json"))["test_split_scored"]: raise SystemExit("Already frozen and test-scored; this script must not rewrite FREEZE.json.")
M = joblib.load(ROOT / "models/model.joblib"); META = json.load(open(ROOT / "models/metadata.json")); t = META["threshold"]
d = development_data(); Y = d.stroke.values; age = d.age.values; Z = np.load(ROOT / "artifacts/results/final_oof.npz"); P = Z["Logistic 4 (selected)|p"]; G = Z["Logistic 4 (selected)|g"]
assert (pd.Series(Y).sum(), len(Y)) == (199, 4088)
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return (c - h, c + h)
B = {"0-40": (age <= 40), "41-60": (age > 40) & (age <= 60), "61-70": (age > 60) & (age <= 70), "71+": age > 70}
B["No BMI recorded"] = d.bmi.isna().values; B["BMI recorded"] = d.bmi.notna().values; B["Women"] = (d.gender == "Female").values; B["Men"] = (d.gender == "Male").values
fz = M.predict_proba(d)[:, 1]; rows = []
for k, m in B.items():
    s = m & (Y == 1); n = int(m.sum()); c = int(s.sum()); tp = G[:, s].sum(1).mean(); fl = G[:, m].sum(1).mean()
    rows.append(dict(group=k, n=n, strokes=c, caught=tp, sensitivity=tp / c, sens_lo=wilson(tp, c)[0], sens_hi=wilson(tp, c)[1], flagged=fl / n, flag_lo=wilson(fl, n)[0], flag_hi=wilson(fl, n)[1], oof_risk=P[:, m].mean(), observed=c / n, obs_lo=wilson(c, n)[0], obs_hi=wilson(c, n)[1],
        frozen_flagged=float((fz[m] >= t).mean()), frozen_flagged_n=int((fz[m] >= t).sum()), frozen_min_risk=float(fz[m].min()), frozen_risk=float(fz[m].mean())))
T = pd.DataFrame(rows); T.to_csv(ROOT / "artifacts/results/subgroups_verified.csv", index=False); pd.set_option("display.width", 400, "display.max_columns", 40); print(T.round(3).to_string())
for lab, r in {"no conditions, glucose 55": (55, 0, 0), "no conditions, glucose 92 (median)": (92, 0, 0), "glucose 200, hypertension, heart disease": (200, 1, 1)}.items():
    a = np.arange(18, 83, .25); q = M.predict_proba(pd.DataFrame(dict(age=a, avg_glucose_level=r[0], hypertension=r[1], heart_disease=r[2])))[:, 1]; print("crossing age:", lab, a[q >= t].min())
# missing BMI calibration, matched on model risk: is it just older/sicker?
miss = d.bmi.isna().values; print("missing BMI: expected strokes", round(P[:, miss].mean(0).sum(), 1), "observed", Y[miss].sum(), "| by age:", {k: (round(float(P[:, miss & B[k]].mean(0).sum()), 1), int(Y[miss & B[k]].sum()), int((miss & B[k]).sum())) for k in ["0-40", "41-60", "61-70", "71+"]})
# ---- aggregate app data
grid = np.round(np.arange(.005, .3001, .0025), 4); bands = ["All", "0-40", "41-60", "61-70", "71+"]; mk = lambda k: np.ones(len(Y), bool) if k == "All" else B[k]
curve = {k: dict(P=int((mk(k) & (Y == 1)).sum()), N=int((mk(k) & (Y == 0)).sum()), tp=[float((P[:, mk(k) & (Y == 1)] >= g).sum(1).mean()) for g in grid], fp=[float((P[:, mk(k) & (Y == 0)] >= g).sum(1).mean()) for g in grid]) for k in bands}
pm = P.mean(0); ctx = {}
for a in range(18, 83):
    m = np.abs(age - a) <= 3; ctx[a] = dict(n=int(m.sum()), strokes=int(Y[m].sum()), pct=[float(x) for x in np.quantile(pm[m], np.linspace(0, 1, 101))])
sp = make_pipeline(SplineTransformer(n_knots=5, extrapolation="linear"), LogisticRegression(max_iter=3000)).fit(age[:, None], Y)
q = np.linspace(0, 1, 201); dec = lambda c: [float(x) for x in np.quantile(d[c], np.linspace(0, 1, 11))]
hist = lambda c, e: [float(v) for v in np.histogram(d[c], e)[0] / len(d)]
L = np.load(ROOT / "artifacts/results/final_oof.npz")
cal = pd.DataFrame(dict(p=P.ravel(), y=np.tile(Y, 4))); cal = cal.groupby(pd.qcut(cal.p, 10)).mean()
out = dict(threshold=t, grid=grid.tolist(), curve=curve, age_context=ctx, age_only={a: float(sp.predict_proba([[a]])[0, 1]) for a in range(18, 83)},
  coef=dict(zip(FEATURES, map(float, M[-1].coef_[0]))), mean=dict(zip(FEATURES, map(float, M[0].transformers_[0][1].mean_))), scale=dict(zip(FEATURES, map(float, M[0].transformers_[0][1].scale_))), intercept=float(M[-1].intercept_[0]),
  drift=dict(quantiles={c: [float(x) for x in np.quantile(d[c], q)] for c in ("age", "avg_glucose_level")}, edges={c: dec(c) for c in ("age", "avg_glucose_level")}, rates={c: float(d[c].mean()) for c in ("hypertension", "heart_disease")}, n=len(d)),
  calibration=dict(pred=cal.p.tolist(), obs=cal.y.tolist()), risk_by_age=[dict(age=a, lo=ctx[a]["pct"][10], mid=ctx[a]["pct"][50], hi=ctx[a]["pct"][90], n=ctx[a]["n"]) for a in range(18, 83)])
json.dump(out, open(ROOT / "artifacts/app/data.json", "w"))
fr = {str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted([*(ROOT / "models").glob("*.*"), *(ROOT / "artifacts/results").glob("*.*"), ROOT / "artifacts/app/data.json"])}
fr = {k: v for k, v in fr.items() if "FREEZE" not in k}
json.dump(dict(frozen_on="2026-11-25", test_split_scored=False, source_of_truth="notebook section 9; FINAL_RESULTS.csv; subgroups_verified.csv", superseded=["classification_table.csv", "final_benchmark.csv (operating points n/a; untuned trees)", "bootstrap_ci.csv (age-matched column)", "operating_point_nested.csv"], sha256=fr), open(ROOT / "models/FREEZE.json", "w"), indent=1)
print("coef", out["coef"], "intercept", out["intercept"])
