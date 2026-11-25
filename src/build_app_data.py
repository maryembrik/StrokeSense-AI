"""Derive aggregate-only display data for the app from saved out-of-fold predictions.

No model is fitted here and the test split is never read. Run: python -m src.build_app_data
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.train import ROOT, development_data

RES = ROOT / "artifacts/results"
MAIN = "Logistic 4 (selected)"


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def main():
    d = development_data()
    y, age = d.stroke.values, d.age.values
    z = np.load(RES / "final_oof.npz")
    p, flags = z[MAIN + "|p"], z[MAIN + "|g"]
    t = json.load(open(ROOT / "models/metadata.json"))["threshold"]
    bands = {"40 or under": age <= 40, "41–60": (age > 40) & (age <= 60), "61–70": (age > 60) & (age <= 70), "71+": age > 70}
    masks = {"All": np.ones(len(y), bool), **bands}

    # row order check: these predictions must reproduce the frozen subgroup table
    ref = pd.read_csv(RES / "subgroups_verified.csv").set_index("group")
    for a, b in zip(bands, ["0-40", "41-60", "61-70", "71+"]):
        assert np.isclose(flags[:, bands[a] & (y == 1)].sum(1).mean(), ref.caught[b])

    # thresholds 1%..30% plus the exact frozen value
    grid = sorted({*np.round(np.arange(.01, .3001, .001), 3).tolist(), t})
    thr = {"grid": grid, "frozen": t, "groups": {}}
    for k, m in masks.items():
        pos, neg = p[:, m & (y == 1)], p[:, m & (y == 0)]
        thr["groups"][k] = dict(P=pos.shape[1], N=neg.shape[1], tp=[float((pos >= g).sum(1).mean()) for g in grid],
                                fp=[float((neg >= g).sum(1).mean()) for g in grid])

    # curves: folds pooled inside each repeat, then averaged over four repeats
    fx, rx = np.linspace(0, 1, 201), np.linspace(.01, 1, 200)
    curves, cal = {}, {}
    for m in [k[:-2] for k in z.files if k.endswith("|p")]:
        tprs, prs = [], []
        for q in z[m + "|p"]:
            f, tp, _ = roc_curve(y, q)
            tprs.append(np.interp(fx, f, tp))
            pr, rc, _ = precision_recall_curve(y, q)
            prs.append([pr[rc >= r].max() for r in rx])  # best precision at recall of at least r
        curves[m] = dict(tpr=[0.0, *np.mean(tprs, 0)[1:].round(5).tolist()], prec=np.mean(prs, 0).round(5).tolist())
        c = pd.DataFrame(dict(p=z[m + "|p"].ravel(), y=np.tile(y, 4)))
        c = c.groupby(pd.qcut(c.p, 10, duplicates="drop"), observed=True).mean()
        cal[m] = dict(pred=c.p.round(5).tolist(), obs=c.y.round(5).tolist())

    # residence subgroup: same nested flags and formulas as subgroups_verified.csv
    res = []
    for k in ("Urban", "Rural"):
        m = (d.Residence_type if "Residence_type" in d else d.residence_type).values == k
        s = m & (y == 1)
        tp, fl = flags[:, s].sum(1).mean(), flags[:, m].sum(1).mean()
        (a, b), (c, e), (g, h) = wilson(tp, s.sum()), wilson(fl, m.sum()), wilson(s.sum(), m.sum())
        res.append(dict(group=k, n=int(m.sum()), strokes=int(s.sum()), caught=tp, sensitivity=tp / s.sum(), sens_lo=a, sens_hi=b, flagged=fl / m.sum(),
                        flag_lo=c, flag_hi=e, oof_risk=float(p[:, m].mean()), observed=s.sum() / m.sum(), obs_lo=g, obs_hi=h))
    # false-alarm rate by age band at the nested operating point
    far = {k: float(flags[:, m & (y == 0)].sum(1).mean() / (m & (y == 0)).sum()) for k, m in bands.items()}

    miss = d.bmi.isna().values
    missing = dict(overall=[int(miss.sum()), len(y)], stroke=[int((miss & (y == 1)).sum()), int(y.sum())], no_stroke=[int((miss & (y == 0)).sum()), int((y == 0).sum())])
    missing["ci"] = {k: wilson(*missing[k]) for k in ("overall", "stroke", "no_stroke")}

    e = np.arange(0, 91, 5)
    pm = p.mean(0)
    dist = dict(edges=e.tolist(), people=np.histogram(age, e)[0].tolist(), strokes=np.histogram(age[y == 1], e)[0].tolist(),
                risk_q={k: np.quantile(pm[m], [0, .05, .1, .25, .5, .75, .9, .95, 1]).round(5).tolist() for k, m in bands.items()})
    # our own artifact; the app separately verifies its SHA-256 against FREEZE.json
    frozen = joblib.load(ROOT / "models/model.joblib").predict_proba(d)[:, 1]
    g_edges = np.arange(50, 281, 10)
    ref_drift = dict(glucose_edges=g_edges.tolist(), glucose=np.histogram(d.avg_glucose_level, g_edges)[0].tolist(),
                     risk_mean=float(frozen.mean()), risk_quantiles=np.quantile(frozen, np.linspace(0, 1, 201)).tolist())

    out = dict(threshold=thr, roc_x=fx.tolist(), pr_x=rx.tolist(), curves=curves, calibration=cal, residence=res, false_alarm_by_age=far,
               missing_bmi=missing, age=dist, reference=ref_drift, prevalence=float(y.mean()), n=len(y), strokes=int(y.sum()))
    json.dump(out, open(ROOT / "artifacts/app/research.json", "w"), default=float)
    print(missing, far, pd.DataFrame(res).round(3).to_string(), dist["risk_q"], sep="\n")


if __name__ == "__main__":
    main()
