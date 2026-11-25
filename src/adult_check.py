"""Development split only. Run after the freeze; changes no model, threshold or test result.
1. Does training on all ages, instead of adults only, hurt adults?  2. Calibration inside subgroups.
3. What catching 80% at 41-60 would cost.  4. Age context for every training age (the frozen app data covers 18-82).   python -m src.adult_check"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.train import FEATURES, ROOT, development_data, pick_threshold

warnings.filterwarnings("ignore")
RES = ROOT / "artifacts/results"


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000))


def slope(p, y):
    z = np.log(p / (1 - p))[:, None]
    return float(LogisticRegression(C=1e6).fit(z, y).coef_[0][0])


def main():
    d = development_data()
    X, Y, age = d[FEATURES].values, d.stroke.values, d.age.values
    assert (Y.sum(), len(Y)) == (199, 4088)
    sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42).split(X, Y))
    rows = []
    for i, (a, b) in enumerate(sp):
        te = b[age[b] >= 18]
        for nm, tr in (("all ages", a), ("adults only", a[age[a] >= 18])):
            q = np.zeros(len(tr))
            for u, v in StratifiedKFold(5, shuffle=True, random_state=1000 + i).split(X[tr], Y[tr]):
                q[v] = lr().fit(X[tr[u]], Y[tr[u]]).predict_proba(X[tr[v]])[:, 1]
            ad = age[tr] >= 18
            p = lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]
            f = p >= pick_threshold(Y[tr][ad], q[ad])
            y = Y[te]
            rows.append(dict(fold=i, train=nm, n=len(te), strokes=int(y.sum()), pr=average_precision_score(y, p), roc=roc_auc_score(y, p), brier=brier_score_loss(y, p), slope=slope(p, y), mean_pred=p.mean(), observed=y.mean(),
                             tp=int((f & (y == 1)).sum()), fp=int((f & (y == 0)).sum())))
    r = pd.DataFrame(rows)
    out = r.groupby("train")[["pr", "roc", "brier", "slope", "mean_pred", "observed"]].mean()
    out[["tp", "fp"]] = r.groupby("train")[["tp", "fp"]].sum() / 4
    out[["n", "strokes"]] = int((age >= 18).sum()), int(Y[age >= 18].sum())
    for k in ("pr", "roc", "brier"):
        w = r.pivot(index="fold", columns="train", values=k)
        e = w["all ages"] - w["adults only"]
        out.loc["difference", [k, k + "_2se"]] = e.mean(), 2 * e.std() * np.sqrt(1 / 20 + .25)
    out.to_csv(RES / "adult_training_check.csv")
    print(out.round(5).to_string())

    Z = np.load(RES / "final_oof.npz")
    P = Z["Logistic 4 (selected)|p"]
    rng = np.random.default_rng(0)
    G = {"Everyone": age >= 0, "Under 18": age < 18, "18-40": (age >= 18) & (age <= 40), "41-60": (age > 40) & (age <= 60), "61-70": (age > 60) & (age <= 70), "71+": age > 70,
         "Women": (d.gender == "Female").values, "Men": (d.gender == "Male").values, "BMI recorded": d.bmi.notna().values, "No BMI recorded": d.bmi.isna().values}
    rows = []
    for k, m in G.items():
        y, p, c = Y[m], P[:, m], int(Y[m].sum())
        x = dict(group=k, n=int(m.sum()), strokes=c, expected=p.sum(1).mean())
        if c >= 10:
            one, zero, bs = np.flatnonzero(y == 1), np.flatnonzero(y == 0), []
            for _ in range(300):
                j = np.r_[rng.choice(one, len(one)), rng.choice(zero, len(zero))]
                k_ = rng.poisson(c)  # case count varies too, otherwise O/E would not
                bs.append((np.mean([slope(q[j], y[j]) for q in p]), k_ / x["expected"]))
            bs = np.array(bs)
            x |= dict(slope=np.mean([slope(q, y) for q in p]), slope_lo=np.quantile(bs[:, 0], .025), slope_hi=np.quantile(bs[:, 0], .975), oe=c / x["expected"], oe_lo=np.quantile(bs[:, 1], .025), oe_hi=np.quantile(bs[:, 1], .975))
        rows.append(x)
    c = pd.DataFrame(rows)
    c.to_csv(RES / "calibration_subgroups.csv", index=False)
    print(c.round(3).to_string())

    m, y = (age > 40) & (age <= 60), Y[(age > 40) & (age <= 60)]
    rows = []
    for s in (.3333, .5, .8):
        res = []
        for q in P[:, m]:
            t = np.sort(q[y == 1])[::-1][int(np.ceil(s * y.sum())) - 1]
            f = q >= t
            res.append((t, (f & (y == 1)).sum(), (f & (y == 0)).sum()))
        t, tp, fp = np.mean(res, 0)
        rows.append(dict(target=s, threshold=t, caught=tp, strokes=int(y.sum()), false_alarms=fp, people=int(m.sum()), alarms_per_stroke=fp / tp))
    e = pd.DataFrame(rows)
    e.to_csv(RES / "younger_group_cost.csv", index=False)
    print(e.round(4).to_string())

    pm, ctx = P.mean(0), {}
    for a in range(83):
        k = np.abs(age - a) <= 3
        ctx[a] = dict(n=int(k.sum()), strokes=int(Y[k].sum()), pct=[float(v) for v in np.quantile(pm[k], np.linspace(0, 1, 101))])
    s = make_pipeline(SplineTransformer(n_knots=5, extrapolation="linear"), LogisticRegression(max_iter=3000)).fit(age[:, None], Y)
    ao = {a: float(s.predict_proba([[a]])[0, 1]) for a in range(83)}
    old = json.loads((ROOT / "artifacts/app/data.json").read_text())
    for a in range(18, 83):  # frozen values must be reproduced exactly
        assert old["age_context"][str(a)]["n"] == ctx[a]["n"] and np.allclose(old["age_context"][str(a)]["pct"], ctx[a]["pct"]) and np.isclose(old["age_only"][str(a)], ao[a])
        ctx[a], ao[a] = old["age_context"][str(a)], old["age_only"][str(a)]
    (ROOT / "artifacts/app/age_all.json").write_text(json.dumps(dict(age_context=ctx, age_only=ao)), encoding="utf-8")
    print("under 18:", int((age < 18).sum()), "strokes", int(Y[age < 18].sum()), {a: (ctx[a]["n"], ctx[a]["strokes"], round(ao[a], 5)) for a in (0, 1, 5, 10, 17)})


if __name__ == "__main__":
    main()
