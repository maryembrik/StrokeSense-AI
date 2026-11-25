"""Development split only, run after the freeze. Changes no model, threshold, feature or test result.   python -m src.robustness
1. Four inputs against three (no heart disease).  2. Does smoking stay unselected when 'Unknown' is treated differently?
3. Intervals that include feature selection and retraining: each bootstrap selects and refits, then scores the people left out."""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.adult_check import slope
from src.corrections import matched, select
from src.day1 import CORE, FEAT, age_spline, lr
from src.train import ROOT, development_data

RES = ROOT / "artifacts/results"
BAND = [-1, 17, 40, 60, 100]


def folds(d):
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42).split(d, d.stroke))


def three_vs_four(d):
    Y, rows = d.stroke.values, []
    sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42).split(d, d.stroke))  # first 20 folds are the project folds
    for i, (a, b) in enumerate(sp):
        for k, c in (("four", CORE), ("three", CORE[:3])):
            p = lr(c).fit(d.iloc[a], Y[a]).predict_proba(d.iloc[b])[:, 1]
            rows.append(dict(fold=i, model=k, pr=average_precision_score(Y[b], p), roc=roc_auc_score(Y[b], p), brier=brier_score_loss(Y[b], p)))
    r, out = pd.DataFrame(rows), []
    for lab, q in (("project folds, 4 x 5", r[r.fold < 20]), ("more repeats, 20 x 5", r)):
        for k in ("pr", "roc", "brier"):
            w = q.pivot(index="fold", columns="model", values=k)
            e, rep = w["four"] - w["three"], (w["four"] - w["three"]).groupby(w.index // 5).mean()
            good = (lambda v: v < 0) if k == "brier" else (lambda v: v > 0)
            out.append(dict(folds=lab, metric=k, four=w["four"].mean(), three=w["three"].mean(), difference=e.mean(), two_se=2 * e.std() * np.sqrt(1 / len(e) + .25), folds_better=int(good(e).sum()), n_folds=len(e), repeats_better=int(good(rep).sum()), n_repeats=len(rep)))
    out = pd.DataFrame(out)
    out.to_csv(RES / "three_vs_four.csv", index=False)
    print(out.round(4).to_string())


def treat(tr, how, seed):
    """Smoking treatment, learned on the training fold only."""
    tr = tr.copy()
    unk = tr.smoking_status == "Unknown"
    if how == "Unknown removed (sensitivity only)":
        return tr[~unk]
    if how.startswith("Extreme: all Unknown as "):
        tr.loc[unk, "smoking_status"] = how.split(" as ")[1]
    if how == "Unknown drawn at random within age band":
        r, g = np.random.default_rng(seed), pd.cut(tr.age, BAND, labels=False)
        for k in range(4):
            known = tr.smoking_status[(g == k) & ~unk].values
            m = (g == k) & unk
            tr.loc[m, "smoking_status"] = r.choice(known, int(m.sum()))
    if how == "Ever smoked, never smoked, unknown":
        tr["smoking_status"] = tr.smoking_status.replace({"smokes": "ever", "formerly smoked": "ever"})
    return tr


def smoke_job(how, i, tr, draw=0):
    warnings.filterwarnings("ignore")
    q = treat(tr, how, 500 + i + 100 * draw)
    k, f = select(q.reset_index(drop=True), q.stroke.reset_index(drop=True), i)
    return dict(treatment=how, fold=i, n=len(q), **f)


def smoking(d):
    H = ["Unknown as its own category (original)", "Unknown drawn at random within age band", "Ever smoked, never smoked, unknown", "Unknown removed (sensitivity only)", "Extreme: all Unknown as never smoked", "Extreme: all Unknown as smokes"]
    r = pd.DataFrame(Parallel(n_jobs=-2)(delayed(smoke_job)(h, i, d.iloc[a], draw) for h in H for i, (a, _) in enumerate(folds(d)) for draw in range(5 if "drawn" in h else 1)))
    out = pd.DataFrame([dict(treatment=h, selections=len(q), mean_n=q.n.mean(), smoking_freq=q.smoking_status.mean(), smoking_max=q.smoking_status.max(), **{"sel_" + f: int((q[f] >= .8).sum()) for f in FEAT}) for h, q in r.groupby("treatment", sort=False)])
    out.to_csv(RES / "smoking_sensitivity.csv", index=False)
    print(out.round(3).T.to_string())


def boot_job(d, s):
    """One resample: select features again, refit, score the people not drawn."""
    r, Y = np.random.default_rng(s), d.stroke.values
    i = np.r_[r.choice(np.flatnonzero(Y == 1), int(Y.sum())), r.choice(np.flatnonzero(Y == 0), int((Y == 0).sum()))]
    o = np.setdiff1d(np.arange(len(Y)), i)
    tr, y, age = d.iloc[i].reset_index(drop=True), Y[o], d.age.values[o]
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        k, _ = select(tr, tr.stroke, s)
        P = {n: m.fit(tr, tr.stroke).predict_proba(d.iloc[o])[:, 1] for n, m in (("selected", lr(k or ["age"])), ("four", lr(CORE)), ("three", lr(CORE[:3])), ("age", age_spline()))}
        x = dict(seed=s, strokes=int(y.sum()), n_out=len(o), selected="+".join(k), empty=not k)
        for n, p in P.items():
            x |= {f"pr_{n}": average_precision_score(y, p), f"roc_{n}": roc_auc_score(y, p), f"brier_{n}": brier_score_loss(y, p), f"slope_{n}": slope(np.clip(p, 1e-5, 1 - 1e-5), y), f"match_{n}": matched(p, np.zeros(len(y)), y, age, 3, 0)}
    x["not_converged"] = sum("Convergence" in w.category.__name__ for w in log)
    return x


def retrain(d, B=500):
    r = pd.DataFrame(Parallel(n_jobs=-2)(delayed(boot_job)(d, s) for s in range(B)))
    r.to_csv(RES / "retraining_resamples.csv", index=False)
    rows = []
    for m in ("pr", "roc", "brier", "slope", "match"):
        S = {n: r[f"{m}_{n}"] for n in ("selected", "four", "three", "age")}
        S |= {"four minus age": S["four"] - S["age"], "selected minus age": S["selected"] - S["age"], "four minus three": S["four"] - S["three"], "selected minus four": S["selected"] - S["four"]}
        rows += [dict(metric=m, model=n, mean=v.mean(), lo=v.quantile(.025), hi=v.quantile(.975), share_above_zero=(v > 0).mean(), mc_se_of_mean=v.std() / np.sqrt(B)) for n, v in S.items()]
    t = pd.DataFrame(rows)
    t.to_csv(RES / "retraining_bootstrap.csv", index=False)
    print(t.round(4).to_string())
    print("strokes left out", r.strokes.mean(), "people", r.n_out.mean(), "empty", r["empty"].sum(), "not converged", r.not_converged.sum(), r.selected.value_counts())
    for n in (250, 500):
        e = (r.pr_four - r.pr_age)[:n]
        print("stability", n, e.quantile([.025, .975]).round(4).tolist(), (r.match_four - r.match_age)[:n].quantile([.025, .975]).round(4).tolist())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    D = development_data().reset_index(drop=True)
    assert (D.stroke.sum(), len(D)) == (199, 4088)
    three_vs_four(D)
    retrain(D)
    smoking(D)
