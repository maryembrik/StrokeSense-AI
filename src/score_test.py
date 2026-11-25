"""One-time scoring of the locked test split.

    python -m src.score_test --dry-run   uses development rows only, writes nothing
    python -m src.score_test --final     scores the 1,022 test rows; refuses to run twice

Fixed before the test rows were scored: the frozen four-feature model and its threshold, an age-only spline logistic
baseline (same recipe as the benchmark) fitted on development rows with a threshold from the same out-of-fold rule,
the metrics below, 95% patient bootstrap intervals (4,000 resamples, seed 42) and Wilson intervals for rates.
PR-AUC is primary (average precision). Nothing is tuned here and the model file is never written.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from src.build_app_data import wilson
from src.train import ROOT, SEED, development_data, oof_predictions, pick_threshold

OUT = ROOT / "artifacts/results/test_results.json"


class AgeOnly:
    def fit(self, X, y):
        self.m = make_pipeline(SplineTransformer(n_knots=5, extrapolation="linear"), LogisticRegression(max_iter=3000)).fit(X[["age"]].values, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[["age"]].values)


def ranks(y, p):
    return np.array([average_precision_score(y, p), roc_auc_score(y, p), brier_score_loss(y, p)])


def operating(y, f):
    tp, fp, fn, tn = (int((a & b).sum()) for a, b in ((f, y == 1), (f, y == 0), (~f, y == 1), (~f, y == 0)))
    ratio = lambda k, n: dict(value=k / n, ci=wilson(k, n), k=k, n=n) if n else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, flagged=tp + fp, sensitivity=ratio(tp, tp + fn), specificity=ratio(tn, tn + fp), precision=ratio(tp, tp + fp),
                npv=ratio(tn, tn + fn), accuracy=(tp + tn) / len(y), f1=2 * tp / (2 * tp + fp + fn))


def evaluate(test, dev, model, t):
    y, yd = test.stroke.values, dev.stroke.values
    base, bt = AgeOnly().fit(dev, yd), pick_threshold(yd, oof_predictions(dev, dev.stroke, AgeOnly))
    P = {"Logistic, 4 features": (model.predict_proba(test)[:, 1], t), "Age only": (base.predict_proba(test)[:, 1], bt)}
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(4000)]
    boot = {k: np.array([ranks(y[i], p[i]) for i in idx]) for k, (p, _) in P.items()}
    ci = lambda a: [float(v) for v in np.quantile(a, [.025, .975])]
    out = {"n": len(y), "strokes": int(y.sum()), "prevalence": float(y.mean()), "models": {}}
    for k, (p, th) in P.items():
        f, v = p >= th, ranks(y, p)
        lo = np.log(p / (1 - p))[:, None]
        cal = LogisticRegression(C=1e9, max_iter=3000).fit(lo, y)
        out["models"][k] = dict(threshold=th, **{n: dict(value=float(v[j]), ci=ci(boot[k][:, j])) for j, n in enumerate(("pr_auc", "roc_auc", "brier"))},
                                **operating(y, f), mean_predicted=float(p.mean()), expected_strokes=float(p.sum()), calibration_slope=float(cal.coef_[0, 0]))
    a, b = list(P)
    out["difference"] = {n: dict(value=out["models"][a][n]["value"] - out["models"][b][n]["value"], ci=ci(boot[a][:, j] - boot[b][:, j])) for j, n in enumerate(("pr_auc", "roc_auc", "brier"))}
    p, f, age = P[a][0], P[a][0] >= t, test.age.values
    masks = {"40 or under": age <= 40, "41–60": (age > 40) & (age <= 60), "61–70": (age > 60) & (age <= 70), "71+": age > 70, "Women": (test.gender == "Female").values,
             "Men": (test.gender == "Male").values, "BMI recorded": test.bmi.notna().values, "No BMI recorded": test.bmi.isna().values}
    out["subgroups"] = []
    for g, m in masks.items():
        tp, pos, neg = int((f & m & (y == 1)).sum()), int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
        fp = int((f & m & (y == 0)).sum())
        out["subgroups"].append(dict(group=g, n=pos + neg, strokes=pos, caught=tp, false_alarms=fp, flagged=tp + fp, mean_predicted=float(p[m].mean()), observed=pos / (pos + neg),
                                     sens_ci=wilson(tp, pos) if pos else None, fa_ci=wilson(fp, neg)))
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    fz_path = ROOT / "models/FREEZE.json"
    fz, meta = json.loads(fz_path.read_text(encoding="utf-8")), json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    sha = lambda b: hashlib.sha256(b).hexdigest()
    for k, v in fz["sha256"].items():
        assert sha((ROOT / k).read_bytes()) == v, f"frozen file changed: {k}"
    raw = ROOT / "data/raw/stroke_data.csv"
    assert sha(raw.read_bytes()) == meta["data_sha256"], "raw data changed"
    dev = development_data()
    assert sha(",".join(map(str, sorted(dev.id))).encode()) == meta["dev_ids_sha256"], "development rows changed"
    model, t = joblib.load(ROOT / "models/model.joblib"), meta["threshold"]
    if a.dry_run:
        fit, hold = train_test_split(dev, test_size=.2, stratify=dev.stroke, random_state=1)
        print("DRY RUN, development rows only:", json.dumps(evaluate(hold, fit, model, t))[:3000])
        return
    assert not OUT.exists() and fz["test_split_scored"] is False, "the test split has already been scored; it is scored once"
    d = pd.read_csv(raw)
    test = d[~d.id.isin(dev.id)]
    assert (len(test), int(test.stroke.sum())) == (1022, 50)
    r = evaluate(test, dev, model, t)
    r["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r["model_sha256"] = fz["sha256"]["models/model.joblib"]
    OUT.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    fz["test_split_scored"], fz["test_scored_at"] = True, r["scored_at"]
    fz["sha256"]["artifacts/results/test_results.json"] = sha(OUT.read_bytes())
    fz["metadata_note"] = "models/metadata.json is kept byte-identical to the freeze, so its test_split_scored field describes the moment of freezing; this file is authoritative"
    fz_path.write_text(json.dumps(fz, indent=1), encoding="utf-8")
    print(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
