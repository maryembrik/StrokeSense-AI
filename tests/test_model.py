import json, joblib, numpy as np, pandas as pd, pytest
from src.train import ROOT, FEATURES, build, development_data, oof_predictions, pick_threshold
RAW = pytest.mark.skipif(not (ROOT / "data/raw/stroke_data.csv").exists(), reason="raw dataset is not redistributed")
M = joblib.load(ROOT / "models/model.joblib"); META = json.load(open(ROOT / "models/metadata.json"))
BASE = dict(age=65, avg_glucose_level=120, hypertension=0, heart_disease=0)
risk = lambda **k: M.predict_proba(pd.DataFrame([{**BASE, **k}]))[0, 1]
def test_no_bmi_used(): assert "bmi" not in FEATURES and "bmi" not in " ".join(json.load(open(ROOT / "models/scorecard.json"))["points"])
@RAW
def test_test_rows_excluded():
    d = development_data(); assert len(d) == 4088 and d.stroke.sum() == 199
@pytest.mark.parametrize("lo,hi", [(dict(age=40), dict(age=80)), (dict(avg_glucose_level=90), dict(avg_glucose_level=220)), (dict(hypertension=0), dict(hypertension=1)), (dict(heart_disease=0), dict(heart_disease=1))])
def test_direction(lo, hi): assert risk(**hi) > risk(**lo)
def test_column_order_irrelevant():
    r = pd.DataFrame([BASE]); assert np.isclose(M.predict_proba(r)[0, 1], M.predict_proba(r[r.columns[::-1]])[0, 1])
@RAW
def test_retrain_reproduces():
    d = development_data(); assert np.allclose(build().fit(d, d.stroke)[-1].coef_, M[-1].coef_)
@RAW
def test_threshold_reproduces_and_hits_target():
    d = development_data(); p = oof_predictions(d, d.stroke); t = pick_threshold(d.stroke, p)
    assert t == META["threshold"] and (p[d.stroke.values == 1] >= t).mean() >= .8
