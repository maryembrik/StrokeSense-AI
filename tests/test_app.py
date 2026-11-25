from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import src.monitoring as mon
from src.inference import Predictor

R = Path(__file__).resolve().parents[1]
P = Predictor()
X = dict(age=65, avg_glucose_level=120, hypertension=0, heart_disease=0)


def app(v, monkeypatch):
    monkeypatch.syspath_prepend(str(R / "app"))
    return AppTest.from_file(str(R / f"app/views/{v}.py"), default_timeout=90).run()


def gauge_value(a):
    import re
    return [re.search(r"Estimated probability of the stroke class ([^,]+),", h.body).group(1) for h in a.get("html") if 'class="gauge"' in h.body][-1]


@pytest.mark.parametrize("v", ["overview", "simulator", "detective", "lab", "threshold", "drift"])
def test_page_runs(v, monkeypatch):
    a = app(v, monkeypatch)
    assert not a.exception, a.exception


def test_home_shell(monkeypatch):
    monkeypatch.syspath_prepend(str(R / "app"))
    a = AppTest.from_file(str(R / "app/Home.py"), default_timeout=90).run()
    assert not a.exception, a.exception


def test_simulator_matches_saved_pipeline(monkeypatch):
    a = app("simulator", monkeypatch)
    assert gauge_value(a) == "–"
    shown = []
    for age in (1, 17, 58, 82):
        a.slider[0].set_value(age)
        a.button[0].click().run()
        assert not a.exception
        shown.append(gauge_value(a))
        assert shown[-1] == f"{P.predict(dict(age=age, avg_glucose_level=105, hypertension=0, heart_disease=0))['risk']:.1%}"
    assert len(set(shown)) == 4


def test_full_age_range_keeps_frozen_values():
    for a, v in P.data["age_context"].items():
        assert P.ages["age_context"][a] == v and P.ages["age_only"][a] == P.data["age_only"][a]
    for a in (.08, 1, 17.5, 82):
        assert 0 < P.predict({**X, "age": a})["risk"] < 1


def test_research_json_matches_sources():
    import json
    import pandas as pd
    T = json.load(open(R / "artifacts/app/research.json", encoding="utf-8"))
    g = T["threshold"]
    assert g["frozen"] == P.data["threshold"] and g["frozen"] in g["grid"]
    al = g["groups"]["All"]
    assert (al["P"], al["N"]) == (199, 3889)
    for k in ("tp", "fp"):
        tot = np.sum([g["groups"][b][k] for b in g["groups"] if b != "All"], 0)
        assert np.allclose(tot, al[k]) and np.all(np.diff(al[k]) <= 0)
    old, i, j = P.data, P.data["grid"].index(.1), g["grid"].index(.1)
    assert np.isclose(old["curve"]["All"]["tp"][i], al["tp"][j])
    s = pd.read_csv(R / "artifacts/results/subgroups_verified.csv").set_index("group")
    assert T["missing_bmi"]["stroke"][0] == s.strokes["No BMI recorded"] and T["missing_bmi"]["overall"][0] == s.n["No BMI recorded"]
    assert sum(r["n"] for r in T["residence"]) == 4088 and sum(T["age"]["strokes"]) == 199


@pytest.mark.parametrize("v,lo,hi", [("threshold", 1.0, 30.0)])
def test_threshold_extremes(v, lo, hi, monkeypatch):
    a = app(v, monkeypatch)
    for t in (lo, hi):
        a.slider[0].set_value(t).run()
        assert not a.exception


@pytest.mark.parametrize("bad", [dict(age=0), dict(age=83), dict(age=float("nan")), dict(avg_glucose_level=900), dict(hypertension=.5), dict(age="abc"), dict(age=None)])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        P.predict({**X, **bad})


def test_missing_field():
    with pytest.raises(ValueError):
        P.predict({"age": 50})


def test_contributions_rebuild_risk():
    r = P.predict({**X, "hypertension": 1})
    z = P.data["intercept"] + sum(r["contrib"].values())
    assert np.isclose(1 / (1 + np.exp(-z)), r["risk"])


@pytest.mark.skipif(not (R / "data/raw/stroke_data.csv").exists(), reason="raw dataset is not redistributed")
def test_scorecard_matches_training_class():
    from src.scorecard import IntegerScorecard
    from src.train import development_data
    d = development_data()
    s = IntegerScorecard().fit(d.drop(columns="bmi"), d.stroke)
    q = d.head(60)
    assert [sum(P.points(dict(r)).values()) for _, r in q.iterrows()] == list(s.score(q))


def test_log_has_no_inputs():
    assert set(mon.log_prediction(.07, 1, "abcdefghij")) == {"event", "ts", "id", "model", "band", "latency_ms", "status"}


def test_drift_detects_and_stays_quiet():
    ref = P.data["drift"]
    assert all(r["status"] == "stable" for r in mon.drift_report(mon.simulate(ref, 1000), ref))
    assert any(r["status"] == "drift" and r["feature"] == "age" for r in mon.drift_report(mon.simulate(ref, 1000, age_shift=18), ref))


def test_frozen_files_and_single_test_scoring():
    import hashlib
    import json
    fz = json.load(open(R / "models/FREEZE.json", encoding="utf-8"))
    for k, v in fz["sha256"].items():
        assert hashlib.sha256((R / k).read_bytes()).hexdigest() == v, k
    T = json.load(open(R / "artifacts/results/test_results.json", encoding="utf-8"))
    m = T["models"]["Logistic, 4 features"]
    assert fz["test_split_scored"] and T["model_sha256"] == P.version and m["threshold"] == P.data["threshold"]
    assert (T["n"], T["strokes"]) == (1022, 50) and m["tp"] + m["fn"] == 50 and m["tp"] + m["fp"] + m["fn"] + m["tn"] == 1022
    age = T["subgroups"][:4]
    assert sum(g["n"] for g in age) == 1022 and sum(g["caught"] for g in age) == m["tp"] and sum(g["false_alarms"] for g in age) == m["fp"]


@pytest.mark.parametrize("tab", ["ROC and PR curves", "Feature stability", "Calibration", "Age-matched ranking", "Subgroups", "Final test", "Robustness", "Limitations"])
def test_lab_tabs(tab, monkeypatch):
    monkeypatch.syspath_prepend(str(R / "app"))
    a = AppTest.from_file(str(R / "app/views/lab.py"), default_timeout=90)
    a.session_state["labtabs"] = tab
    a.run()
    assert not a.exception, a.exception
    heading = {"ROC and PR curves": "ROC and precision-recall", "Age-matched ranking": "Ranking people of the same age", "Subgroups": "Subgroup analysis"}.get(tab, tab)
    assert any(f"<h2>{heading}</h2>" in h.body for h in a.get("html"))


def test_monitor_status(tmp_path):
    import src.monitor_job as job
    ref = job.reference_bands()
    assert np.isclose(sum(ref.values()), 1)
    few = [dict(event="prediction", band="<2%", latency_ms=5, status="success", model="x")] * 5
    assert job.summarise(few, ref)["status"] == "insufficient data"
    same = [dict(few[0], band=b) for b, v in ref.items() for _ in range(round(v * 500))]
    assert job.summarise(same, ref)["status"] == "stable"
    assert job.summarise([dict(few[0], band="10%+")] * 100, ref)["status"] == "drift"
    assert job.summarise([], ref)["events"] == 0


@pytest.mark.parametrize("age,word", [(39, "0, non-stroke"), (80, "1, stroke")])
def test_simulator_shows_class(age, word, monkeypatch):
    a = app("simulator", monkeypatch)
    a.slider[0].set_value(age)
    a.button[0].click().run()
    assert not a.exception
    assert any('class="verdict"' in h.body and word in h.body for h in a.get("html"))
