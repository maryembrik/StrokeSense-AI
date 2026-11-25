"""Single inference path for the app. Loads frozen artifacts; never trains."""
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"age": (18, 82), "avg_glucose_level": (55.0, 272.0), "hypertension": (0, 1), "heart_disease": (0, 1)}


class Predictor:
    def __init__(self):
        self.version = hashlib.sha256((ROOT / "models/model.joblib").read_bytes()).hexdigest()
        if self.version != json.load(open(ROOT / "models/FREEZE.json"))["sha256"]["models/model.joblib"]:
            raise RuntimeError("model.joblib does not match the frozen hash")
        # pickle is loaded only after its SHA-256 matched the frozen hash above
        self.model = joblib.load(ROOT / "models/model.joblib")
        self.meta = json.load(open(ROOT / "models/metadata.json"))
        self.card = json.load(open(ROOT / "models/scorecard.json"))
        self.data = json.load(open(ROOT / "artifacts/app/data.json"))
        self.features = self.meta["features"]

    @staticmethod
    def validate(x):
        out = {}
        for k, (lo, hi) in LIMITS.items():
            if k not in x:
                raise ValueError(f"missing {k}")
            try:
                v = float(x[k])
            except (TypeError, ValueError):
                raise ValueError(f"{k} must be a number")
            if not np.isfinite(v) or not lo <= v <= hi:
                raise ValueError(f"{k} must be between {lo} and {hi}")
            if hi == 1 and v not in (0, 1):
                raise ValueError(f"{k} must be 0 or 1")
            out[k] = v
        return out

    def points(self, x):
        p = {}
        for rule, v in self.card["points"].items():
            if rule.startswith("age"):
                hit = x["age"] >= float(rule.split()[-1])
            elif rule.startswith("glucose"):
                hit = x["avg_glucose_level"] >= float(rule.split()[-1])
            else:
                hit = x[rule.replace(" ", "_")] == 1
            p[rule] = v if hit else 0
        return p

    def predict(self, x):
        t0 = time.perf_counter()
        x = self.validate(x)
        r = float(self.model.predict_proba(pd.DataFrame([x])[self.features])[0, 1])
        d, a = self.data, str(int(round(x["age"])))
        c, pts = d["age_context"][a], self.points(x)
        s = sum(pts.values())
        return dict(
            risk=r, flagged=r >= d["threshold"], threshold=d["threshold"], age_only=d["age_only"][a], points=pts, total=s,
            card_risk=float(1 / (1 + np.exp(-(self.card["slope"] * s + self.card["intercept"])))), context=c,
            percentile=float(100 * (np.array(c["pct"]) < r).mean()),
            contrib={k: d["coef"][k] * (x[k] - d["mean"][k]) / d["scale"][k] for k in self.features},
            ms=1000 * (time.perf_counter() - t0))
