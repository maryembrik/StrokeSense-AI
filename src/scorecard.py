"""Integer 'pocket' scorecard: yes/no thresholds -> whole points a person can add by hand."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def _ll(z, y, a=1.0, b=0.0):
    return np.mean(np.logaddexp(0, z) - y * z)


class IntegerScorecard(ClassifierMixin, BaseEstimator):
    """L1 path picks at most `max_rules`; coefficients are scaled/rounded to integers,
    then polished by coordinate descent on log-loss. A 2-parameter logistic maps points to probability."""

    def __init__(self, max_rules=9, max_points=6):
        self.max_rules, self.max_points = max_rules, max_points

    def _rules(self, X):
        r = {}
        for t in (40, 50, 55, 60, 65, 70, 75):
            r[f"age >= {t}"] = X["age"] >= t
        for t in (126, 160, 200):
            r[f"glucose >= {t}"] = X["avg_glucose_level"] >= t
        if "bmi" in X:
            r["bmi >= 30"] = X["bmi"] >= 30
        r["hypertension"] = X["hypertension"] == 1
        r["heart disease"] = X["heart_disease"] == 1
        r["ever smoked"] = X["smoking_status"].isin(["smokes", "formerly smoked"])
        return pd.DataFrame(r).astype(float)

    def fit(self, X, y):
        R, y = self._rules(X), np.asarray(y)
        best = None
        for C in np.geomspace(0.003, 1, 25):
            m = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=500).fit(R, y)
            k = np.flatnonzero(m.coef_[0] > 1e-6)
            if 0 < len(k) <= self.max_rules:
                best = k
        self.cols_ = list(R.columns[best])
        M = R[self.cols_].values
        m = LogisticRegression(C=10, max_iter=500).fit(M, y)
        w = np.maximum(m.coef_[0], 0)
        top = None
        for scale in np.linspace(1, self.max_points, 11):
            p = np.round(w / w.max() * scale).astype(int)
            for _ in range(5):
                moved = False
                for j in range(len(p)):
                    cand = [(self._loss(M, np.r_[p[:j], v, p[j + 1:]], y), v)
                            for v in range(max(0, p[j] - 1), min(self.max_points, p[j] + 1) + 1)]
                    v = min(cand)[1]
                    moved |= v != p[j]
                    p[j] = v
                if not moved:
                    break
            l = self._loss(M, p, y)
            if top is None or l < top[0] - 1e-6:
                top = (l, p.copy())
        self.points_ = dict(zip(self.cols_, map(int, top[1])))
        self.points_ = {k: v for k, v in self.points_.items() if v}
        self.link_ = LogisticRegression(C=1e6).fit(self.score(X)[:, None], y)
        self.classes_ = np.array([0, 1])
        return self

    @staticmethod
    def _loss(M, p, y):
        s = M @ p
        if s.std() == 0:
            return 9.0
        # Closed grouping makes the 1-D fit cheap: Newton on (a, b)
        a, b = 0.5 / max(1, s.std()), np.log(y.mean() / (1 - y.mean()))
        sc = s - s.mean()
        for _ in range(12):
            q = 1 / (1 + np.exp(-(a * sc + b)))
            v = q * (1 - q) + 1e-9
            g = np.array([((q - y) * sc).sum(), (q - y).sum()])
            H = np.array([[(v * sc * sc).sum() + 1e-6, (v * sc).sum()], [(v * sc).sum(), v.sum()]])
            st = np.linalg.solve(H, g)
            a, b = a - st[0], b - st[1]
        return _ll(a * sc + b, y)

    def score(self, X, y=None):  # points, not accuracy
        R = self._rules(X)
        return sum(R[k].values * v for k, v in self.points_.items())

    def predict_proba(self, X):
        return self.link_.predict_proba(self.score(X)[:, None])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
