"""Train and freeze the selected model. Run: python -m src.train   (never scores the test split)"""
import hashlib, json, platform
import joblib, numpy as np, pandas as pd, sklearn
from pathlib import Path
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.scorecard import IntegerScorecard

ROOT = Path(__file__).resolve().parents[1]; SEED = 42; TARGET_SENSITIVITY = 0.80
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

def development_data():
    d = pd.read_csv(ROOT / "data/raw/stroke_data.csv")
    dev, _locked = train_test_split(d, test_size=.2, stratify=d.stroke, random_state=SEED)
    return dev

def build():
    return make_pipeline(make_column_transformer((StandardScaler(), FEATURES)), LogisticRegression(C=.1, max_iter=3000))

def oof_predictions(X, y, factory=build):
    P = np.zeros((4, len(y)))
    for i, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(X, y)):
        P[i // 5, b] = factory().fit(X.iloc[a], y.iloc[a]).predict_proba(X.iloc[b])[:, 1]
    return P.mean(0)

def pick_threshold(y, p, sens=TARGET_SENSITIVITY):
    """Highest threshold whose out-of-fold sensitivity reaches the target."""
    pos = np.sort(p[np.asarray(y) == 1])[::-1]
    return float(np.floor(pos[int(np.ceil(sens * len(pos))) - 1] * 1e6) / 1e6)

def main():
    d = development_data(); X, y = d, d.stroke
    p = oof_predictions(X, y); t = pick_threshold(y, p); f = p >= t; yy = y.values
    m = build().fit(X, y); (ROOT / "models").mkdir(exist_ok=True); joblib.dump(m, ROOT / "models/model.joblib")
    sc = IntegerScorecard().fit(X.drop(columns="bmi"), y)
    json.dump({"points": sc.points_, "slope": float(sc.link_.coef_[0][0]), "intercept": float(sc.link_.intercept_[0])}, open(ROOT / "models/scorecard.json", "w"), indent=1)
    meta = {"model": "L2 logistic regression, C=0.1, no BMI", "features": FEATURES, "threshold": t,
      "threshold_rule": f"highest out-of-fold threshold with sensitivity >= {TARGET_SENSITIVITY}", "n_dev": len(y), "n_dev_strokes": int(y.sum()),
      "oof_at_threshold_optimistic": {"sensitivity": float(f[yy == 1].mean()), "specificity": float((~f)[yy == 0].mean()), "precision": float(yy[f].mean()), "flagged_fraction": float(f.mean())},
      "nested_estimate": {"sensitivity": 0.793, "specificity": 0.732, "precision": 0.131, "source": "artifacts/results/operating_point_nested.csv"}, "standardised_coefficients": dict(zip(FEATURES, map(float, m[-1].coef_[0]))), "test_split_scored": False,
      "data_sha256": hashlib.sha256((ROOT / "data/raw/stroke_data.csv").read_bytes()).hexdigest(), "dev_ids_sha256": hashlib.sha256(",".join(map(str, sorted(d.id))).encode()).hexdigest(),
      "sklearn": sklearn.__version__, "python": platform.python_version(), "seed": SEED}
    json.dump(meta, open(ROOT / "models/metadata.json", "w"), indent=1); print(json.dumps(meta, indent=1)); print(sc.points_)
if __name__ == "__main__":
    main()
