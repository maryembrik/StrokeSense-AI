"""Does constant BMI imputation smuggle the missing-BMI signal into tree models? Development split only."""
import src.experiments as E, src.experiments_shape as S
from src.experiments import *
from src.features import _basis
E.make = S.make
class Safe(StrokeFeatures):
    """mode 'draw': age-conditional random draw; mode 'drop': no BMI at all; mode 'median': constant fill."""
    def __init__(self, mode="median"): self.mode = mode
    def fit(self, X, y=None):
        for k in ("bmi_flag", "peer", "smoke_split", "burden"): setattr(self, k, False)
        self.bmi = "median"; super().fit(X, y)
        ok = X.bmi.notna(); self.resid_ = (np.log(X.loc[ok, "bmi"]) - self.bmi_lr_.predict(_basis(X.loc[ok, "age"]))).values
        return self
    def transform(self, X):
        o = super().transform(X); m = X["bmi"].isna()
        if self.mode == "draw" and m.any():
            r = np.random.default_rng(int(1000 * X.age.sum()) % 2**32)
            o.loc[m, "bmi"] = np.exp(self.bmi_lr_.predict(_basis(X.loc[m, "age"])) + r.choice(self.resid_, m.sum())).round(1)
        return o.drop(columns="bmi") if self.mode == "drop" else o
if __name__ == "__main__":
    X, _, y, _ = load_split(); E.StrokeFeatures = Safe
    sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(X, y))
    R = []
    for n in ["Age only (logistic)", "Logistic", "Logistic + splines", "XGBoost", "HistGradientBoosting", "HGB", "HGB mono", "HGB mono additive", "HGB mono pairs", "HGB mono pairs small", "Balanced RF", "CatBoost", "EBM monotone", "EBM"]:
        for mode in ["median", "draw", "drop"]:
            if n in ("CatBoost", "EBM", "EBM monotone") and mode == "median": continue
            out = [one(n, dict(mode=mode), X, y, a, b, i) for i, (a, b) in enumerate(sp)]
            R += [dict(label=n, mode=mode, **o) for o in out]
    R = pd.DataFrame(R); old = pd.read_csv(ROOT / "artifacts/results/models_cv.csv")
    R = pd.concat([R, old[old.label.isin(["CatBoost", "EBM", "EBM monotone"])].assign(mode="median")])
    R.to_csv(ROOT / "artifacts/results/leak_cv.csv", index=False)
    print(R.pivot_table(index="label", columns="mode", values=["pr_auc", "roc_auc", "age_matched_auc", "net_benefit_10"]).round(4).to_string())
    P = R[R["mode"] == "drop"].pivot(index="fold", columns="label", values="pr_auc")
    for ref in ["Age only (logistic)", "Logistic"]:
        for c in P:
            d = P[c] - P[ref]; print(c, "-", ref, round(d.mean(), 4), "+/-", round(2 * d.std() * np.sqrt(1/20 + .25), 4), (d > 0).sum())
