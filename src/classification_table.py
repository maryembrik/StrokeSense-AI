"""Plain classification results, out-of-fold, development split, same folds. 4 features for all except marked rows."""
import warnings; warnings.filterwarnings("ignore")
import src.experiments as E, src.experiments_shape as S
from src.experiments import *
from src.day1 import age_spline, CORE, FEAT
from src.experiments_leak import Safe
from src.train import pick_threshold
from sklearn.metrics import accuracy_score
E.make = S.make
def cv(fn, X, y):
    P = np.zeros((4, len(y)))
    for i, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(X, y)): P[i // 5, b] = fn(X.iloc[a], y.iloc[a], X.iloc[b])
    return P
def four(n):
    def f(A, yA, B):
        A, B = A[CORE].copy(), B[CORE].copy()
        if n in ("XGBoost", "Random Forest", "Balanced RF", "Random Forest + SMOTE", "Logistic + SMOTE") or n == "Logistic" or n == "HGB mono pairs": A["g"] = B["g"] = "x"  # constant category, carries no information
        return np.asarray(E.make(n, A).fit(A, yA).predict_proba(B))[:, 1]
    return f
def ageonly(A, yA, B): return age_spline().fit(A, yA).predict_proba(B)[:, 1]
def card(A, yA, B): return IntegerScorecard().fit(A[FEAT], yA).predict_proba(B[FEAT])[:, 1]
def art(A, yA, B):
    t = Safe("median").fit(A); Z = t.transform(A); return E.make("HGB mono pairs", Z).fit(Z, yA).predict_proba(t.transform(B))[:, 1]
if __name__ == "__main__":
    X, _, y, _ = load_split(); del _; Y = y.values; n1 = Y.sum(); n0 = (Y == 0).sum(); R = []
    fs = {"Age only": ageonly, **{n: four(n) for n in ["Logistic", "CatBoost", "EBM monotone", "HGB mono pairs", "XGBoost", "Random Forest", "Balanced RF", "Logistic + SMOTE", "Random Forest + SMOTE"]}, "Scorecard": card, "Boosting + median BMI (artifact)": art}
    R.append(dict(model="Say 'no stroke' to everyone", acc50=n0 / len(Y), caught50=0, acc80=np.nan))
    for n, f in fs.items():
        P = cv(f, X, y); rs = []
        for p in P:
            t = pick_threshold(Y, p); g = p >= t; h = p >= .5; tp = (g & (Y == 1)).sum(); fp = (g & (Y == 0)).sum()
            top = np.zeros(len(p), bool); top[np.argsort(-p, kind="stable")[:409]] = 1
            rs.append(dict(acc50=(h == Y).mean(), caught50=(h & (Y == 1)).sum(), caught80=tp, missed=n1 - tp, false_alarms=fp, sens=tp / n1, spec=1 - fp / n0, precision=tp / (tp + fp), npv=(~g & (Y == 0)).sum() / (~g).sum(),
                acc80=(g == Y).mean(), balacc=(tp / n1 + 1 - fp / n0) / 2, f1=2 * tp / (2 * tp + fp + n1 - tp), top10_caught=Y[top].mean() and Y[top].sum(), pr=average_precision_score(Y, p), roc=roc_auc_score(Y, p), brier=brier_score_loss(Y, p), mean_pred=p.mean()))
        R.append(dict(model=n, **pd.DataFrame(rs).mean()))
    T = pd.DataFrame(R); T.to_csv(ROOT / "artifacts/results/classification_table.csv", index=False); pd.set_option("display.width", 300); print(T.round(3).to_string())
