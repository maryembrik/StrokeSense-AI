"""Overfitting / underfitting / leakage checks. Development split only."""
import warnings; warnings.filterwarnings("ignore")
import src.experiments as E, src.experiments_shape as S
from src.experiments import *
from src.day1 import CORE, FEAT, lr
from src.classification_table import four
E.make = S.make
X, _, y, _ = load_split(); del _
sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(X, y))
def sc(f, A, yA, B, yB):
    return [(average_precision_score(v, p), roc_auc_score(v, p)) for U, v in ((A, yA), (B, yB)) for p in [f(A, yA, U)]]
L = lambda c, C=.1: (lambda A, v, B: lr(c, C).fit(A, v).predict_proba(B)[:, 1])
print("== train vs unseen (PR, ROC)")
for n, f in {"Logistic 4": L(CORE), "Logistic 9": L(FEAT), "XGBoost": four("XGBoost"), "HGB mono pairs": four("HGB mono pairs"), "CatBoost": four("CatBoost"), "Random Forest": four("Random Forest")}.items():
    r = np.mean([sc(f, X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b]) for a, b in sp[:10]], 0); print(f"{n:16s} train PR {r[0][0]:.3f} ROC {r[0][1]:.3f} | unseen PR {r[1][0]:.3f} ROC {r[1][1]:.3f}")
print("== regularisation, logistic 4")
for C in (.001, .01, .1, 1, 100): 
    r = np.mean([sc(L(CORE, C), X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b]) for a, b in sp], 0); print(C, r.round(3).tolist())
print("== learning curve, logistic 4: fraction of training fold")
for fr in (.1, .2, .4, .6, .8, 1):
    o = []
    for i, (a, b) in enumerate(sp):
        k = a if fr == 1 else train_test_split(a, train_size=fr, stratify=y.iloc[a], random_state=i)[0]; o.append(sc(L(CORE), X.iloc[k], y.iloc[k], X.iloc[b], y.iloc[b]))
    r = np.mean(o, 0); print(fr, "train", r[0].round(3), "unseen", r[1].round(3))
print("== underfitting probe: add interactions/splines to the 4 features")
from sklearn.preprocessing import PolynomialFeatures
for nm, pre in {"squares+interactions": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler()), "splines": make_column_transformer((make_pipeline(SplineTransformer(n_knots=5, extrapolation="linear"), StandardScaler()), CORE[:2]), ("passthrough", CORE[2:]))}.items():
    f = lambda A, v, B: make_pipeline(pre, LogisticRegression(C=.1, max_iter=3000)).fit(A[CORE], v).predict_proba(B[CORE])[:, 1]
    print(nm, np.mean([sc(f, X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b]) for a, b in sp], 0).round(3).tolist())
print("== leakage"); _a, T, _b, _c = load_split(); D = pd.read_csv(ROOT / "data/raw/stroke_data.csv")
k = lambda d: set(map(tuple, d[RAW].fillna(-1).astype(str).values)); print("identical feature rows dev/test:", len(k(X) & k(T)), "| within dev:", len(X) - len(k(X)))
print("id alone AUC (dev):", round(roc_auc_score(y, D.loc[X.index, "id"]), 3))
m = np.polyfit(*[np.array(v) for v in (np.arange(3), np.arange(3))], 1) is not None
st = lr(CORE); ps = [st.fit(X.iloc[a], y.iloc[a])[0].transformers_[0][1].mean_[0] for a, b in sp[:5]]; print("scaler means differ by fold (fit inside folds):", np.round(ps, 3))
print("== permutation 200 fits"); rng = np.random.default_rng(0); z = []
for i in range(200):
    v = pd.Series(rng.permutation(y.values), index=y.index); a, b = sp[i % 20]; z.append(roc_auc_score(v.iloc[b], L(CORE)(X.iloc[a], v.iloc[a], X.iloc[b])))
print(round(np.mean(z), 3), round(np.std(z), 3), "max", round(max(z), 3))
