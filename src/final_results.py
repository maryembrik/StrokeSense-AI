"""FINAL fair benchmark. Every model: regularised search space, hyperparameters AND feature set (4 or 9, no BMI) AND threshold
chosen inside each outer training fold (inner 5-fold, PR-AUC). Outer 4 x stratified 5-fold. Development split only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, brier_score_loss
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from src.experiments import load_split, ROOT, SEED
from src.train import pick_threshold
F4 = ["age", "avg_glucose_level", "hypertension", "heart_disease"]; C = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]; F9 = F4 + C
def prep(F): return make_column_transformer((StandardScaler(), [f for f in F if f in F4]), (OneHotEncoder(handle_unknown="ignore", sparse_output=False), [f for f in F if f in C]))
def P(F, m): return make_pipeline(prep(F), m)
Ch = lambda r, *o: o[r.integers(len(o))]
def space(name, r):
    if name == "Logistic": return dict(C=Ch(r, .003, .01, .03, .1, .3, 1))
    if name in ("Random Forest", "Balanced RF"): return dict(max_depth=Ch(r, 2, 3, 4, 6), min_samples_leaf=Ch(r, 10, 20, 40, 80), max_features=Ch(r, 1, 2, .5))
    if name == "XGBoost": return dict(max_depth=Ch(r, 1, 2, 3), learning_rate=Ch(r, .01, .03, .05), n_estimators=Ch(r, 60, 120, 200), min_child_weight=Ch(r, 5, 15, 40), reg_lambda=Ch(r, 5, 20, 80), subsample=.8, colsample_bytree=Ch(r, .5, 1))
    if name == "CatBoost": return dict(depth=Ch(r, 1, 2, 3), learning_rate=Ch(r, .01, .03), iterations=Ch(r, 80, 150, 250), l2_leaf_reg=Ch(r, 10, 40, 100))
    if name == "HGB monotone": return dict(max_leaf_nodes=Ch(r, 2, 3, 4), learning_rate=Ch(r, .01, .03, .05), max_iter=Ch(r, 60, 120, 200), min_samples_leaf=Ch(r, 40, 80, 150), l2_regularization=Ch(r, 5, 30, 100))
    if name == "EBM monotone": return dict(learning_rate=Ch(r, .005, .015), max_rounds=Ch(r, 300, 800, 2000), min_samples_leaf=Ch(r, 20, 60), max_leaves=Ch(r, 2, 3))
def build(name, F, k):
    if name == "Logistic": return P(F, LogisticRegression(max_iter=3000, **k))
    if name == "Random Forest": return P(F, RandomForestClassifier(200, random_state=0, **k))
    if name == "Balanced RF": return P(F, BalancedRandomForestClassifier(200, sampling_strategy="all", replacement=True, bootstrap=False, random_state=0, **k))
    if name == "XGBoost": return P(F, XGBClassifier(n_jobs=1, random_state=0, **k))
    if name == "CatBoost": return P(F, CatBoostClassifier(verbose=0, thread_count=1, random_seed=0, allow_writing_files=False, **k))
    if name == "HGB monotone":
        pre = prep(F)
        return make_pipeline(pre, HistGradientBoostingClassifier(random_state=0, monotonic_cst=None, **k)) if False else _Mono(F, k)
    if name == "EBM monotone": return ExplainableBoostingClassifier(feature_types=["nominal" if f in C else "continuous" for f in F], interactions=0, outer_bags=4, n_jobs=1, random_state=0, monotone_constraints=[int(f in F4) for f in F], **k)
class _Mono:
    def __init__(s, F, k): s.F, s.k = F, k
    def fit(s, X, y):
        s.pre = prep(s.F).fit(X); Z = s.pre.transform(X); n = sum(f in F4 for f in s.F)
        s.m = HistGradientBoostingClassifier(random_state=0, monotonic_cst=[1] * n + [0] * (Z.shape[1] - n), **s.k).fit(Z, y); return s
    def predict_proba(s, X): return s.m.predict_proba(s.pre.transform(X))
def fp(name, F, k, A, yA, *Bs):
    m = build(name, F, k).fit(A[F], yA); return [np.asarray(m.predict_proba(B[F]))[:, 1] for B in Bs]
def fixed(F):
    return lambda A, yA, *Bs: (lambda m: [m.predict_proba(B[F])[:, 1] for B in Bs])(P(F, LogisticRegression(C=.1, max_iter=3000)).fit(A[F], yA))
def ageonly(A, yA, *Bs):
    m = make_pipeline(SplineTransformer(n_knots=5, extrapolation="linear"), LogisticRegression(max_iter=3000)).fit(A[["age"]].values, yA); return [m.predict_proba(B[["age"]].values)[:, 1] for B in Bs]
FIX = {"Age only": ageonly, "Logistic 4 (selected)": fixed(F4), "Logistic 9": fixed(F9)}
def job(name, i, a, b, X, y):
    import warnings; warnings.filterwarnings("ignore")
    A, yA, B = X.iloc[a], y.iloc[a], X.iloc[b]; S = list(StratifiedKFold(5, shuffle=True, random_state=500 + i).split(A, yA)); info = None
    def oof(f):
        q = np.zeros(len(a))
        for u, v in S: q[v] = f(A.iloc[u], yA.iloc[u], A.iloc[v])[0]
        return q
    if name in FIX: f = FIX[name]; q = oof(f)
    else:
        r = np.random.default_rng(i); best = None; n = 6 if name == "EBM monotone" else 12; seen = set()
        for c in range(n):
            F = F4 if c % 2 == 0 else F9; k = space(name, r)
            if (len(F), str(k)) in seen: continue
            seen.add((len(F), str(k))); g = lambda U, v, *W, F=F, k=k: fp(name, F, k, U, v, *W); z = oof(g); s = np.mean([AP(yA.iloc[v], z[v]) for _, v in S])
            if best is None or s > best[0]: best = (s, g, z, len(F), k)
        _, f, q, nf, k = best; info = nf
    pt, pv = f(A, yA, A, B); t = pick_threshold(yA.values, q); yB = y.iloc[b].values
    return dict(model=name, i=i, pr_train=AP(yA, pt), roc_train=AUC(yA, pt), pr=AP(yB, pv), roc=AUC(yB, pv), brier=brier_score_loss(yB, pv), nf=info), b, pv, pv >= t
if __name__ == "__main__":
    X, _, y, _ = load_split(); del _; Y = y.values; n = len(Y); sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(X, y))
    names = list(FIX) + ["Logistic", "HGB monotone", "XGBoost", "CatBoost", "Random Forest", "Balanced RF", "EBM monotone"]
    R = Parallel(n_jobs=-2, verbose=0)(delayed(job)(m, i, a, b, X, y) for m in names for i, (a, b) in enumerate(sp))
    Pp = {m: np.zeros((4, n)) for m in names}; G = {m: np.zeros((4, n)) for m in names}; rows = []
    for d, b, p, g in R: rows.append(d); Pp[d["model"]][d["i"] // 5, b] = p; G[d["model"]][d["i"] // 5, b] = g
    D = pd.DataFrame(rows); D.to_csv(ROOT / "artifacts/results/final_folds.csv", index=False); np.savez_compressed(ROOT / "artifacts/results/final_oof.npz", **{f"{m}|p": v for m, v in Pp.items()}, **{f"{m}|g": v for m, v in G.items()})
    T = D.groupby("model", sort=False)[["pr_train", "pr", "roc_train", "roc", "brier"]].mean(); T.insert(2, "pr_sd", D.groupby("model").pr.std()); T["gap_pr"] = T.pr_train - T.pr; T["gap_roc"] = T.roc_train - T.roc; T["uses4"] = D.groupby("model").nf.apply(lambda s: (s == 4).sum())
    for m in names:
        f = G[m].mean(0); tp, fpn = f[Y == 1].sum(), f[Y == 0].sum(); T.loc[m, ["tp", "fp", "sens", "spec", "prec", "npv", "acc", "f1", "mean_pred"]] = [tp, fpn, tp/199, 1-fpn/(n-199), tp/(tp+fpn), (n-199-fpn)/(n-tp-fpn), (tp+n-199-fpn)/n, 2*tp/(2*tp+fpn+199-tp), Pp[m].mean()]
    PV = D.pivot(index="i", columns="model", values="pr"); RV = D.pivot(index="i", columns="model", values="roc"); nbse = np.sqrt(1/20 + .25)
    for m in names:
        for V, l in ((PV, "pr"), (RV, "roc")):
            e = V[m] - V["Logistic 4 (selected)"]; T.loc[m, f"d{l}"] = e.mean(); T.loc[m, f"d{l}_2se"] = 2 * e.std() * nbse
    T.to_csv(ROOT / "artifacts/results/FINAL_RESULTS.csv"); pd.set_option("display.width", 400, "display.max_columns", 50); print(T.round(3).to_string())
    g = G["Logistic 4 (selected)"].mean(0); p = Pp["Logistic 4 (selected)"]; rr = []
    for lab, s in {"gender": X.gender, "residence": X.Residence_type, "age": pd.cut(X.age, [-1, 40, 60, 70, 83], labels=["<=40", "41-60", "61-70", "71+"]), "hypertension": X.hypertension.map({0: "no", 1: "yes"}), "smoking": X.smoking_status, "BMI recorded": X.bmi.notna()}.items():
        for v in pd.unique(s):
            k = (s == v).values; c = k & (Y == 1); z = k & (Y == 0)
            if c.sum() < 5: rr.append(dict(group=lab, value=v, n=k.sum(), strokes=c.sum())); continue
            se = g[c].mean(); w = 1.96; nn = c.sum(); ctr = (se + w*w/(2*nn)) / (1 + w*w/nn); h = w*np.sqrt(se*(1-se)/nn + w*w/(4*nn*nn)) / (1 + w*w/nn)
            rr.append(dict(group=lab, value=v, n=k.sum(), strokes=nn, sens=se, lo=ctr-h, hi=ctr+h, flagged=g[k].mean(), precision=g[c].sum()/g[k].sum(), roc=np.mean([AUC(Y[k], q[k]) for q in p]), mean_pred=p[:, k].mean(), observed=Y[k].mean()))
    U = pd.DataFrame(rr); U.to_csv(ROOT / "artifacts/results/subgroups.csv", index=False); print(U.round(3).to_string()); print("DONE")
