"""Adversarial review of Day 1. Development split only (test parts discarded)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pathlib import Path
R = Path(__file__).resolve().parents[1]
d = pd.read_csv(R / "data/raw/stroke_data.csv"); d, _ = train_test_split(d, test_size=.2, stratify=d.stroke, random_state=42); del _
d = d.reset_index(drop=True); y = d.stroke; F4 = ["age", "avg_glucose_level", "hypertension", "heart_disease"]; C = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]; F9 = F4 + C
sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42).split(d, y))
def prep(F): return make_column_transformer((StandardScaler(), [f for f in F if f in F4 + ["bmi"]]), (OneHotEncoder(handle_unknown="ignore", sparse_output=False), [f for f in F if f in C]))
def am(yt, p, a):
    yt, a = np.asarray(yt), np.asarray(a); n = m = 0
    for i in np.where(yt == 1)[0]:
        k = (yt == 0) & (abs(a - a[i]) <= 3); n += (p[i] > p[k]).sum() + .5 * (p[i] == p[k]).sum(); m += k.sum()
    return n / m
L = lambda F: make_pipeline(prep(F), LogisticRegression(C=.1, max_iter=3000))
print("== 1 independent recompute")
for F in (F4, F9):
    r = [(AP(y[b], p), AUC(y[b], p), am(y[b], p, d.age[b])) for a, b in sp for p in [L(F).fit(d.loc[a, F], y[a]).predict_proba(d.loc[b, F])[:, 1]]]
    print(len(F), np.mean(r, 0).round(4))
print("== 2 permuted-label sanity (expect ROC .5, PR ~.049)")
rng = np.random.default_rng(1); q = []
for k in range(10):
    yp = pd.Series(rng.permutation(y)); a, b = sp[k]; p = L(F4).fit(d.loc[a, F4], yp[a]).predict_proba(d.loc[b, F4])[:, 1]; q.append((AUC(yp[b], p), AP(yp[b], p)))
print(np.mean(q, 0).round(3))
print("== 3 nested tuning, 5-fold outer x1 repeat x 12 configs, 3-fold inner, PR-AUC")
G = {"HGB": (lambda F: make_pipeline(prep(F), HistGradientBoostingClassifier(random_state=0)), {"histgradientboostingclassifier__learning_rate": [.01, .02, .04, .08], "histgradientboostingclassifier__max_iter": [50, 100, 200, 300], "histgradientboostingclassifier__max_leaf_nodes": [2, 3, 4, 8], "histgradientboostingclassifier__min_samples_leaf": [20, 40, 80, 150], "histgradientboostingclassifier__l2_regularization": [0, 5, 30]}),
     "XGB": (lambda F: make_pipeline(prep(F), XGBClassifier(n_jobs=1, random_state=0, subsample=.8)), {"xgbclassifier__learning_rate": [.01, .02, .05], "xgbclassifier__n_estimators": [50, 100, 200, 300], "xgbclassifier__max_depth": [1, 2, 3], "xgbclassifier__min_child_weight": [3, 10, 30], "xgbclassifier__reg_lambda": [1, 10, 50], "xgbclassifier__colsample_bytree": [.5, .8, 1]}),
     "CatBoost": (lambda F: make_pipeline(prep(F), CatBoostClassifier(verbose=0, thread_count=1, random_seed=0, allow_writing_files=False)), {"catboostclassifier__learning_rate": [.01, .03, .06], "catboostclassifier__iterations": [80, 150, 300], "catboostclassifier__depth": [2, 3, 4], "catboostclassifier__l2_leaf_reg": [3, 10, 30]}),
     "Logistic": (L, {"logisticregression__C": [.003, .01, .03, .1, .3, 1, 10]})}
rows = []
for F in (F4, F9):
    for n, (mk, g) in G.items():
        for i, (a, b) in enumerate(sp[:10]):
            s = RandomizedSearchCV(mk(F), g, n_iter=min(12, 7 if n == "Logistic" else 12), scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=i), random_state=i, n_jobs=-2).fit(d.loc[a, F], y[a])
            p = s.predict_proba(d.loc[b, F])[:, 1]; rows.append(dict(m=n, F=len(F), fold=i, pr=AP(y[b], p), roc=AUC(y[b], p), am=am(y[b], p, d.age[b])))
T = pd.DataFrame(rows); T.to_csv(R / "artifacts/results/nested_tuning.csv", index=False); print(T.groupby(["F", "m"])[["pr", "roc", "am"]].mean().round(4).to_string())
P = T.pivot(index="fold", columns=["m", "F"], values="pr")
for c in P: 
    e = P[c] - P[("Logistic", 4)]; print(c, round(e.mean(), 4), "+/-", round(2 * e.std() * np.sqrt(1/10 + .25), 4), int((e > 0).sum()), "/10")
print("== 4 BMI mechanism: where does the gain live?")
F10 = F9 + ["bmi"]; miss = d.bmi.isna().values; out = []
H = lambda F: make_pipeline(prep(F), HistGradientBoostingClassifier(max_iter=120, learning_rate=.04, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=5, random_state=0))
for i, (a, b) in enumerate(sp):
    A, B = d.loc[a].copy(), d.loc[b].copy(); med = A.bmi.median(); ms = miss[b]; ob = ~ms; r = {}
    for nm, sd, use in [("no BMI", 0, 0), ("median", 0, 1), ("median + jitter sd=4", 4, 1), ("constant 28.137 (unique)", -1, 1)]:
        A2, B2 = A.copy(), B.copy(); g = np.random.default_rng(i)
        for D in (A2, B2):
            k = D.bmi.isna(); D.loc[k, "bmi"] = (28.137 if sd < 0 else med + (g.normal(0, sd, k.sum()) if sd else 0))
        F = F10 if use else F9; p = H(F).fit(A2[F], y[a]).predict_proba(B2[F])[:, 1]
        r.update({f"{nm}|all": AP(y[b], p), f"{nm}|BMI observed": AP(y[b][ob], p[ob]), f"{nm}|risk of blank": p[ms].mean(), f"{nm}|risk of observed": p[ob].mean()})
    out.append(r)
o = pd.DataFrame(out).mean().round(4); o.index = pd.MultiIndex.from_tuples([tuple(s.split("|")) for s in o.index]); o = o.unstack(); o.to_csv(R / "artifacts/results/bmi_mechanism.csv"); print(o.to_string())
print("rows exactly at median bmi among observed:", int((d.bmi == d.bmi.median()).sum()), "missing:", miss.sum())
print("== 5 fixed vs drop-children population (age>=40)"); 
r = []
for a, b in sp:
    k = (d.age[b] >= 40).values; p = L(F4).fit(d.loc[a, F4], y[a]).predict_proba(d.loc[b, F4])[:, 1]; r.append((AUC(y[b][k], p[k]), AUC(y[b][k], d.age[b][k])))
print("ROC age>=40: model, age", np.mean(r, 0).round(4))
