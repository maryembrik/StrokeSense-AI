"""Phase-1 experiments. Uses the 80% development split only; the 20% test split is never touched here."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_pipe
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from src.features import *
from src.scorecard import IntegerScorecard

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

def load_split():
    d = pd.read_csv(ROOT / "data/raw/stroke_data.csv")
    return train_test_split(d[RAW], d[TARGET], test_size=0.2, stratify=d[TARGET], random_state=SEED)

def nums(X): return list(X.select_dtypes("number"))
def cats(X): return list(X.select_dtypes(exclude="number"))
def enc(X, spline=False):
    n = nums(X); s = [c for c in NUM if c in n]; o = [c for c in n if c not in s]
    num = make_pipeline(SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), StandardScaler()) if spline else StandardScaler()
    return make_column_transformer((num, s), (StandardScaler(), o),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats(X)))

MONO_UP = {"age", "avg_glucose_level", "hypertension", "heart_disease", "burden"}

def make(name, X):
    c = cats(X)
    if name == "Age only (logistic)":
        return make_pipeline(make_column_transformer((StandardScaler(), ["age"])), LogisticRegression())
    if name == "Logistic": return make_pipeline(enc(X), LogisticRegression(C=.1, max_iter=2000))
    if name == "Logistic + splines": return make_pipeline(enc(X, True), LogisticRegression(C=.1, max_iter=2000))
    if name == "Logistic + SMOTE": return imb_pipe(enc(X), SMOTE(random_state=0), LogisticRegression(C=.1, max_iter=2000))
    if name == "Random Forest": return make_pipeline(enc(X), RandomForestClassifier(400, min_samples_leaf=5, class_weight="balanced_subsample", random_state=0))
    if name == "Random Forest + SMOTE": return imb_pipe(enc(X), SMOTE(random_state=0), RandomForestClassifier(400, min_samples_leaf=5, random_state=0))
    if name == "Balanced RF": return make_pipeline(enc(X), BalancedRandomForestClassifier(400, min_samples_leaf=5, sampling_strategy="all", replacement=True, bootstrap=False, random_state=0))
    if name == "XGBoost": return make_pipeline(enc(X), XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.04, subsample=.8, colsample_bytree=.8, min_child_weight=5, reg_lambda=5, n_jobs=1, random_state=0))
    if name == "CatBoost": return CatBoostClassifier(iterations=300, depth=4, learning_rate=.03, l2_leaf_reg=6, cat_features=c, verbose=0, thread_count=1, random_seed=0, allow_writing_files=False)
    if name == "HistGradientBoosting": return make_pipeline(enc(X), HistGradientBoostingClassifier(max_iter=120, learning_rate=.04, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=5, random_state=0))
    if name.startswith("EBM"):
        k = dict(outer_bags=8, interactions=0 if "additive" in name or "monotone" in name else 5, max_rounds=3000,
                 learning_rate=.015, min_samples_leaf=20, n_jobs=1, random_state=0)
        if "monotone" in name: k["monotone_constraints"] = [1 if f in MONO_UP else 0 for f in X.columns]
        return ExplainableBoostingClassifier(feature_types=["nominal" if f in c else "continuous" for f in X.columns], **k)
    if name == "Integer scorecard": return IntegerScorecard()
    raise KeyError(name)

def age_matched_auc(y, p, age, w=3):
    """P(model ranks a stroke patient above a non-stroke patient of nearly the same age). 0.5 = only knows age."""
    y, p, age = map(np.asarray, (y, p, age)); num = den = 0
    for i in np.flatnonzero(y == 1):
        m = (y == 0) & (np.abs(age - age[i]) <= w)
        num += (p[i] > p[m]).sum() + .5 * (p[i] == p[m]).sum(); den += m.sum()
    return num / den

def net_benefit(y, p, t):
    f = p >= t; n = len(y)
    return ((f & (y == 1)).sum() - (f & (y == 0)).sum() * t / (1 - t)) / n

def one(name, fs, X, y, tr, te, fold):
    t0 = time.time(); f = StrokeFeatures(**fs).fit(X.iloc[tr]); A, B = f.transform(X.iloc[tr]), f.transform(X.iloc[te])
    m = make(name, A).fit(A, y.iloc[tr]); p = np.asarray(m.predict_proba(B))[:, 1]; yt = y.iloc[te].values
    top = np.zeros(len(p), bool); top[np.argsort(-p, kind="stable")[: int(round(.2 * len(p)))]] = True
    return dict(fold=fold, pr_auc=average_precision_score(yt, p), roc_auc=roc_auc_score(yt, p),
        age_matched_auc=age_matched_auc(yt, p, X.iloc[te].age), brier=brier_score_loss(yt, p),
        mean_pred=p.mean(), recall_top20=(yt[top].sum() / yt.sum()), net_benefit_10=net_benefit(yt, p, .10), seconds=time.time() - t0)

def run(jobs, X, y, rep=4):
    sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=SEED).split(X, y))
    tasks = [(l, n, fs, i, s) for l, n, fs in jobs for i, s in enumerate(sp)]
    out = Parallel(n_jobs=-2)(delayed(one)(n, fs, X, y, s[0], s[1], i) for l, n, fs, i, s in tasks)
    return pd.DataFrame([dict(label=t[0], **o) for t, o in zip(tasks, out)])

BASE = dict(bmi="median")
MODELS = ["Age only (logistic)", "Logistic", "Logistic + SMOTE", "Logistic + splines", "Random Forest", "Random Forest + SMOTE", "Balanced RF",
          "XGBoost", "CatBoost", "HistGradientBoosting", "EBM", "EBM additive", "EBM monotone", "Integer scorecard"]
ABL = {"A raw + median BMI": {}, "B peer BMI imputation": dict(bmi="peer"), "C + BMI-missing flag": dict(bmi_flag=True),
       "D + peer-relative features": dict(bmi="peer", peer=True), "E + smoking split": dict(smoke_split=True),
       "F + burden count": dict(burden=True), "G all except flag": dict(bmi="peer", peer=True, smoke_split=True, burden=True)}
if __name__ == "__main__":
    X, _, y, _ = load_split()
    a = run([(m, m, BASE) for m in MODELS], X, y); a.to_csv(ROOT / "artifacts/results/models_cv.csv", index=False)
    b = run([(f"{k} | {m}", m, v) for m in ["Logistic", "EBM monotone"] for k, v in ABL.items()], X, y)
    b.to_csv(ROOT / "artifacts/results/ablation_cv.csv", index=False)
    for r in (a, b): print(r.groupby("label", sort=False).mean().drop(columns="fold").round(4).to_string(), "\n")
    print(a.groupby("label").pr_auc.std().round(3).to_dict())
    for k in ["CatBoost","XGBoost","EBM monotone","Logistic","Integer scorecard","Logistic + splines"]:
        for q in ["pr_auc","age_matched_auc","brier"]:
            dd = a[a.label==k].set_index("fold")[q]-a[a.label=="Age only (logistic)"].set_index("fold")[q]
            print(k,q,round(dd.mean(),4),"wins",(dd>0).sum(),"/",len(dd))
    print(IntegerScorecard().fit(StrokeFeatures().fit_transform(X), y).points_)
