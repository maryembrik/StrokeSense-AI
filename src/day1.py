"""Day-1 final evidence. Development split only: load_split()'s test parts are discarded immediately.
Selection rule, written before running: among BMI-free models take the highest mean PR-AUC as reference, then the
SIMPLEST model whose PR-AUC is not worse than it by more than one corrected SE. An age-only model is a benchmark, not a candidate."""
import json
import src.experiments as E, src.experiments_shape as S
from src.experiments import *
from src.experiments_leak import Safe
from sklearn.base import clone
E.make = S.make
OUT = ROOT / "artifacts/results"
FEAT = ["age", "avg_glucose_level", "hypertension", "heart_disease", "gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
CORE = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

def lr(cols, C=.1):
    n = [c for c in cols if c in NUM + BIN]; k = [c for c in cols if c in CAT]
    return make_pipeline(make_column_transformer((StandardScaler(), n), (OneHotEncoder(handle_unknown="ignore", sparse_output=False), k)), LogisticRegression(C=C, max_iter=3000))
def age_spline():
    return make_pipeline(make_column_transformer((SplineTransformer(n_knots=5, extrapolation="linear"), ["age"])), LogisticRegression(C=1, max_iter=3000))
class FlagSafe(Safe):
    def transform(self, X):
        o = super().transform(X); o["bmi_missing"] = X.bmi.isna().astype(int); return o
class Raw:  # full-data passthrough, no BMI
    def fit(self, X, y=None): return self
    def transform(self, X): return X[FEAT].copy()

CAND = {"Age spline only": age_spline, "Logistic (9 features)": lambda: lr(FEAT), "Logistic (4 clinical)": lambda: lr(CORE),
        "Scorecard (BMI-free)": IntegerScorecard}
def oof(fit_predict, X, y, seed, rep):
    P = np.full((rep, len(y)), np.nan); rows = []
    for i, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=seed).split(X, y)):
        p = fit_predict(X.iloc[a], y.iloc[a], X.iloc[b]); P[i // 5, b] = p; yt = y.iloc[b].values
        rows.append(dict(fold=i, pr_auc=average_precision_score(yt, p), roc_auc=roc_auc_score(yt, p), age_matched_auc=age_matched_auc(yt, p, X.iloc[b].age),
                         brier=brier_score_loss(yt, p), mean_pred=p.mean(), net_benefit_10=net_benefit(yt, p, .1)))
    return P, pd.DataFrame(rows)
def simple(mk):
    def f(A, yA, B): return mk().fit(A[FEAT], yA).predict_proba(B[FEAT])[:, 1]
    return f
def bench(name, mode, flag=False):
    def f(A, yA, B):
        t = (FlagSafe if flag else Safe)(mode).fit(A); Z = t.transform(A)
        return np.asarray(E.make(name, Z).fit(Z, yA).predict_proba(t.transform(B)))[:, 1]
    return f
def calib(y, p):
    z = np.log(np.clip(p, 1e-5, 1-1e-5) / (1 - np.clip(p, 1e-5, 1-1e-5)))[:, None]
    m = LogisticRegression(C=1e6).fit(z, y); q = pd.qcut(p, 10, duplicates="drop"); g = pd.DataFrame(dict(y=y, p=p)).groupby(q, observed=True)
    return dict(slope=m.coef_[0][0], intercept=m.intercept_[0], ece=float((g.size() / len(y) * (g.y.mean() - g.p.mean()).abs()).sum()),
                mean_pred=p.mean(), q50=np.median(p), q95=np.quantile(p, .95), q99=np.quantile(p, .99))

if __name__ == "__main__":
    warnings.filterwarnings("ignore"); X, _, y, _ = load_split(); del _
    old = pd.read_csv(OUT / "leak_cv.csv"); R = [old]; OO = {}
    for n, mk in CAND.items():
        P, r = oof(simple(mk), X, y, SEED, 4); OO[n] = P; R.append(r.assign(label=n, mode="drop"))
    for n, mode, flag in [("HGB mono pairs", "drop", 0), ("HGB mono pairs", "median", 0), ("Logistic", "median", 0), ("Logistic", "median", 1), ("HGB mono pairs", "median", 1), ("XGBoost", "median", 1),
                          ("XGBoost", "median", 0), ("XGBoost", "drop", 0), ("Logistic + SMOTE", "drop", 0), ("Balanced RF", "drop", 0)]:
        P, r = oof(bench(n, mode, flag), X, y, SEED, 4); OO[f"{n} | {mode}{'+flag' if flag else ''}"] = P
        if flag: R.append(r.assign(label=n, mode="median+flag"))
    for n in ["CatBoost", "EBM monotone"]:
        OO[f"{n} | drop"], _r = oof(bench(n, "drop"), X, y, SEED, 1)
    A = pd.concat(R); A.to_csv(OUT / "benchmark_all.csv", index=False)
    F = A[A["mode"] == "drop"]; T = F.groupby("label")[["pr_auc", "roc_auc", "age_matched_auc", "brier", "net_benefit_10"]].mean(); T["pr_sd"] = F.groupby("label").pr_auc.std()
    P = F.pivot(index="fold", columns="label", values="pr_auc"); best = T.pr_auc.idxmax()
    d = P.sub(P[best], axis=0); T["vs_best"] = d.mean(); T["se"] = d.std() * np.sqrt(1/20 + .25); T["within_1se"] = T.vs_best >= -T.se
    da = P.sub(P["Age spline only"], axis=0); T["vs_age_spline"] = da.mean(); T["2se_age"] = 2 * da.std() * np.sqrt(1/20 + .25)
    T = T.sort_values("pr_auc", ascending=False); T.to_csv(OUT / "final_benchmark.csv"); print(T.round(4).to_string()); print(A[A["mode"]=="median+flag"].groupby("label").pr_auc.mean())
    c = pd.DataFrame({k: calib(y.values, v.mean(0)) for k, v in OO.items()}).T
    c.to_csv(OUT / "calibration.csv"); print(c.round(3).to_string())
    np.savez_compressed(OUT / "oof.npz", y=y.values, age=X.age.values, **{k.replace("|", "~"): v.mean(0) for k, v in OO.items()})
    # persistence: fresh fold assignments
    rs = []
    for seed in (7, 101, 2024):
        for n in ["Logistic", "HGB mono pairs", "XGBoost"]:
            rs.append(dict(seed=seed, label=n, **{m: oof(bench(n, m), X, y, seed, 2)[1].pr_auc.mean() for m in ("median", "drop")}))
    rs = pd.DataFrame(rs); rs.to_csv(OUT / "persistence.csv", index=False); print(rs.round(4).to_string())
    # bootstrap CI over people, OOF predictions
    rng = np.random.default_rng(0); yy, ag = y.values, X.age.values; B = []
    use = {"Age spline only": OO["Age spline only"], **{k: OO[k] for k in list(CAND)[1:]}}
    for _b in range(500):
        i = np.r_[rng.choice(np.flatnonzero(yy == 1), 199), rng.choice(np.flatnonzero(yy == 0), (yy == 0).sum())]
        for k, v in use.items():
            p = v.mean(0)[i]; B.append(dict(b=_b, m=k, pr=average_precision_score(yy[i], p), roc=roc_auc_score(yy[i], p), am=age_matched_auc(yy[i], p, ag[i])))
    B = pd.DataFrame(B); ci = B.groupby("m")[["pr", "roc", "am"]].quantile([.025, .5, .975]).unstack().round(3); ci.to_csv(OUT / "bootstrap_ci.csv"); print(ci.to_string())
    pv = B.pivot(index="b", columns="m", values=["pr", "am"])
    for k in list(CAND)[1:]:
        for q in ("pr", "am"): print("paired vs age spline", k, q, np.quantile(pv[q][k] - pv[q]["Age spline only"], [.025, .5, .975]).round(3))
    cnt = [int(((yy == 0) & (np.abs(ag - ag[i]) <= 3)).sum()) for i in np.flatnonzero(yy == 1)]; print("age-matched pairs", sum(cnt), "cases without match", sum(k == 0 for k in cnt))
    # stability: L1 selection on 200 bootstraps; tuning of C
    Z = lr(FEAT)[0].fit(X[FEAT]); M = Z.transform(X[FEAT]); nm = [s.split("__")[1] for s in Z.get_feature_names_out()]; sel = []
    for s in range(200):
        i = rng.choice(len(yy), len(yy)); sel.append(LogisticRegression(penalty="l1", solver="liblinear", C=.1).fit(M[i], yy[i]).coef_[0])
    sel = np.array(sel); st = pd.DataFrame(dict(feature=nm, selected=(np.abs(sel) > 1e-6).mean(0), positive=(sel > 1e-6).mean(0), negative=(sel < -1e-6).mean(0), median=np.median(sel, 0))).sort_values("selected", ascending=False)
    st.to_csv(OUT / "stability.csv", index=False); print(st.round(3).to_string())
    for cols, l in ((FEAT, "9"), (CORE, "4")):
        print(l, {C: round(oof(simple(lambda: lr(cols, C)), X, y, SEED, 4)[1].pr_auc.mean(), 4) for C in (.003, .01, .03, .1, .3, 1, 10)})
    sc = IntegerScorecard().fit(X[FEAT], y); print(sc.points_, float(sc.link_.coef_[0][0]), float(sc.link_.intercept_[0]))
