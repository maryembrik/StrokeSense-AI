"""Corrections requested in review. Development split only; fixed model settings; no BMI.
Outer: 4 x stratified 5-fold. Thresholds and feature selection use inner data of each outer training fold only."""
import warnings; warnings.filterwarnings("ignore")
import src.experiments as E, src.experiments_shape as S
from src.experiments import *
from src.day1 import age_spline, CORE, FEAT, lr
from src.train import pick_threshold
from src.classification_table import four, ageonly
from sklearn.model_selection import StratifiedKFold
E.make = S.make; OUT = ROOT / "artifacts/results"
MODELS = {"Age only": ageonly, "Logistic 4": lambda A, y, B: lr(CORE).fit(A, y).predict_proba(B)[:, 1], "Logistic 9": lambda A, y, B: lr(FEAT).fit(A, y).predict_proba(B)[:, 1],
          "XGBoost": four("XGBoost"), "EBM monotone": four("EBM monotone"), "CatBoost": four("CatBoost"), "HGB mono pairs": four("HGB mono pairs")}

def select(A, y, seed, n=100, cut=.8):
    """Stability selection: raw variables kept if any of their columns has a nonzero L1 coefficient in >= cut of bootstraps."""
    Z = lr(FEAT)[0].fit(A[FEAT]); M = Z.transform(A[FEAT]); src = [next(f for f in FEAT if s.split("__")[1].startswith(f)) for s in Z.get_feature_names_out()]
    r = np.random.default_rng(seed); hit = dict.fromkeys(FEAT, 0); yy = y.values
    for _ in range(n):
        i = r.choice(len(yy), len(yy)); c = LogisticRegression(penalty="l1", solver="liblinear", C=.03).fit(M[i], yy[i]).coef_[0]
        for f in {s for s, v in zip(src, c) if abs(v) > 1e-6}: hit[f] += 1
    return [f for f in FEAT if hit[f] / n >= cut], {f: hit[f] / n for f in FEAT}
def job(name, i, a, b, X, y):
    warnings.filterwarnings("ignore"); import src.experiments as EE, src.experiments_shape as SS; EE.make = SS.make
    A, yA, B = X.iloc[a], y.iloc[a], X.iloc[b]; extra = {}
    if name == "Logistic nested-selection":
        def fn(U, v, W, s=i): k, _ = select(U, v, s); return lr(k).fit(U, v).predict_proba(W)[:, 1]
        _, extra = select(A, yA, i)
    else: fn = MODELS[name]
    q = np.zeros(len(a))
    for u, v in StratifiedKFold(5, shuffle=True, random_state=1000 + i).split(A, yA): q[v] = fn(A.iloc[u], yA.iloc[u], A.iloc[v])
    return name, i, b, fn(A, yA, B), pick_threshold(yA.values, q), extra
def matched(P, fold, Y, age, w, cw, wt=None, same_fold=True):
    """P: scores; pairs share an outer fold if same_fold. wt: patient multiplicities (bootstrap). cw: each case counts once."""
    wt = np.ones(len(Y)) if wt is None else wt; N = D = 0.
    for j in np.flatnonzero((Y == 1) & (wt > 0)):
        m = (Y == 0) & (np.abs(age - age[j]) <= w + 1e-9) & (wt > 0)
        if same_fold: m &= fold == fold[j]
        if not m.any(): continue
        u = wt[m]; s = ((P[j] > P[m]) + .5 * (P[j] == P[m])) @ u; t = u.sum()
        if cw: N += wt[j] * s / t; D += wt[j]
        else: N += wt[j] * s; D += wt[j] * t
    return N / D if D else np.nan

if __name__ == "__main__":
    X, _, y, _ = load_split(); del _; Y = y.values; age = X.age.values; n = len(Y)
    sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(X, y))
    names = list(MODELS) + ["Logistic nested-selection"]
    res = Parallel(n_jobs=-2)(delayed(job)(m, i, a, b, X, y) for m in names for i, (a, b) in enumerate(sp))
    P = {m: np.zeros((4, n)) for m in names}; G = {m: np.zeros((4, n), bool) for m in names}; fold = np.zeros((4, n), int); thr = {m: [] for m in names}; freq = []
    for m, i, b, p, t, ex in res:
        P[m][i // 5, b] = p; G[m][i // 5, b] = p >= t; fold[i // 5, b] = i; thr[m].append(t)
        if ex: freq.append(ex)
    np.savez_compressed(OUT / "corrections_oof.npz", y=Y, age=age, fold=fold, **{m: v for m, v in P.items()}, **{m + " flag": v for m, v in G.items()})
    rng = np.random.default_rng(0); W = []
    for _b in range(1000):
        i = np.r_[rng.choice(np.flatnonzero(Y == 1), (Y == 1).sum()), rng.choice(np.flatnonzero(Y == 0), (Y == 0).sum())]; W.append(np.bincount(i, minlength=n))
    W = np.array(W)
    def op(m, w):  # averaged over 4 repeats
        f = G[m].mean(0); tp = (w * f)[..., Y == 1].sum(-1); fp = (w * f)[..., Y == 0].sum(-1); return tp, fp
    one_ = np.ones(n); rows = []
    for m in names:
        per = [((g & (Y == 1)).sum(), (g & (Y == 0)).sum()) for g in G[m]]; tp, fp = op(m, one_); bt, bf = op(m, W); s = bt / 199; sp_ = 1 - bf / (n - 199); pr = bt / (bt + bf); f1 = 2 * bt / (bt + bf + 199)
        q = lambda v: [round(float(z), 3) for z in np.quantile(v, [.025, .975])]
        rows.append(dict(model=m, tp=tp, fp=fp, sens=tp / 199, sens_ci=q(s), spec=1 - fp / (n - 199), spec_ci=q(sp_), precision=tp / (tp + fp), prec_ci=q(pr), f1=2 * tp / (tp + fp + 199), f1_ci=q(f1), fp_ci=q(bf), fp_range=(min(x[1] for x in per), max(x[1] for x in per)), tp_range=(min(x[0] for x in per), max(x[0] for x in per)), thr=np.mean(thr[m]),
            pr_fold=np.mean([average_precision_score(Y[b], P[m][i // 5, b]) for i, (a, b) in enumerate(sp)]), pr_fold_sd=np.std([average_precision_score(Y[b], P[m][i // 5, b]) for i, (a, b) in enumerate(sp)], ddof=1), pr_pooled=np.mean([average_precision_score(Y, p) for p in P[m]]),
            roc_fold=np.mean([roc_auc_score(Y[b], P[m][i // 5, b]) for i, (a, b) in enumerate(sp)])))
    T = pd.DataFrame(rows); T.to_csv(OUT / "operating_point_nested.csv", index=False); pd.set_option("display.width", 400, "display.max_columns", 50); print(T.round(3).to_string())
    print("== paired: other minus Logistic 4"); pr = []
    t0, f0 = op("Logistic 4", W)
    for m in names:
        if m == "Logistic 4": continue
        t, f = op(m, W); a, c = op(m, one_); a0, c0 = op("Logistic 4", one_); nb = lambda t, f: (t - f * .05 / .95) / n
        pr.append(dict(model=m, d_tp=a - a0, d_tp_ci=np.quantile(t - t0, [.025, .975]).round(1), d_fp=c - c0, d_fp_ci=np.quantile(f - f0, [.025, .975]).round(1), d_netbenefit_per1000=1000 * (nb(a, c) - nb(a0, c0)), ci=(1000 * np.quantile(nb(t, f) - nb(t0, f0), [.025, .975])).round(2)))
    pr = pd.DataFrame(pr); pr.to_csv(OUT / "paired_vs_logistic.csv", index=False); print(pr.round(2).to_string())
    print("== paired PR/ROC, patient bootstrap, mean over 4 repeats of pooled OOF"); b = []
    for w in W[:300]:
        o = {}
        for m in ["Logistic 4", "XGBoost", "Age only", "Logistic 9", "Logistic nested-selection"]:
            o[m] = np.mean([average_precision_score(Y, p, sample_weight=w) for p in P[m]]), np.mean([roc_auc_score(Y, p, sample_weight=w) for p in P[m]])
        b.append({(m, k): o["Logistic 4"][j] - o[m][j] for m in o if m != "Logistic 4" for j, k in enumerate(("PR", "ROC"))})
    print(pd.DataFrame(b).quantile([.025, .5, .975]).T.round(4).to_string())
    fr = pd.DataFrame(freq); fs = pd.DataFrame(dict(mean_bootstrap_freq=fr.mean(), min=fr.min(), outer_folds_selected=(fr >= .8).sum())); fs.to_csv(OUT / "nested_selection.csv"); print(fs.round(3).to_string())
    print("== age matched discrepancy"); 
    for m in ("Age only", "Logistic 4"):
        print(m, "fold-mean", round(np.mean([age_matched_auc(Y[b], P[m][i // 5, b], age[b]) for i, (a, b) in enumerate(sp)]), 4), "| same-fold pairs, per repeat", round(np.mean([matched(P[m][r], fold[r], Y, age, 3, 0) for r in range(4)]), 4),
              "| cross-fold pairs, per repeat", round(np.mean([matched(P[m][r], fold[r], Y, age, 3, 0, same_fold=False) for r in range(4)]), 4), "| cross-fold, scores averaged over repeats", round(matched(P[m].mean(0), fold[0], Y, age, 3, 0, same_fold=False), 4))
    print("raw age as score, all pairs +/-3:", round(matched(age, fold[0], Y, age, 3, 0, same_fold=False), 4))
    rows = []
    for w in (0, 1, 3, 5):
        for cw in (0, 1):
            f = lambda m, wt=None: np.mean([matched(P[m][r], fold[r], Y, age, w, cw, wt) for r in range(4)])
            pairs = np.mean([sum(((Y == 0) & (abs(age - age[j]) <= w + 1e-9) & (fold[r] == fold[r][j])).sum() for j in np.flatnonzero(Y == 1)) for r in range(4)])
            bs = np.array([(f("Logistic 4", q), f("Age only", q)) for q in W[:150]]); L, Ag = f("Logistic 4"), f("Age only")
            rows.append(dict(window=w, weighting="case" if cw else "pair", pairs=pairs, logistic=L, lo=np.quantile(bs[:, 0], .025), hi=np.quantile(bs[:, 0], .975), age_only=Ag, gain=L - Ag, gain_lo=np.quantile(bs[:, 0] - bs[:, 1], .025), gain_hi=np.quantile(bs[:, 0] - bs[:, 1], .975)))
    am = pd.DataFrame(rows); am.to_csv(OUT / "age_matched_sensitivity.csv", index=False); print(am.round(3).to_string())
    print("== calibration, per repeat then averaged"); rows = []
    def cal(p, w=None):
        z = np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5))); m = LogisticRegression(C=1e6).fit(z[:, None], Y, sample_weight=w); a = 0.
        for _ in range(20):
            q = 1 / (1 + np.exp(-(z + a))); ww = np.ones(n) if w is None else w; a += (ww * (Y - q)).sum() / (ww * q * (1 - q)).sum()
        return m.coef_[0][0], a
    for m in ("Logistic 4", "XGBoost", "EBM monotone"):
        c = np.mean([cal(p) for p in P[m]], 0); bs = np.array([np.mean([cal(p, w) for p in P[m]], 0) for w in W[:150]])
        rows.append(dict(model=m, slope=c[0], slope_lo=np.quantile(bs[:, 0], .025), slope_hi=np.quantile(bs[:, 0], .975), intercept_in_large=c[1], int_lo=np.quantile(bs[:, 1], .025), int_hi=np.quantile(bs[:, 1], .975), brier=np.mean([brier_score_loss(Y, p) for p in P[m]]), mean_pred=P[m].mean()))
    c = pd.DataFrame(rows); c.to_csv(OUT / "calibration_outer.csv", index=False); print(c.round(3).to_string())
