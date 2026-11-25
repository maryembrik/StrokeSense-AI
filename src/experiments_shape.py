"""Phase-1b: shape-constrained boosting. Development split only."""
import os; os.environ["PYTHONWARNINGS"] = "ignore"
import src.experiments as E
from src.experiments import *
_old = E.make
def make(name, X):
    if not name.startswith("HGB"): return _old(name, X)
    n, c = nums(X), cats(X)
    pre = make_column_transformer(("passthrough", n), (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), c))
    k = dict(max_iter=150, learning_rate=.04, min_samples_leaf=40, l2_regularization=5, random_state=0)
    k["categorical_features"] = list(range(len(n), len(n) + len(c)))
    if "mono" in name: k["monotonic_cst"] = [int(f in MONO_UP) for f in n] + [0] * len(c)
    if "additive" in name: k["interaction_cst"] = "no_interactions"
    if "pairs" in name: k["interaction_cst"] = "pairwise"
    k["max_leaf_nodes"] = 4 if "small" in name else 8
    return make_pipeline(pre, HistGradientBoostingClassifier(**k))
E.make = make
NAMES = ["HGB", "HGB mono", "HGB mono additive", "HGB mono pairs", "HGB mono pairs small", "HGB small"]
if __name__ == "__main__":
    X, _, y, _ = load_split()
    from src.experiments_shape import make as mk
    def wrap(n, fs, X, y, tr, te, i):
        import src.experiments as EE, src.experiments_shape as S; EE.make = S.make
        return EE.one(n, fs, X, y, tr, te, i)
    sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(X, y))
    jobs = [(n, i, s) for n in NAMES for i, s in enumerate(sp)]
    o = [dict(label=n, **wrap(n, {}, X, y, s[0], s[1], i)) for n, i, s in jobs]
    a = pd.DataFrame(o); a.to_csv(ROOT / "artifacts/results/shape_cv.csv", index=False)
    print(a.groupby("label", sort=False).mean().drop(columns="fold").round(4).to_string())
    old = pd.read_csv(ROOT / "artifacts/results/models_cv.csv"); al = pd.concat([old, a])
    P = al.pivot(index="fold", columns="label", values="pr_auc")
    n, t = len(P), None
    for ref in ["HistGradientBoosting", "Logistic", "Age only (logistic)"]:
        for c in ["HGB mono pairs", "HGB mono pairs small", "HGB mono", "EBM monotone", "XGBoost"]:
            d = P[c] - P[ref]; se = d.std() * np.sqrt(1 / n + .25)  # Nadeau-Bengio corrected
            print(c, "-", ref, round(d.mean(), 4), "+/-", round(2 * se, 4), (d > 0).sum(), "/20")
