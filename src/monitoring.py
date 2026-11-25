"""Privacy-preserving prediction log (no patient inputs, no exact risk) and drift statistics."""
import datetime
import json
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
from scipy import stats

LOG = Path(os.environ.get("STROKESENSE_LOG", Path(__file__).resolve().parents[1] / "logs/predictions.jsonl"))


def logger():
    l = logging.getLogger("strokesense")
    if not l.handlers:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(LOG, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(message)s"))
        l.addHandler(h)
        l.setLevel(logging.INFO)
        l.propagate = False
    return l


def band(r):
    return "<2%" if r < .02 else "2-5%" if r < .05 else "5-10%" if r < .10 else "10%+"


def log_prediction(risk, ms, version):
    e = dict(event="prediction", ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             id=uuid.uuid4().hex[:8], model=version[:8], band=band(risk), latency_ms=round(ms, 1), status="success")
    logger().info(json.dumps(e))
    return e


def read_log(n=500):
    if not LOG.exists():
        return []
    out = []
    for s in LOG.read_text(encoding="utf-8").splitlines()[-n:]:
        try:
            out.append(json.loads(s))
        except ValueError:
            pass
    return out


def psi(ref, new, eps=1e-4):
    r, n = np.clip(ref, eps, None), np.clip(new, eps, None)
    return float(((n - r) * np.log(n / r)).sum())


def verdict(p):
    return "stable" if p < .1 else "watch" if p < .25 else "drift"


def drift_report(batch, ref):
    rows = []
    for c, q in ref["quantiles"].items():
        v, e = np.asarray(batch[c], float), np.unique(ref["edges"][c])
        g = np.linspace(0, 1, len(q))
        u, ix = np.unique(q, return_index=True)
        new = np.histogram(np.clip(v, e[0], e[-1]), e)[0] / len(v)
        r0 = np.diff(np.interp(e, u, g[ix]))
        p = psi(r0 / r0.sum(), new)
        ks = stats.ks_1samp(v, lambda z: np.interp(z, u, g[ix]))
        rows.append(dict(feature=c, test="KS", psi=p, stat=float(ks.statistic), p_value=float(ks.pvalue), reference=float(np.mean(q)), incoming=float(v.mean()), status=verdict(p)))
    for c, r in ref["rates"].items():
        k, n = float(np.sum(batch[c])), len(batch[c])
        p = psi(np.array([r, 1 - r]), np.array([k / n, 1 - k / n]))
        table = [[r * ref["n"], (1 - r) * ref["n"]], [k, n - k]]
        chi = stats.chi2_contingency(table)
        rows.append(dict(feature=c, test="chi-square", psi=p, stat=float(chi[0]), p_value=float(chi[1]), reference=r, incoming=k / n, status=verdict(p)))
    return rows


def prediction_drift(risk, ref_quantiles):
    """PSI (ten reference deciles) and KS statistic of estimated risk against its development distribution."""
    q = np.asarray(ref_quantiles)
    e = np.interp(np.linspace(0, 1, 11), np.linspace(0, 1, len(q)), q)
    new = np.histogram(np.clip(risk, e[0], e[-1]), e)[0] / len(risk)
    ks = stats.ks_1samp(risk, lambda v: np.interp(v, q, np.linspace(0, 1, len(q))))
    return psi(np.full(10, .1), new), float(ks.statistic)


def simulate(ref, n=500, age_shift=0, glucose_mult=1.0, ht_mult=1.0, seed=0, hd_mult=1.0):
    g, u = np.random.default_rng(seed), np.linspace(0, 1, 201)
    s = lambda c: np.interp(g.random(n), u, ref["quantiles"][c])
    return dict(age=np.clip(s("age") + age_shift, 0, 100), avg_glucose_level=s("avg_glucose_level") * glucose_mult,
                hypertension=(g.random(n) < min(1, ref["rates"]["hypertension"] * ht_mult)).astype(int),
                heart_disease=(g.random(n) < min(1, ref["rates"]["heart_disease"] * hd_mult)).astype(int))
