"""Scheduled monitor: reads the privacy-preserving log, writes one status file. No inputs are ever read.

    python -m src.monitor_job --once        one pass
    python -m src.monitor_job --every 300   loop (the Compose `monitor` service)

Signal: PSI of logged risk bands against the development out-of-fold predictions, plus latency and failures.
A drift status means the people using the simulator differ from the development data, not that the model is wrong.
"""
import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from src.inference import ROOT
from src.monitoring import LOG, band, psi, read_log, verdict

BANDS, MIN_EVENTS = ["<2%", "2-5%", "5-10%", "10%+"], 30
STATUS = LOG.parent / "monitor_status.json"


def reference_bands():
    p = np.load(ROOT / "artifacts/results/final_oof.npz")["Logistic 4 (selected)|p"].mean(0)
    c = Counter(band(v) for v in p)
    return {b: c[b] / len(p) for b in BANDS}


def summarise(events, ref):
    e = [v for v in events if v.get("event") == "prediction"]
    out = dict(checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), events=len(e))
    if not e:
        return dict(out, status="insufficient data")
    c, lat = Counter(v.get("band") for v in e), [v["latency_ms"] for v in e if "latency_ms" in v]
    new = {b: c[b] / len(e) for b in BANDS}
    s = psi(np.array([ref[b] for b in BANDS]), np.array([new[b] for b in BANDS]))
    out.update(bands=new, reference=ref, psi=s, latency_p95_ms=float(np.quantile(lat, .95)) if lat else None,
               failures=sum(v.get("status") != "success" for v in e), models=sorted({v.get("model", "?") for v in e}))
    return dict(out, status=verdict(s) if len(e) >= MIN_EVENTS else "insufficient data")


def run():
    r = summarise(read_log(), reference_bands())
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1), encoding="utf-8")
    os.replace(tmp, STATUS)
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--once", action="store_true")
    a.add_argument("--every", type=int, default=300)
    a = a.parse_args()
    while True:
        run()
        if a.once:
            break
        time.sleep(a.every)
