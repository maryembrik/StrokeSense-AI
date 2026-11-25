import json
import os
from pathlib import Path

import numpy as np

from components.charts import *
from components.core import *
from src.monitoring import drift_report, prediction_drift, read_log, simulate

P, T = predictor(), research()
D = P.data["drift"]
LAB = {"age": "Age", "avg_glucose_level": "Glucose", "hypertension": "Hypertension", "heart_disease": "Heart disease"}
left, right = st.columns([1.3, 1], gap="large", vertical_alignment="center")
with left:
    hero("Monitoring demonstration", "The world changes. <em>Does the data?</em>", "A model is frozen on the day it is trained. The people it meets afterwards are not. Shift a synthetic population and see what the monitoring would notice.")
slot = right.container()
warn("SYNTHETIC DATA. Incoming batches on this page are generated. Nothing is connected to patients or to a production system.")

section("Create an incoming population")
C, G = st.columns([1, 1.7], gap="large")
with C, panel("ctl"):
    s = st.slider("Shift in age (years)", -15, 20, 0)
    g = st.slider("Glucose multiplier", .7, 1.5, 1.0, .05)
    h = st.slider("Hypertension prevalence multiplier", .5, 4.0, 1.0, .25)
    hh = st.slider("Heart disease prevalence multiplier", .5, 4.0, 1.0, .25)
    n = st.select_slider("Batch size", [250, 500, 1000, 2000], 1000)
B = simulate(D, n, s, g, h, seed=42, hd_mult=hh)
F = drift_report(B, D)
X = pd.DataFrame(B)[P.features]
risk = P.model.predict_proba(X)[:, 1]
pd_psi, pd_ks = prediction_drift(risk, T["reference"]["risk_quantiles"])
worst = max([r["psi"] for r in F] + [pd_psi])
state, col, ic = ("Stable", "#25C2B1", "●") if worst < .1 else ("Investigation recommended", "#D9A45B", "▲") if worst < .25 else ("Drift detected", "#E0786B", "■")
with slot, panel("status"):
    chart_title("Simulated batch status")
    html(f'<span class="status" style="color:{col};font-size:1.05rem">{ic} {state}</span><div class="readout" style="margin-top:1.1rem"><div><span>Batch</span><b>{n:,} synthetic people</b></div><div><span>Largest PSI</span><b>{worst:.3f}</b></div><div><span>Reference</span><b>{D["n"]:,} records</b></div></div>')
    st.caption("A label about data, not a statement about clinical safety.")
with G:
    which = st.segmented_control("Feature", ["Age", "Glucose"], default="Age", key="feat") or "Age"
with G, panel("dist", paper=True):
    chart_title(f"{which}: development data and synthetic batch", "Share of people in each bin.")
    if which == "Age":
        e, ref, v = np.array(T["age"]["edges"]), np.array(T["age"]["people"]), B["age"]
    else:
        e, ref, v = np.array(T["reference"]["glucose_edges"]), np.array(T["reference"]["glucose"]), np.clip(B["avg_glucose_level"], 50, 279.9)
    mid, new = (e[:-1] + e[1:]) / 2, np.histogram(np.clip(v, e[0], e[-1] - .01), e)[0] / n
    f = fig()
    line(f, mid, ref / ref.sum(), "development data", DTEAL, fill=.22, width=2.5, line_shape="hvh")
    line(f, mid, new, "synthetic batch", DAMBER, "dash", line_shape="hvh")
    show(f, "dist_c", 330, paper=True, x="age (years)" if which == "Age" else "average glucose (mg/dL)", y="share of people", yfmt=".0%", unified=True)

section("Feature drift", "Did the inputs change?")
word = {"stable": "● Stable", "watch": "▲ Investigation recommended", "drift": "■ Drift detected"}
wc = {"stable": "#25C2B1", "watch": "#D9A45B", "drift": "#E0786B"}
cards = ""
for r in F:
    b = r["feature"] in ("hypertension", "heart_disease")
    q = (lambda u: f"{u:.1%}") if b else (lambda u: f"{u:.1f}")
    cards += f'<div><h4>{LAB[r["feature"]]}</h4><b>{r["psi"]:.3f}</b><small>PSI</small><span style="color:{wc[r["status"]]};font-size:.86rem">{word[r["status"]]}</span><small style="margin-top:.5rem">{q(r["reference"])} → {q(r["incoming"])} {"prevalence" if b else "mean"}<br>{r["test"]} statistic {r["stat"]:.3f}, p {"&lt; 0.001" if r["p_value"] < .001 else f"= {r["p_value"]:.3f}"}</small></div>'
html(f'<div class="dr">{cards}</div>')
st.caption("PSI uses ten bins set at the development deciles for age and glucose, and the two categories for conditions; empty bins are floored at 0.0001. Under 0.10 stable, 0.10 to 0.25 investigation recommended, above that drift detected: a convention, not a law. KS compares continuous features, chi-square compares proportions. With no shift, about one p-value in twenty is still below 0.05.")

a, b = st.columns([1, 1.2], gap="large")
with a:
    section("Prediction drift", "Did the model's outputs change?")
    kpis([(f"{pd_psi:.3f}", "PSI of estimated risk"), (f"{risk.mean():.1%}", f"mean estimated risk, reference {T['reference']['risk_mean']:.1%}"), (f"{(risk >= P.data['threshold']).mean():.0%}", "would be flagged")], True)
    st.caption(f"KS statistic {pd_ks:.3f}. The generator draws each feature independently, so age no longer travels with glucose or conditions; the mean risk can differ slightly from the reference even with no shift. The frozen model scores the synthetic batch; both sides use the frozen model, and its development scores are in-sample.")
with b:
    section("Performance degradation", "Did the predictions become wrong?")
    callout("<b>Not measurable here.</b> It needs true outcomes for the new people. Feature drift alone does not prove the model is worse, and a stable page does not prove it is right.", "a")

section("Reference proportions")
with panel("props"):
    f, k = fig(), ["hypertension", "heart_disease"]
    for nme, vals, c, p_ in [("development data", [D["rates"][i] for i in k], TEAL, "/"), ("synthetic batch", [float(np.mean(B[i])) for i in k], AMBER, "")]:
        f.add_trace(go.Bar(x=[LAB[i] for i in k], y=vals, name=nme, marker=dict(color=c, cornerradius=4), width=.24, text=[f"{v:.1%}" for v in vals], textposition="outside", textfont=dict(color=WHITE), hovertemplate="%{x}: %{y:.1%}<extra>" + nme + "</extra>"))
    f.update_layout(bargap=.5)
    show(f, "props_c", 290, y="share of people", yfmt=".0%", yr=[0, max(.2, 1.25 * max(float(np.mean(B[i])) for i in k))])

section("How a prediction travels", "Only parts that exist in this application today.")
html('<div class="flow">' + "".join(f"<div><b>{x}</b><small>{y}</small></div>" for x, y in [
    ("Data", "Four inputs from the simulator"), ("Validation", "Types, ranges, binary flags"), ("Preprocessing", "Saved scaler in the pipeline"), ("Saved logistic model", "SHA-256 checked at start"),
    ("Prediction", "Risk, then threshold"), ("Monitoring", "JSON log without inputs, scheduled monitor"), ("Drift report", "PSI, KS, chi-square on this page")]) + "</div>")
section("Application log", "Structured JSON. Each event stores time, model version, latency and a coarse risk band. Inputs and exact risks are never stored.")
E = pd.DataFrame(read_log())
if E.empty:
    st.info("No predictions logged yet. Run the patient simulator, then return here.")
else:
    kpis([(f"{len(E)}", "logged events"), (f"{E.latency_ms.median():.1f} ms", "median latency"), (f"{(E.status == 'success').mean():.0%}", "succeeded")], True)
    table(E.tail(6)[["ts", "id", "model", "band", "latency_ms", "status"]].iloc[::-1].rename(columns={"ts": "Time (UTC)", "id": "Event", "model": "Model", "band": "Risk band", "latency_ms": "Latency, ms", "status": "Status"}))
section("Monitor job", "A separate scheduled process reads the log only. It compares logged risk bands with the development predictions.")
mon_file = ROOT / "logs/monitor_status.json" if "STROKESENSE_LOG" not in os.environ else Path(os.environ["STROKESENSE_LOG"]).parent / "monitor_status.json"
try:
    ms = json.loads(mon_file.read_text(encoding="utf-8"))
    kpis([(ms["status"], "status"), (f"{ms['events']}", "events checked"), (f"{ms['psi']:.3f}" if "psi" in ms else "–", "PSI of risk bands"), (f"{ms['latency_p95_ms']:.0f} ms" if ms.get("latency_p95_ms") is not None else "–", "95th percentile latency")], True)
    st.caption(f"Last check {ms['checked_at']}. Fewer than 30 events is reported as insufficient data. Simulator visitors are not a patient population.")
except (OSError, ValueError, KeyError):
    st.info("The monitor has not run here. Start it with docker compose up, or run: python -m src.monitor_job --once")
disclaimer()
