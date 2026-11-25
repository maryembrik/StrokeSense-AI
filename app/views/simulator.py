from components.charts import *
from components.core import *
from components.visuals import gauge
from src.monitoring import log_prediction

P = predictor()
t = P.data["threshold"]
hero("Interactive simulation", "Ask the model about <em>a hypothetical person</em>", "Four inputs, one frozen logistic regression. This shows how the model responds. It does not show what would happen to a real person if something about them changed.")
L, R = st.columns([1, 1.55], gap="large")
with L, panel("inputs"):
    chart_title("Profile", "Only the four features the final model uses.")
    age = st.slider("Age (years)", 18, 82, 58)
    glu = st.slider("Average glucose (mg/dL)", 55, 272, 105)
    ht = st.toggle("Hypertension")
    hd = st.toggle("Heart disease")
    x = dict(age=age, avg_glucose_level=glu, hypertension=int(ht), heart_disease=int(hd))
    run = st.button("Run risk simulation", type="primary", width="stretch")
    st.caption("BMI, smoking, work, residence, gender and marital status are not used by the final model.")
if run:
    try:
        r = P.predict(x)
        st.session_state["prev"] = st.session_state["sim"]["risk"] if "sim" in st.session_state else 0
        st.session_state["sim"], st.session_state["sim_x"] = r, x
        log_prediction(r["risk"], r["ms"], P.version)
    except ValueError as e:
        st.error(f"The profile could not be scored: {e}")
r = st.session_state.get("sim")
with R, panel("result"):
    a, b = st.columns([1.15, 1], gap="medium", vertical_alignment="center")
    with a:
        html(gauge(r["risk"] if r else None, st.session_state.get("prev", 0), t))
    with b:
        if not r:
            chart_title("Ready", "Set a profile and run the simulation.")
        else:
            s = st.session_state["sim_x"]
            if s != x:
                html('<span class="status" style="color:#D9A45B">△ inputs changed, run again</span>')
            html(f'<div class="readout"><div><span>Profile</span><b>{s["age"]} y, {s["avg_glucose_level"]} mg/dL</b></div><div><span>Estimated model risk</span><b>{r["risk"]:.1%}</b></div>'
                 f'<div><span>Age-only model</span><b>{r["age_only"]:.1%}</b></div><div><span>Pocket scorecard</span><b>{r["card_risk"]:.1%}</b></div></div>')
            html(f'<p class="cs" style="margin:1rem 0 .4rem">Research classification, a separate step from the risk</p><span class="status" style="color:{"#D9A45B" if r["flagged"] else "#70D9D0"}">{"▲ at or above" if r["flagged"] else "● below"} the {t:.1%} threshold</span>')
if r:
    s, c, q = st.session_state["sim_x"], r["context"], r["context"]["pct"]
    ops = [r["risk"], r["age_only"], r["card_risk"]]
    if max(ops) - min(ops) > .05:
        callout(f"The three models differ by <b>{100 * (max(ops) - min(ops)):.0f} percentage points</b> for this profile. Treat the estimate as uncertain.", "a")
    section("Compared with people of a similar age", "The number above is easier to read next to what the model says about other people this age.")
    with panel("context", paper=True):
        chart_title(f"Estimated risk among {c['n']} development records aged {s['age'] - 3} to {s['age'] + 3}", f"{c['strokes']} of them had a stroke. Shaded bars cover the middle 80% and middle 50%.")
        f = fig()
        for lo, hi, w, n in [(10, 90, 16, "middle 80%"), (25, 75, 30, "middle 50%")]:
            f.add_trace(go.Scatter(x=[q[lo], q[hi]], y=[0, 0], mode="lines", line=dict(color=rgba(DTEAL, .28 if w < 20 else .42), width=w), name=n, hovertemplate="%{x:.1%}<extra>" + n + "</extra>"))
        points(f, [q[50]], [0], "median", INK, "line-ns-open", 26, hover="median %{x:.1%}", ring=INK)
        points(f, [r["risk"]], [0], "this profile", DAMBER, "diamond", 18, hover="this profile %{x:.1%}", ring="#F3F5F2")
        f.update_yaxes(visible=False)
        show(f, "context_c", 190, paper=True, x="estimated model risk", xfmt=".0%", yr=[-1, 1], xr=[0, max(q[99], r["risk"]) * 1.12])
        st.markdown(f"This estimate is higher than about **{r['percentile']:.0f}%** of those records.")
    if c["strokes"] < MIN_CASES:
        callout(f"Only {c['strokes']} stroke cases fall in this age range, too few to judge how well the model works here.", "a")
    st.caption("Method: held-out cross-validation estimates for development records within three years of the chosen age. It describes this dataset, not a population. A position below the reference does not mean a person is safe.")
    section("What moves the estimate", "Each bar is a fitted coefficient multiplied by the standardised input, relative to the average development record.")
    a, b = st.columns([1.5, 1], gap="large")
    with a, panel("contrib"):
        lab = {"age": "Age", "avg_glucose_level": "Glucose", "hypertension": "Hypertension", "heart_disease": "Heart disease"}
        k = list(r["contrib"])[::-1]
        v = [r["contrib"][i] for i in k]
        m = max(.6, 1.4 * max(map(abs, v)))
        f = fig()
        f.add_trace(go.Bar(x=v, y=[lab[i] for i in k], orientation="h", width=.46, marker=dict(color=[TEAL if j >= 0 else SLATE for j in v], cornerradius=4), text=[f"{j:+.2f}" for j in v], textposition="outside",
                           textfont=dict(family="JetBrains Mono, Consolas, monospace", size=13, color=WHITE), hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
        f.add_vline(x=0, line=dict(color=MUTE, width=1))
        show(f, "contrib_c", 260, legend=False, x="contribution to log-odds (teal raises, grey lowers)", xr=[-m, m])
    with b:
        callout("These are associations the model learned, not causes. Changing an input here does not show the effect of treating a condition.")
        pts = {a_: b_ for a_, b_ in r["points"].items() if b_}
        callout(f"Pocket scorecard: <b>{r['total']} points</b>" + (". " + ", ".join(f"{a_.replace('_', ' ')} ({b_})" for a_, b_ in pts.items()) if pts else "") + ". It can be added by hand, is less accurate, and is shown only for explanation.", "a")
        callout(f"The {t:.1%} threshold is an experimental setting chosen to catch about 80% of strokes. With no conditions and median glucose, age alone crosses it at about 61.")
warn(DISCLAIMER)
