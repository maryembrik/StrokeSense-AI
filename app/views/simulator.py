import numpy as np

from components.charts import *
from components.core import *
from components.visuals import gauge
from src.monitoring import log_prediction

P = predictor()
t = P.data["threshold"]
hero("Interactive classification", "Ask the model about <em>a hypothetical person</em>", "The model was trained on records labelled 1 (had a stroke) or 0 (did not). It estimates how likely this profile is to carry label 1, then assigns a class. It does not say whether a stroke will happen in the future.")
L, R = st.columns([1, 1.55], gap="large")
with L, panel("inputs"):
    chart_title("Profile", "All the inputs the final model uses, over the full age range it was trained on.")
    age = st.slider("Age (years)", 1, 82, 58)
    glu = st.slider("Average glucose (mg/dL)", 55, 272, 105)
    ht = st.toggle("Hypertension")
    hd = st.toggle("Heart disease")
    x = dict(age=age, avg_glucose_level=glu, hypertension=int(ht), heart_disease=int(hd))
    run = st.button("Classify this profile", type="primary", width="stretch")
    with st.expander("Why only four inputs?"):
        N = csv("nested_selection.csv", 0).outer_folds_selected
        st.markdown(f"The dataset has ten possible inputs. All were studied, but the saved model contains four, so any other field here would change nothing.\n\n- **Kept** (times selected in 20 repeated selections): age {N['age']}, glucose {N['avg_glucose_level']}, hypertension {N['hypertension']}, heart disease {N['heart_disease']}.\n- **Gender, smoking, work, residence, marriage**: selected 0 times. This means they added little here once age and the other inputs were known, not that they have no effect on stroke. A model with all nine inputs scores {results().loc['Logistic 9', 'pr']:.3f} PR-AUC, against {results().loc[MAIN, 'pr']:.3f} with four; the difference is within uncertainty.\n- **BMI**: excluded because models were using the fact that it was blank. See Leak Detective.")
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
    b, a = st.columns([1, 1.15], gap="medium", vertical_alignment="center")  # class first, also on mobile
    with a:
        html(gauge(r["risk"] if r else None, st.session_state.get("prev", 0), t))
    with b:
        if not r:
            chart_title("Ready", "Set a profile and classify it.")
        else:
            s = st.session_state["sim_x"]
            if s != x:
                html('<span class="status" style="color:#D9A45B">△ inputs changed, run again</span>')
            pr = f"{r['risk']:.3%}" if abs(r["risk"] - t) < .0005 else f"{r['risk']:.1%}"  # near the cut-off, rounding could contradict the class
            col = "#D9A45B" if r["flagged"] else "#70D9D0"
            html(f'<div class="verdict" style="--c:{col}" role="status"><span>Predicted class</span><b>{"▲ 1, stroke" if r["flagged"] else "● 0, non-stroke"}</b>'
                 f'<p>Stroke-class probability: {pr}. The model estimates that about {round(1000 * r["risk"])} in 1,000 similar records carry the stroke label. This is {"at or above" if r["flagged"] else "below"} the {t:.3%} threshold, so it assigns class {int(r["flagged"])}.</p></div>')
            st.caption(f"Profile: {s['age']} years, {s['avg_glucose_level']} mg/dL" + (", hypertension" if s["hypertension"] else "") + (", heart disease" if s["heart_disease"] else ""))
if r:
    T4 = final_test()["models"]["Logistic, 4 features"]
    callout("<b>This is the model's classification, not the person's actual status.</b> It is not a future stroke chance: the data has no dates and no follow-up. "
            + (f"In the final test, only {T4['precision']['k']} of {T4['precision']['n']} people put in class 1 had a recorded stroke." if r["flagged"] else f"In the final test, {T4['fn']} of the 50 people with a recorded stroke were also put in class 0, so class 0 does not mean safe."))
    with st.expander("What do the simpler models say?"):
        html(f'<div class="readout"><div><span>Age-only model, probability</span><b>{r["age_only"]:.1%}</b></div><div><span>Pocket scorecard, probability</span><b>{r["card_risk"]:.1%}</b></div></div>')
    s, c, q = st.session_state["sim_x"], r["context"], r["context"]["pct"]
    ops = [r["risk"], r["age_only"], r["card_risk"]]
    if max(ops) - min(ops) > .05:
        callout(f"The three models differ by <b>{100 * (max(ops) - min(ops)):.0f} percentage points</b> for this profile. Treat the estimate as uncertain.", "a")
    section("Compared with people of a similar age", "The number above is easier to read next to what the model says about other people this age.")
    with panel("context", paper=True):
        chart_title(f"Estimated probability among {c['n']} development records aged {max(0, s['age'] - 3)} to {min(82, s['age'] + 3)}", f"{c['strokes']} of them had a recorded stroke. Shaded bars cover the middle 80% and middle 50%.")
        f = fig()
        for lo, hi, w, n in [(10, 90, 16, "middle 80%"), (25, 75, 30, "middle 50%")]:
            f.add_trace(go.Scatter(x=[q[lo], q[hi]], y=[0, 0], mode="lines", line=dict(color=rgba(DTEAL, .28 if w < 20 else .42), width=w), name=n, hovertemplate="%{x:.1%}<extra>" + n + "</extra>"))
        points(f, [q[50]], [0], "median", INK, "line-ns-open", 26, hover="median %{x:.1%}", ring=INK)
        points(f, [r["risk"]], [0], "this profile", DAMBER, "diamond", 18, hover="this profile %{x:.1%}", ring="#F3F5F2")
        f.update_yaxes(visible=False)
        show(f, "context_c", 190, paper=True, x="probability of the stroke class", xfmt=".0%" if q[99] > .05 else ".1%", yr=[-1, 1], xr=[0, max(q[99], r["risk"]) * 1.12])
        st.markdown(f"This estimate is higher than about **{r['percentile']:.0f}%** of those records.")
    if c["strokes"] < MIN_CASES:
        callout(f"Only {c['strokes']} stroke cases fall in this age range, too few to judge how well the model works here.", "a")
    if s["age"] < 18:
        y = csv("calibration_subgroups.csv").set_index("group").loc["Under 18"]
        warn(f"Under 18, the development data has {int(y.n)} people and only {int(y.strokes)} with a recorded stroke. The model returns a value because it was trained on all ages, but almost nothing here can be checked.")
    st.caption("Method: held-out cross-validation estimates for development records within three years of the chosen age. It describes this dataset, not a population. A position below the reference does not mean a person is safe.")
    section("What moves the estimate", "Each bar is a fitted coefficient multiplied by the standardised input, relative to the average development record. Text below gives the same effect as a multiplier of the odds.")
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
        st.markdown("Odds against the average record: " + ", ".join(f"{lab[i].lower()} **×{np.exp(r['contrib'][i]):.2f}**" for i in r["contrib"]) + ". Above 1 raises the estimate, below 1 lowers it.")
    with b:
        callout("These are associations the model learned, not causes. Changing an input here does not show the effect of treating a condition.")
        pts = {a_: b_ for a_, b_ in r["points"].items() if b_}
        callout(f"Pocket scorecard: <b>{r['total']} points</b>" + (". " + ", ".join(f"{a_.replace('_', ' ')} ({b_})" for a_, b_ in pts.items()) if pts else "") + ". It can be added by hand, is less accurate, and is shown only for explanation.", "a")
        callout(f"The {t:.1%} threshold is an experimental setting chosen to catch about 80% of strokes. With no conditions and median glucose, age alone crosses it at about 61.")
warn(DISCLAIMER)
