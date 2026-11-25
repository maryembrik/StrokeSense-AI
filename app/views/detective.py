from components.charts import *
from components.core import *
from components.visuals import lens

T, V = research(), subgroups()
miss = T["missing_bmi"]
nb, wb = V.loc["No BMI recorded"], V.loc["BMI recorded"]

left, right = st.columns([1, 1.15], gap="large", vertical_alignment="center")
with left:
    hero("Case file / missing BMI", "When missing data becomes a <em>shortcut</em>", "Investigating how missing BMI values created an apparent predictive advantage.")
with right:
    lens(int(nb.n), int(nb.strokes), round(wb.observed * nb.n))
    st.caption(f"Amber points are strokes. Left: the {int(nb.n)} development records with no BMI, {int(nb.strokes)} strokes. Right: the rate among people with a BMI ({wb.observed:.1%}) scaled to {int(nb.n)} records, about {round(wb.observed * nb.n)}. Positions are random. Move the lens.")

section("The missingness mystery", "BMI is blank for a small share of people, but not evenly.")
with panel("missing"):
    rows = ""
    for k, l, c in [("overall", "Everyone", "#6F8599"), ("stroke", "People who had a stroke", "#D9A45B"), ("no_stroke", "People who did not", "#25C2B1")]:
        (a, n), (lo, hi) = miss[k], miss["ci"][k]
        rows += f'<div class="r"><div><b>{a / n:.1%}</b><small>{l}</small></div><div><div class="track" style="--c:{c}"><i style="width:{100 * a / n / .3:.1f}%"></i><u style="left:{100 * lo / .3:.1f}%;width:{100 * (hi - lo) / .3:.1f}%"></u></div><small style="color:#A3B4C4">{a} of {n:,} have no BMI</small></div></div>'
    html(f'<div class="split">{rows}</div>')
    st.caption("Share with no BMI recorded, development data. White brackets are 95% Wilson intervals. Bars run from 0% to 30%. We do not know why the values are missing.")

section("The experiment", "The same models, trained three ways. Only the treatment of blank BMI changes.")
NM = {"HGB mono pairs": "Constrained boosting", "HistGradientBoosting": "Gradient boosting", "XGBoost": "XGBoost", "CatBoost": "CatBoost", "EBM monotone": "EBM", "Logistic": "Logistic, 9 inputs", "Age only (logistic)": "Age only"}
METRIC = {"PR-AUC": "pr_auc", "ROC-AUC": "roc_auc", "Strokes in top 20%": "recall_top20"}
met = st.segmented_control("Metric", list(METRIC), default="PR-AUC", key="metric") or "PR-AUC"
L = csv("leak_cv.csv")
Q = L[L.label.isin(NM)].pivot_table(index="label", columns="mode", values=METRIC[met]).loc[list(NM)]
with panel("slope", paper=True):
    chart_title(f"{met} when blank BMI is filled, randomised or removed", "Each line is one model. Amber: tree boosting. Teal: logistic regression and age alone.")
    f, x = fig(), ["Filled with one median", "Filled with random plausible values", "BMI removed"]
    ends = sorted(Q["drop"].items(), key=lambda kv: kv[1])
    gap, ys, last = (Q.max().max() - Q.min().min()) * .062, {}, -9
    for k, v in ends:  # keep end labels from colliding
        ys[k] = last = max(v, last + gap)
    for k in Q.index:
        tree = k not in ("Logistic", "Age only (logistic)")
        col = DAMBER if tree else DTEAL
        yy = [Q.loc[k, c] for c in ("median", "draw", "drop")]
        f.add_trace(go.Scatter(x=x, y=yy, name=NM[k], mode="lines+markers", line=dict(color=col, width=2.2 if k != "Logistic" else 3.4, dash="dot" if k.startswith("Age") else "solid"),
                               marker=dict(size=11, symbol="diamond" if tree else "circle", color=col, line=dict(color="#F3F5F2", width=2)), hovertemplate=f"{NM[k]}<br>%{{x}}: %{{y:.3f}}<extra></extra>"))
        f.add_annotation(x=1, xref="paper", y=ys[k], text=f"{NM[k]} {Q.loc[k, 'drop']:.3f}", showarrow=False, xanchor="left", xshift=8, font=dict(size=12, color=INK))
    f.update_xaxes(range=[-.12, 2.04])
    show(f, "slope_c", 440, paper=True, legend=False, y=met, mr=190)
    st.caption("Mean over cross-validation folds on development data, fixed model settings, to isolate the effect of filling. These are not the tuned scores in the Research Lab. False alarms were not computed for this experiment, so none are shown.")
d = Q["median"] - Q["drop"]
table(pd.DataFrame([{"Model": NM[k], "Median-filled": f"{Q.loc[k, 'median']:.3f}", "Random fill": f"{Q.loc[k, 'draw']:.3f}", "BMI removed": f"{Q.loc[k, 'drop']:.3f}", "Change when removed": f"{-d[k]:+.3f}"} for k in Q.index]))

insight("The apparent performance advantage associated with median-imputed BMI was <em>substantially reduced</em> when BMI was excluded.")
a, b = st.columns([1.2, 1], gap="large")
with a, panel("mech"):
    chart_title("Interrogating the shortcut", "Average probability gradient boosting gives to people with no BMI, and its ranking score on people who do have one.")
    ren = {"median": "Median", "constant 28.137 (unique)": "Another constant", "median + jitter sd=4": "Median plus noise", "no BMI": "BMI removed"}
    B = csv("bmi_mechanism.csv", 0).loc[list(ren)]
    table(pd.DataFrame([{"Fill": ren[k], "Probability given to blanks": f"{r['risk of blank']:.1%}", "PR-AUC, everyone": f"{r['all']:.3f}", "PR-AUC, BMI recorded": f"{r['BMI observed']:.3f}"} for k, r in B.iterrows()]))
with b:
    callout("A tree can isolate one exact value. If every blank becomes the same number, it can single out the blanks. Any constant gives the gain, noise removes it, and among people who <b>have</b> a BMI the median-filled model is worse than having no BMI.")
    cb = csv("calibration_subgroups.csv").set_index("group").loc["No BMI recorded"]
    callout(f"<b>Group level, not individual.</b> Among the {int(cb.n)} development records with no BMI the model expected {cb.expected:.0f} strokes in total and {int(cb.strokes)} occurred, {cb.oe:.1f} times as many (approximate 95% interval {cb.oe_lo:.1f} to {cb.oe_hi:.1f}). This says the group total is too low. It does not say that each person in it is underestimated, and a blank BMI does not show that a person had a stroke: {int(cb.n - cb.strokes)} of the {int(cb.n)} did not. The Simulator does not ask for BMI and makes no correction.", "a")
    callout("The findings suggest a missing-data artifact. They do not prove why BMI was missing, and they do not prove conventional target leakage.", "r")

section("Then a second pattern appeared: age", "With the shortcut gone, one global threshold was applied to everyone. This is what it did.")
cells = {"0": [], "1": []}
blend = lambda v, c: f"background:rgba({c},{.10 + .62 * v:.2f});color:#F3F5F2"
top = "<th></th>" + "".join(f'<th scope="col">{BAND[k]}<br>{int(V.n[k]):,} people, {int(V.strokes[k])} strokes</th>' for k in BAND)
row1 = row2 = row3 = ""
for k in BAND:
    r = V.loc[k]
    row1 += f'<td style="{blend(r.sensitivity, "11,133,120")}"><b>{r.sensitivity:.0%}</b><small>{r.sens_lo:.0%}–{r.sens_hi:.0%}</small></td>' if r.strokes >= MIN_CASES else '<td class="na"><b>Too few cases</b><small>rate not shown</small></td>'
    row2 += f'<td style="{blend(r.flagged, "163,106,28")}"><b>{r.flagged:.0%}</b><small>{r.flag_lo:.0%}–{r.flag_hi:.0%}</small></td>'
    fa = T["false_alarm_by_age"][BAND[k]]
    row3 += f'<td style="{blend(fa, "163,106,28")}"><b>{fa:.0%}</b><small>of people without stroke</small></td>'
with panel("matrix"):
    html(f'<div style="overflow-x:auto"><table class="mx"><thead><tr>{top}</tr></thead><tbody><tr><th scope="row">Strokes caught<small>sensitivity</small></th>{row1}</tr><tr><th scope="row">People flagged<small>flagging rate</small></th>{row2}</tr><tr><th scope="row">False alarms<small>false-alarm rate</small></th>{row3}</tr></tbody></table></div>')
    st.caption("Average of four cross-validation repeats, threshold chosen inside training folds. Small figures are approximate 95% Wilson intervals. Darker means higher. The global threshold flags everyone in the 71+ group in this development evaluation.")
insight("A respectable overall sensitivity can hide <em>severe differences</em> across age groups.")
a, b = st.columns(2, gap="large")
with a, panel("agehist", paper=True):
    chart_title("Where the strokes are", f"{T['strokes']} strokes by five-year age band, development data.")
    e = T["age"]["edges"]
    f = fig()
    f.add_trace(go.Bar(x=[v + 2.5 for v in e[:-1]], y=T["age"]["strokes"], width=4.4, marker=dict(color=DAMBER, cornerradius=4), customdata=[[i, j - 1, p] for i, j, p in zip(e, e[1:], T["age"]["people"])],
                       hovertemplate="age %{customdata[0]}–%{customdata[1]}: %{y} strokes of %{customdata[2]} people<extra></extra>"))
    vline(f, 40.5, "40", "#465868")
    vline(f, 70.5, "70", "#465868")
    show(f, "agehist_c", 340, paper=True, legend=False, x="age (years)", y="stroke cases")
with b, panel("riskbox", paper=True):
    chart_title("What the model predicts", "Estimated probability of the stroke class by age group: median, middle 50%, and 5th to 95th percentile.")
    f = fig()
    for k, q in T["age"]["risk_q"].items():
        f.add_trace(go.Box(x=[k], lowerfence=[q[1]], q1=[q[3]], median=[q[4]], q3=[q[5]], upperfence=[q[7]], fillcolor=rgba(DTEAL, .3), line=dict(color=DTEAL, width=2), name=k, hoverinfo="y"))
    hline(f, T["threshold"]["frozen"], f"research threshold {T['threshold']['frozen']:.1%}", DAMBER, "dash")
    show(f, "riskbox_c", 340, paper=True, legend=False, x="age group", y="estimated probability", yfmt=".0%")
disclaimer()
