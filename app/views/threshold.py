from components.charts import *
from components.core import *

T, R = research(), results()
H = T["threshold"]
t0, grid = H["frozen"], H["grid"]
hero("Interactive instrument", "One threshold. <em>Thousands of different outcomes.</em>", "The threshold decides how much estimated probability of the stroke class is enough to assign a record to it. Lower it and more strokes are caught, with more false alarms. No setting avoids the trade-off.")
with panel("ctl"), st.container(key="bigslider"):
    pct = st.slider("Probability threshold (%)", 1.0, 30.0, round(100 * t0, 1), .1, format="%.1f%%")
frozen = abs(pct - round(100 * t0, 1)) < .01
t = t0 if frozen else round(pct / 100, 3)
i = min(range(len(grid)), key=lambda j: abs(grid[j] - t))


def at(k, j=None):
    c = H["groups"][k]
    tp, fp = c["tp"][i if j is None else j], c["fp"][i if j is None else j]
    return tp, fp, c["P"] - tp, c["N"] - fp, c["P"], c["N"]


tp, fp, fn, tn, P, N = at("All")
prec = tp / (tp + fp) if tp + fp else 0
kpis([(f"{tp / P:.0%}", "sensitivity"), (f"{tn / N:.0%}", "specificity"), (f"{prec:.1%}", "precision"), (f"{2 * tp / (2 * tp + fp + fn):.3f}", "F1 score"), (f"{tp + fp:,.0f}", "people flagged"), (f"{fp:,.0f}", "false positives"), (f"{fn:,.0f}", "false negatives")], True)
st.caption(f"Development data: {P + N:,} records, {P} strokes, held-out cross-validation predictions averaged over four repeats. " + (f"The dial shows 5.2%; the exact frozen value {t0:.3%} is used." if frozen else f"The frozen research threshold is {t0:.3%}."))

X = [100 * v for v in grid]
ALL = [at("All", j) for j in range(len(grid))]
a, b = st.columns(2, gap="large")
with a, panel("sens"):
    chart_title("Strokes caught", "Sensitivity at every threshold.")
    f = fig()
    line(f, X, [q[0] / P for q in ALL], "sensitivity", TEAL, fill=.12, hover="threshold %{x:.1f}%: %{y:.0%}")
    points(f, [pct], [tp / P], "selected", WHITE, hover="%{y:.0%}")
    vline(f, 100 * t0, "frozen", AMBER, "dash")
    show(f, "sens_c", 290, legend=False, x="threshold (%)", y="sensitivity", yfmt=".0%", yr=[0, 1.04])
with b, panel("prec"):
    chart_title("How reliable an alert is", "Precision: share of flagged people who had a stroke.")
    f = fig()
    line(f, X, [q[0] / (q[0] + q[1]) if q[0] + q[1] else None for q in ALL], "precision", CYAN, fill=.10, hover="threshold %{x:.1f}%: %{y:.1%}")
    points(f, [pct], [prec], "selected", WHITE, hover="%{y:.1%}")
    vline(f, 100 * t0, "frozen", AMBER, "dash")
    show(f, "prec_c", 290, legend=False, x="threshold (%)", y="precision", yfmt=".0%", yr=[0, .4])
a, b = st.columns([1.5, 1], gap="large")
with a, panel("trade", paper=True):
    chart_title("False alarms against missed strokes", "Every threshold is one point on this curve. The diamond follows your slider.")
    f = fig()
    f.add_trace(go.Scatter(x=[q[1] for q in ALL], y=[q[2] for q in ALL], mode="lines", line=dict(color=DTEAL, width=3), customdata=X, hovertemplate="threshold %{customdata:.1f}%<br>%{x:,.0f} false alarms, %{y:.0f} missed<extra></extra>"))
    z = at("All", grid.index(t0))
    points(f, [z[1]], [z[2]], "frozen", INK, "circle-open", 15, hover="frozen: %{x:,.0f} false alarms, %{y:.0f} missed", ring=INK)
    points(f, [fp], [fn], "selected", DAMBER, "diamond", 19, hover="%{x:,.0f} false alarms, %{y:.0f} missed", ring="#F3F5F2")
    show(f, "trade_c", 360, paper=True, legend=False, x="false alarms (people without stroke flagged)", y=f"missed strokes (of {P})")
with b, panel("cm"):
    chart_title("Confusion matrix", "Rounded averages over four repeats.")
    html(f'<div class="cm"><div class="tp"><b>{tp:,.0f}</b><span>✓ stroke, flagged</span></div><div class="fn"><b>{fn:,.0f}</b><span>✕ stroke, missed</span></div><div class="fp"><b>{fp:,.0f}</b><span>! no stroke, flagged</span></div><div><b>{tn:,.0f}</b><span>no stroke, cleared</span></div></div>')
    callout(f"For every stroke caught, about <b>{fp / tp:.1f}</b> people without stroke are flagged." if tp else "Nothing is caught at this setting.", "a")

section("The same line, four different ages", "Nothing changes except the age of the people being judged.")
with panel("ages"):
    top, r1, r2, r3 = "<th></th>", "", "", ""
    cell = lambda v, rgb, sub: f'<td style="background:rgba({rgb},{.10 + .62 * v:.2f});color:#F3F5F2"><b>{v:.0%}</b><small>{sub}</small></td>'
    for k in ("40 or under", "41–60", "61–70", "71+"):
        a_, b_, c_, d_, p_, n_ = at(k)
        top += f'<th scope="col">{k}<br>{p_ + n_:,} people, {p_} strokes</th>'
        r1 += cell((a_ + b_) / (p_ + n_), "163,106,28", f"{a_ + b_:,.0f} of {p_ + n_:,}")
        r2 += cell(a_ / p_, "11,133,120", f"{a_:.1f} of {p_}") if p_ >= MIN_CASES else '<td class="na"><b>Too few cases</b><small>rate not shown</small></td>'
        r3 += cell(b_ / n_, "163,106,28", f"{b_:,.0f} of {n_:,}")
    html(f'<div style="overflow-x:auto"><table class="mx"><thead><tr>{top}</tr></thead><tbody><tr><th scope="row">People flagged</th>{r1}</tr><tr><th scope="row">Strokes caught</th>{r2}</tr><tr><th scope="row">False-alarm rate</th>{r3}</tr></tbody></table></div>')
with panel("agecurve"):
    chart_title("Share flagged in each age group", "One vertical line cuts four very different curves.")
    f = fig()
    for k, c, d in zip(("40 or under", "41–60", "61–70", "71+"), ("#6F8599", "#9DB6F5", AMBER, TEAL), ("dot", "dashdot", "dash", "solid")):
        g = H["groups"][k]
        line(f, X, [(u + v) / (g["P"] + g["N"]) for u, v in zip(g["tp"], g["fp"])], k, c, d, hover="%{y:.0%}")
    vline(f, pct, f"{pct:.1f}%", WHITE)
    show(f, "agecurve_c", 340, x="threshold (%)", y="people flagged", yfmt=".0%", yr=[0, 1.14], unified=True)
callout("The estimated probability follows age so closely that a single cut-off separates age groups more than it separates people. Age-specific thresholds were not added: they would be another unvalidated layer.")
callout(f"The slider and the age groups on this page are illustrations. The model uses one fixed threshold, {t0:.3%}, for everyone, and nothing on this page changes it.")
warn("The selected threshold is a research operating point derived from development data. It has not been clinically validated.")
st.caption(f"Because the slider is read against the same predictions, these figures are slightly optimistic. With thresholds chosen on separate inner data, sensitivity was {R.loc[MAIN, 'sens']:.1%} and precision {R.loc[MAIN, 'prec']:.1%}. Test-set results are separate, in the Research Lab.")
disclaimer()
