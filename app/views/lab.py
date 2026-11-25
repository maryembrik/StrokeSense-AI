from components.charts import *
from components.core import *

T, R, V, F = research(), results(), subgroups(), final_test()
m = R.loc[MAIN]
hero("Evidence and methods", "Ten models, <em>one honest comparison</em>", "Every score is for people the model had not seen: 4 repeats of 5-fold cross-validation on 4,088 development records with 199 strokes. The last tab shows the single test-set scoring, kept separate.")
tabs = st.tabs(["Benchmark", "ROC and PR curves", "Feature stability", "Calibration", "Age-matched ranking", "Subgroups", "Final test"], on_change="rerun", key="labtabs")

if tabs[0].open:  # draw only when visible so charts get the right width
    with tabs[0]:
        section("Model benchmark", "Sorted by the metric you choose. The primary model is highlighted because it was selected, not because it is proven better.")
        COL = {"PR-AUC": ("pr", "{:.3f}"), "ROC-AUC": ("roc", "{:.3f}"), "Sensitivity": ("sens", "{:.1%}"), "Specificity": ("spec", "{:.1%}"), "Precision": ("prec", "{:.1%}"), "F1": ("f1", "{:.3f}"),
               "Gap, PR": ("gap_pr", "{:+.3f}"), "Gap, ROC": ("gap_roc", "{:+.3f}"), "False alarms": ("fp", "{:.0f}")}
        s = st.selectbox("Sort by", list(COL), key="sort")
        S = R.sort_values(COL[s][0], ascending=s.startswith(("Gap", "False")))
        with panel("bench"):
            table(pd.DataFrame([{"Model": NAMES[i], **{n: f.format(r[c]) for n, (c, f) in COL.items()}} for i, r in S.iterrows()]), NAMES[MAIN])
        with st.expander("What the metrics mean"):
            st.markdown("- **PR-AUC**: how well strokes are ranked above others, with emphasis on precision. Random is 0.049.\n- **ROC-AUC**: chance that a random stroke case ranks above a random non-case. Random is 0.5.\n- **Sensitivity**: share of strokes caught. **Specificity**: share of people without stroke cleared. **Precision**: share of flagged people who had a stroke. **F1**: balance of precision and sensitivity.\n- **Gap**: training score minus unseen score. Near zero means no sign of overfitting.\n- Sensitivity, specificity, precision and F1 use a threshold chosen inside each training fold. PR-AUC and ROC-AUC are means over 20 folds.")
        with panel("diff"):
            chart_title("PR-AUC difference from the primary model", "Bars are plus or minus two corrected standard errors (Nadeau-Bengio). Repeated folds are not independent, so plain standard errors would be too small.")
            D = R.drop(index=MAIN).sort_values("dpr")
            f = fig()
            points(f, D.dpr.tolist(), [NAMES[i] for i in D.index], "difference", CYAN, lo=(D.dpr - D.dpr_2se).tolist(), hi=(D.dpr + D.dpr_2se).tolist(), horizontal=True, hover="%{y}: %{x:+.3f}")
            f.add_vline(x=0, line=dict(color=TEAL, width=2))
            show(f, "diff_c", 400, legend=False, x="PR-AUC minus primary model (left of the line: lower)")
            st.caption("Every interval includes zero, so no difference is established. The four features were chosen after exploring this data, so the primary model's PR-AUC is probably slightly optimistic.")

if tabs[1].open:  # draw only when visible so charts get the right width
    with tabs[1]:
        section("ROC and precision-recall", "With 4.87% strokes, a model can look strong on ROC while most of its alarms are wrong. The precision-recall curve shows that cost, which is why PR-AUC is the primary metric.")
        others = [k for k in NAMES if k not in (MAIN, BASE)]
        pick = st.multiselect("Add models", others, default=[], format_func=NAMES.get, key="models")
        a, b = st.columns(2, gap="medium")
        for col, kind in ((a, "roc"), (b, "pr")):
            with col, panel("c_" + kind):
                chart_title("ROC curve" if kind == "roc" else "Precision-recall curve", "Higher and further left is better." if kind == "roc" else "Higher is better.")
                f = fig()
                for k in [*pick, BASE, MAIN]:
                    col_, dash = STYLE[k]
                    c = T["curves"][k]
                    hv = f"mean-fold {'ROC' if kind == 'roc' else 'PR'}-AUC {R.loc[k, kind]:.3f}<br>%{{x:.0%}}, %{{y:.1%}}"
                    line(f, T["roc_x"] if kind == "roc" else T["pr_x"], c["tpr"] if kind == "roc" else c["prec"], NAMES[k], col_, dash, 3 if k in (MAIN, BASE) else 1.8, hover=hv)
                if kind == "roc":
                    line(f, [0, 1], [0, 1], "random", SLATE, "dot", 1.3)
                else:
                    hline(f, T["prevalence"], f"random {T['prevalence']:.1%}", MUTE, pos="bottom left")
                show(f, kind + "_c", 440, x="false-alarm rate" if kind == "roc" else "recall (strokes caught)", y="strokes caught" if kind == "roc" else "precision", xfmt=".0%", yfmt=".0%", xr=[0, 1], yr=[0, 1.01] if kind == "roc" else [0, max(.3, 1.08 * max(max(T["curves"][k]["prec"]) for k in [*pick, BASE, MAIN]))])
        st.caption("Curves pool the folds inside each repeat and average four repeats; precision is the best value at that recall or higher, which draws slightly above the raw curve. AUC values on hover are means over folds, a different calculation, so the two should not be mixed.")

if tabs[2].open:  # draw only when visible so charts get the right width
    with tabs[2]:
        section("Feature stability", "How often each variable was selected when selection was repeated inside 20 outer folds.")
        N = csv("nested_selection.csv", 0).sort_values("outer_folds_selected")
        four = predictor().features
        with panel("stab", paper=True):
            chart_title("Outer folds in which the variable was selected", "Teal: in the final model. Grey: not in the final model.")
            f = fig()
            f.add_trace(go.Bar(x=N.outer_folds_selected, y=[{"avg_glucose_level": "glucose"}.get(i, i.replace("_", " ")) for i in N.index], orientation="h", width=.5, marker=dict(color=[DTEAL if i in four else "#9AA8B3" for i in N.index], cornerradius=4),
                               text=[f"{v}/20" for v in N.outer_folds_selected], textposition="outside", textfont=dict(family="JetBrains Mono, Consolas, monospace", color=INK), hovertemplate="%{y}: %{x} of 20 folds<extra></extra>"))
            show(f, "stab_c", 400, paper=True, legend=False, x="folds (of 20)", xr=[0, 22.5])
        callout(f"Age is always selected; the others vary. Heart disease was chosen in only <b>{int(N.outer_folds_selected['heart_disease'])} of 20</b> folds and stays partly for clinical plausibility, not because it was consistently selected.", "a")

if tabs[3].open:  # draw only when visible so charts get the right width
    with tabs[3]:
        C = calib()
        L4 = C.loc["Logistic 4"]
        section("Calibration", "When the model says 10%, about 10% of those people should have had a stroke.")
        kpis([(f"{L4.slope:.2f}", f"calibration slope, 95% interval {L4.slope_lo:.2f} to {L4.slope_hi:.2f}; 1 is ideal"), (f"{m.brier:.4f}", "Brier score, lower is better"), (f"{m.mean_pred:.2%}", "mean predicted risk"), (f"{T['prevalence']:.2%}", "observed prevalence")], True)
        sel = st.multiselect("Add models", [k for k in NAMES if k != MAIN], default=["CatBoost"], format_func=NAMES.get, key="calmodels")
        with panel("cal"):
            chart_title("Predicted against observed risk", "Ten equal-size groups by predicted risk, four repeats pooled. Points above the dotted line are underestimates.")
            f = fig()
            line(f, [0, .35], [0, .35], "perfect calibration", SLATE, "dot", 1.4)
            for k in [*sel, MAIN]:
                col_, dash = STYLE[k]
                q = T["calibration"][k]
                f.add_trace(go.Scatter(x=q["pred"], y=q["obs"], name=NAMES[k], mode="lines+markers", line=dict(color=col_, dash=dash, width=3 if k == MAIN else 2), marker=dict(size=9 if k == MAIN else 7, color=col_), hovertemplate="predicted %{x:.1%}, observed %{y:.1%}<extra>" + NAMES[k] + "</extra>"))
            show(f, "cal_c", 430, x="mean predicted risk", y="observed stroke rate", xfmt=".0%", yfmt=".0%")
        table(pd.DataFrame([{"Model": NAMES[i], "Brier": f"{r.brier:.4f}", "Mean predicted": f"{r.mean_pred:.2%}", "Observed": f"{T['prevalence']:.2%}", "Slope (95% interval)": (lambda z: f"{z.slope:.2f} ({z.slope_lo:.2f}–{z.slope_hi:.2f})")(C.loc[{MAIN: "Logistic 4"}.get(i, i)]) if {MAIN: "Logistic 4"}.get(i, i) in C.index else "not computed"} for i, r in R.sort_values("brier").iterrows()]), NAMES[MAIN])
        callout("Calibration is good overall but not for people with no recorded BMI, whose risk is underestimated (see Subgroups).", "a")

if tabs[4].open:  # draw only when visible so charts get the right width
    with tabs[4]:
        section("Ranking people of the same age", "Take one person with a stroke and one without, of similar age. How much more often does the model rank the first higher than age alone would?")
        S = csv("age_matched_sensitivity.csv")
        with panel("am"):
            chart_title("Gain in correctly ranked pairs over age alone", "95% patient-level bootstrap intervals. Zero means no gain.")
            f, n = fig(), {0: "exact age", 1: "±1 year", 3: "±3 years", 5: "±5 years"}
            for w, c, sy, o in [("pair", TEAL, "circle", -.08), ("case", AMBER, "diamond", .08)]:
                z = S[S.weighting == w].sort_values("window")
                points(f, [i + o for i in range(len(z))], z.gain.tolist(), f"{w}-weighted", c, sy, lo=z.gain_lo.tolist(), hi=z.gain_hi.tolist(), hover="gain %{y:.3f}")
            f.add_hline(y=0, line=dict(color=MUTE, width=1.2))
            f.update_xaxes(tickvals=list(range(4)), ticktext=list(n.values()))
            show(f, "am_c", 400, x="how close in age the pair must be", y="gain in concordance")
        z = S[S.weighting == "pair"]
        table(pd.DataFrame([{"Match": n[int(r.window)], "Pairs per repeat": f"{r.pairs:,.0f}", "Model": f"{r.logistic:.3f}", "Age only": f"{r.age_only:.3f}", "Gain": f"{r.gain:+.3f}", "95% interval": f"{r.gain_lo:.3f} to {r.gain_hi:.3f}"} for _, r in z.iterrows()]))
        callout("This is a within-age ranking comparison. It is not a causal effect and does not prove clinical usefulness. Wider windows let age itself rank some pairs.")

if tabs[5].open:  # draw only when visible so charts get the right width
    with tabs[5]:
        section("Subgroup analysis", "One frozen approach, one threshold procedure, many groups. Intervals are approximate 95% Wilson intervals.")
        G = {"Age": list(BAND), "Sex": ["Women", "Men"], "Residence": ["Urban", "Rural"], "BMI": ["BMI recorded", "No BMI recorded"]}
        g = st.segmented_control("Group by", list(G), default="Age", key="grp") or "Age"
        Z = V.loc[G[g]]
        lab = [f"{BAND.get(i, i)}<br>{int(r.n):,} people<br>{int(r.strokes)} strokes" for i, r in Z.iterrows()]
        a, b = st.columns(2, gap="medium")
        with a, panel("sg1"):
            chart_title("Strokes caught and people flagged")
            f = fig()
            for k, lo, hi, nme, c, sy, o in [("sensitivity", "sens_lo", "sens_hi", "strokes caught", TEAL, "circle", -.09), ("flagged", "flag_lo", "flag_hi", "people flagged", AMBER, "diamond", .09)]:
                ok = [i for i in range(len(Z)) if k == "flagged" or Z.strokes.iloc[i] >= MIN_CASES]
                q = Z.iloc[ok]
                points(f, [i + o for i in ok], q[k].tolist(), nme, c, sy, lo=q[lo].tolist(), hi=q[hi].tolist(), hover=nme + " %{y:.1%}")
            f.update_xaxes(tickvals=list(range(len(Z))), ticktext=lab)
            show(f, "sg1_c", 390, yfmt=".0%", yr=[-.04, 1.06], xr=[-.5, len(Z) - .5])
        with b, panel("sg2"):
            chart_title("Predicted against observed stroke rate")
            f = fig()
            points(f, [i + .09 for i in range(len(Z))], Z.observed.tolist(), "observed", AMBER, "diamond", lo=Z.obs_lo.tolist(), hi=Z.obs_hi.tolist(), hover="observed %{y:.1%}")
            points(f, [i - .09 for i in range(len(Z))], Z.oof_risk.tolist(), "mean predicted", TEAL, hover="predicted %{y:.1%}")
            f.update_xaxes(tickvals=list(range(len(Z))), ticktext=lab)
            show(f, "sg2_c", 390, yfmt=".0%", xr=[-.5, len(Z) - .5])
        few = lambda r, s_: s_ if r.strokes >= MIN_CASES else "too few"
        table(pd.DataFrame([{"Group": BAND.get(i, i), "People": f"{int(r.n):,}", "Strokes": int(r.strokes), "Caught": f"{r.caught:.1f}", "Sensitivity (95%)": few(r, f"{r.sensitivity:.0%} ({r.sens_lo:.0%}–{r.sens_hi:.0%})"), "Flagged (95%)": f"{r.flagged:.0%} ({r.flag_lo:.0%}–{r.flag_hi:.0%})", "Predicted": f"{r.oof_risk:.1%}", "Observed (95%)": f"{r.observed:.1%} ({r.obs_lo:.1%}–{r.obs_hi:.1%})"} for i, r in Z.iterrows()]))
        if (Z.strokes < MIN_CASES).any():
            warn(f"A group here has fewer than {MIN_CASES} stroke cases. Its sensitivity is not shown because no reliable conclusion can be drawn.")

if tabs[6].open:
    with tabs[6]:
        section("Final test", f"Scored once on {F['scored_at'][:10]}, after the model, threshold and app were final. {F['n']:,} people, {F['strokes']} strokes. Nothing was changed afterwards.")
        L4, AO, D = F["models"]["Logistic, 4 features"], F["models"]["Age only"], F["difference"]
        iv = lambda c, d=3: f"{max(0, c[0]):.{d}f} to {c[1]:.{d}f}"
        pc = lambda r: f"{r['value']:.1%} ({max(0, r['ci'][0]):.0%}–{r['ci'][1]:.0%}), {r['k']} of {r['n']}"
        rows = [{"Metric": n, "Logistic, 4 features": f"{L4[k]['value']:.3f} ({iv(L4[k]['ci'])})", "Age only": f"{AO[k]['value']:.3f} ({iv(AO[k]['ci'])})", "Difference (95%)": f"{D[k]['value']:+.3f} ({D[k]['ci'][0]:+.3f} to {D[k]['ci'][1]:+.3f})"}
                for n, k in (("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"), ("Brier score", "brier"))]
        rows += [{"Metric": n, "Logistic, 4 features": pc(L4[k]), "Age only": pc(AO[k]), "Difference (95%)": "–"} for n, k in (("Sensitivity", "sensitivity"), ("Specificity", "specificity"), ("Precision", "precision"))]
        rows += [{"Metric": n, "Logistic, 4 features": g(L4), "Age only": g(AO), "Difference (95%)": "–"} for n, g in (("F1", lambda q: f"{q['f1']:.3f}"), ("False alarms", lambda q: str(q["fp"])), ("Missed strokes", lambda q: str(q["fn"])),
                 ("Threshold", lambda q: f"{q['threshold']:.3%}"), ("Expected vs observed strokes", lambda q: f"{q['expected_strokes']:.1f} vs {F['strokes']}"), ("Calibration slope", lambda q: f"{q['calibration_slope']:.2f}"))]
        table(pd.DataFrame(rows))
        st.caption("Ranking intervals: 4,000 patient bootstrap resamples. Rates: Wilson intervals. Each model uses its own threshold from the same development rule, so their sensitivities are not at equal alarm counts.")
        with panel("ft", paper=True):
            chart_title("Development estimate and test result", "Logistic, 4 features. Development: mean over 20 folds, no interval drawn. Test: one sample with its 95% interval.")
            f = fig()
            ks = [("PR-AUC", "pr", "pr_auc"), ("Precision", "prec", "precision"), ("Specificity", "spec", "specificity"), ("Sensitivity", "sens", "sensitivity"), ("ROC-AUC", "roc", "roc_auc")]
            points(f, [m[b] for _, b, _ in ks], [i - .12 for i in range(5)], "development", DTEAL, hover="development %{x:.3f}", ring="#F3F5F2")
            points(f, [L4[c]["value"] for *_, c in ks], [i + .12 for i in range(5)], "test, with 95% interval", DAMBER, "diamond", lo=[L4[c]["ci"][0] for *_, c in ks], hi=[L4[c]["ci"][1] for *_, c in ks], horizontal=True, hover="test %{x:.3f}", ring="#F3F5F2")
            f.update_yaxes(tickvals=list(range(5)), ticktext=[k for k, *_ in ks])
            show(f, "ft_c", 360, paper=True, xr=[0, 1], x="value")
        table(pd.DataFrame([{"Group": q["group"], "People": f"{q['n']:,}", "Strokes": q["strokes"], "Caught": q["caught"], "Flagged": f"{q['flagged'] / q['n']:.0%}", "False alarms": f"{q['false_alarms']} of {q['n'] - q['strokes']}", "Predicted": f"{q['mean_predicted']:.1%}", "Observed": f"{q['observed']:.1%}"} for q in F["subgroups"]]))
        q0, q1, q2, q3 = F["subgroups"][:4]
        callout(f"The age pattern repeated: {q0['caught']} of {q0['strokes']} strokes caught at 40 or under, {q1['caught']} of {q1['strokes']} at 41–60, {q2['caught'] + q3['caught']} of {q2['strokes'] + q3['strokes']} above 60, where nearly everyone is flagged. Several groups have fewer than {MIN_CASES} strokes, so counts are shown instead of rates.", "a")
        callout("Test PR-AUC is higher than the development estimate, but the development value sits inside the test interval. With 50 strokes this is consistent with sampling variation and is not evidence of improvement.")
disclaimer()
