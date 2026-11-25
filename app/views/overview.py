from components.core import *
from components.visuals import age_rings

RB = csv("retraining_bootstrap.csv").set_index(["metric", "model"])
R, T, A, F = results(), research(), age_gain(), final_test()
m, b = R.loc[MAIN], R.loc[BASE]

left, right = st.columns([1.08, 1], gap="large", vertical_alignment="center")
with left:
    hero("Research platform / 2026", "Does AI know more than your <em>birthday?</em>",
         "A leakage-tested study of classifying the recorded stroke label, age dominance and the limits of machine learning.")
    c1, c2, _ = st.columns([1, .85, .7])
    with c1, st.container(key="cta1"):
        link("lab", "Explore the research")
    with c2:
        link("simulator", "Test the model")
with right:
    age_rings(T["age"]["people"], T["age"]["strokes"])
    st.caption(f"Distance from the centre is age. Each amber point is one of the {T['strokes']} strokes in the development data; each teal point stands for 12 people without a stroke. Hover a ring for counts.")

callout("<b>What is predicted.</b> Each record in the dataset is labelled 1 (had a stroke) or 0 (did not). The model estimates the probability of label 1 and turns it into a class. The dataset has no dates or follow-up, so this is not a forecast of a future stroke.")
kpis([("5,110", "people in the dataset"), ("249", "stroke cases"), ("4.87%", "stroke prevalence"), ("Regularized logistic regression", "primary classifier, four inputs, no BMI")])

section("What four features add to age", "The same cross-validation, the same folds. Bars start at the score of random guessing for each metric.")
with panel("compare"):
    def metric(name, rnd, key, top, note):
        w = lambda v: 100 * (v - rnd) / (top - rnd)
        return (f'<div class="m"><h4>{name}</h4><span class="k">Age only</span><div class="bar"><i style="width:{w(b[key]):.1f}%;background:#D9A45B"></i></div><span class="n">{b[key]:.3f}</span>'
                f'<span class="k">Four features</span><div class="bar"><i style="width:{w(m[key]):.1f}%;background:#25C2B1"></i></div><span class="n">{m[key]:.3f}</span><small>{note}</small></div>')
    html('<div class="duo">' + metric("ROC-AUC", .5, "roc", 1, f"Difference {m.roc - b.roc:+.3f}, with two corrected standard errors of ±{b.droc_2se:.3f}. Random guessing 0.500, perfect 1.000.")
         + metric("PR-AUC", T["prevalence"], "pr", .3, f"Difference {m.pr - b.pr:+.3f}, with two corrected standard errors of ±{b.dpr_2se:.3f}. Random guessing {T['prevalence']:.3f}; the axis ends at 0.300.") + "</div>")
    st.caption("Mean over 20 folds (4 repeats of 5-fold cross-validation) on 4,088 development records. The four features were chosen after exploring this data, so the four-feature score is probably a little optimistic.")

insight("Age dominates the classification signal. The other features may add some ranking <em>within</em> the same age, but no model shows a robust advantage over the simpler baseline.")
callout(f"Among people within three years of the same age, the four-feature model ranks stroke pairs better than age alone by <b>{A.gain:.3f}</b> (95% interval {A.gain_lo:.3f} to {A.gain_hi:.3f}). That interval holds the four features fixed. When feature selection and training are repeated in every bootstrap, the gain is <b>{RB.loc[("match", "selected minus age"), "mean"]:+.3f}</b> and its interval ({RB.loc[("match", "selected minus age"), "lo"]:+.3f} to {RB.loc[("match", "selected minus age"), "hi"]:+.3f}) includes zero, so this is suggestive, not established. The PR-AUC and ROC-AUC differences above are within their uncertainty.")

ft = F["models"]["Logistic, 4 features"]
section("The final test, opened once", f"{F['n']:,} people the project had never scored, {F['strokes']} of whom had a stroke. The model and threshold were frozen first.")
kpis([(f"{ft['sensitivity']['k']} of {ft['sensitivity']['n']}", "strokes caught"), (f"{ft['fp']}", "false alarms"), (f"{ft['roc_auc']['value']:.3f}", "ROC-AUC, test"), (f"{ft['pr_auc']['value']:.3f}", "PR-AUC, test")], True)
dp = F["difference"]["pr_auc"]
callout(f"Against age alone the test PR-AUC difference is {dp['value']:+.3f}, 95% interval {dp['ci'][0]:+.3f} to {dp['ci'][1]:+.3f}. It includes zero, so the test does not establish an advantage either. The full table is in the Research Lab.")

section("How the conclusion was reached")
html('<ol class="journey">' + "".join(f"<li><b>{i}. {t}</b><span>{d}</span></li>" for i, (t, d) in enumerate([
    ("Dataset audit", "5,110 records, 201 without BMI. Missingness differs sharply by outcome."),
    ("Leakage investigation", "The first winning model was exploiting blank BMI values."),
    ("Model benchmarking", "Ten models, from age alone to boosted trees."),
    ("Fair evaluation", "Nested tuning, nested thresholds, corrected uncertainty."),
    ("Subgroup analysis", "One threshold behaves like an age cut-off."),
    ("Final interpretation", "A simple, calibrated model with stated limits.")], 1)) + "</ol>")
insight(f"Predictive performance must be read together with its cost: catching {m.sens:.0%} of strokes means only <em>{m.prec:.0%}</em> of flagged people had one.")
html(f"""<div class="foot"><div><h4>Research methodology</h4><p>80/20 stratified split. 4 × 5-fold cross-validation on the development part, with tuning and thresholds chosen inside training folds.</p></div>
<div><h4>Model limitations</h4><p>The group with no recorded BMI had more strokes than the model expected in total. Misses two thirds of strokes at 41–60. Not clinically validated.</p></div>
<div><h4>Dataset</h4><p>Supplied by the university, 5,110 records. Original source, collection process and the time between inputs and stroke are unknown.</p></div>
<div><h4>Version and context</h4><p>Model {predictor().version[:8]}, frozen 25 November 2026. Academic project. The test set was scored once, after the design was final.</p></div></div>""")
disclaimer()
