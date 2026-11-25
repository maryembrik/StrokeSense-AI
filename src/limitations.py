"""One text for the report, the app and docs/LIMITATIONS.md."""
HEAD = ("Limitation", "Why this dataset cannot resolve it", "What would be needed", "Effect on interpretation")
NEEDS_DATA = [
    ("No external validation", "Every row comes from one source, so a split of it cannot show how the model behaves elsewhere.", "An independent dataset with the same inputs and outcome definition, scored once with the frozen model.", "Results describe this dataset. Performance elsewhere is unknown."),
    ("Only 50 strokes in the locked test set", "The dataset holds 249 strokes; re-scoring or re-splitting would break the single test.", "A new test sample with many more strokes.", "Test intervals are wide. A small real advantage over age, or none, are both compatible with the result."),
    ("One stroke among 674 development records under 18", "Nothing can be estimated from one case.", "A paediatric dataset with enough cases, probably with other predictors.", "Output under 18 is extrapolation from older records and cannot be checked."),
    ("Limited discrimination at younger ages", "Age dominates. At 41-60 the threshold catches a third of development strokes (19 of 57); catching 46 of 57 would cost about 591 false alarms among 1,264 people.", "More strokes at these ages and stronger inputs, for example measured blood pressure, cholesterol and medication.", "The threshold works almost as an age cut-off. The 12 test strokes at 60 or under are a warning sign, not a precise estimate."),
    ("Low precision and the false-alarm trade-off", "Strokes are 4.9% of records and four inputs separate classes weakly. Moving the threshold only trades precision against missed strokes.", "Better inputs, or a stated cost of each error from the intended users.", "Class 1 does not mean a stroke: 40 of 293 people put in class 1 in the test had one."),
]
NOT_RESOLVABLE = [
    ("No temporal information", "There are no dates for inputs or strokes and no follow-up.", "A cohort that is stroke-free at baseline, with dated measurements and a fixed horizon.", "The output is the probability of a recorded label, not the chance of a future stroke. Inputs may have been measured after the stroke."),
    ("Unknown collection process and reasons for missing BMI", "The file was supplied by the university without a description of sampling, measurement or missingness.", "The original protocol, or a study recording why values are missing.", "The population is undefined. The blank-BMI pattern may not exist elsewhere, so it is not used as a predictor."),
    ("Exploration-informed feature selection", "The four inputs were fixed after exploring the development data; that cannot be undone. Early integrity and target summaries also covered test labels, though no model used them.", "A pre-specified analysis on new data.", "0.190 is probably optimistic; 0.174 repeats selection inside the folds."),
    ("No independent preregistration of freeze before test", "The order is shown only by project hashes and timestamps, and the repository was published after scoring.", "A time-stamped registration before scoring a new test set.", "A reader must trust the project record for the order of events."),
]


def markdown():
    f = lambda rows: "| " + " | ".join(HEAD) + " |\n|" + "---|" * 4 + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"# Limitations\n\nNo external dataset was downloaded in this phase.\n\n## Need new data\n\n{f(NEEDS_DATA)}\n\n## Cannot be resolved from the current dataset\n\n{f(NOT_RESOLVABLE)}\n"
