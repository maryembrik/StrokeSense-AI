# TRIPOD+AI self-check

Our own assessment against the 27 items of Collins et al., BMJ 2024;385:e078378. Topics are paraphrased, not quoted; check the published checklist for exact wording. "No" and "Partly" are gaps, not errors to hide.

| Item | Topic | Reported | Where, or why not |
|---|---|---|---|
| 1 | Title | Partly | Report title is informal; the subtitle states the task |
| 2 | Abstract | Partly | Answer box, not a structured abstract |
| 3 | Background, intended use | Partly | Model card; healthcare context not developed |
| 4 | Objectives | Yes | Report, Task |
| 5 | Data sources and dates | No | Source and dates unknown |
| 6 | Setting, eligibility | No | Not documented in the dataset |
| 7 | Data preparation | Yes | Report 1-2 |
| 8 | Outcome | Partly | Label defined; assessment and timing unknown |
| 9 | Predictors | Partly | Listed; measurement timing unknown |
| 10 | Sample size | Partly | All available rows used; no calculation |
| 11 | Missing data | Yes | Report 2, Robustness |
| 12 | Analysis methods | Yes | Report 3-5, Robustness |
| 13 | Class imbalance | Yes | No resampling, reason in report 4 |
| 14 | Fairness | Partly | Subgroup results by age, sex, residence; no mitigation |
| 15 | Model output, threshold | Yes | Report, app Simulator |
| 16 | Development vs evaluation differences | Yes | Random split of one source |
| 17 | Ethics | No | No approval information came with the dataset |
| 18 | Open science | Partly | Code public; data not redistributed; no protocol or registration |
| 19 | Patient involvement | No | None |
| 20 | Participants | Partly | Counts and missingness; no flow diagram |
| 21 | Counts per analysis | Yes | Report 1, 5 |
| 22 | Full model | Yes | models/, coefficients in metadata.json |
| 23 | Performance | Yes | Report 5, with intervals |
| 24 | Model updating | Not applicable | Model frozen |
| 25 | Interpretation | Yes | Report 5 |
| 26 | Limitations | Yes | LIMITATIONS.md |
| 27 | Usability | Partly | App warnings; not for clinical use |
