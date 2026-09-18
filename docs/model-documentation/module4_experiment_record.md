# Module 4 experiment record

This document is the running record of Module 4 modelling experiments, outputs, interpretation, and decisions. Raw machine-readable outputs remain under `reports/tables/`; supporting figures are stored under `reports/figures/module4/`.

## 0. Agreed modelling and validation position

### Business use

The model predicts five-day delinquency risk to support prioritisation and follow-up by credit-risk and repayment teams. It is intended as decision support rather than as an autonomous decision-maker.

### Primary modelling population

- Labelled modelling period: **1 June-23 July 2016**
- Development period: **1 June-13 July 2016**
- Final untouched holdout: **14-23 July 2016**
- Post-23 July period: retained only for exploratory robustness, distribution, and score-behaviour checks; not used for supervised performance evaluation

### Acceptance and selection criteria

| Area | Working criterion |
|---|---|
| Discrimination | Final holdout ROC-AUC >= 0.80 |
| Operational capture | Top 20% of predicted-risk observations capture >= 50% of actual delinquencies |
| Temporal stability | Holdout ROC-AUC should not fall by more than 0.03 from development/CV performance |
| Calibration | Brier score should improve on a prevalence-only probability baseline, supported by calibration review |
| Complexity vs value | Prefer the simpler and more interpretable model where business-relevant performance is materially similar |
| Explainability | Selected model must support global and local explanation |
| Sensitivity | Reasonable input changes should not cause implausibly unstable predictions |
| Fairness | The dataset does not contain attributes that directly identify protected demographic groups or other clearly defensible fairness groups. During modelling, available variables are reviewed for direct or indirect traits that should be treated as protected; any such data will be handled in line with fairness principles. |

Calendar-position variables may be explored, but the short observation period does not support treating an apparent day-of-month pattern as a stable seasonal effect.

Customer-history variables `prior_tx_count` and `is_repeat_customer` are excluded from the primary Module 4 specification. They remain sensitivity variables because their incremental predictive value was negligible in Module 3, their interpretation is affected by the limited observation window, and most final-holdout observations belong to customers not previously observed in development.

---

## 1. Frozen chronological split validation

**Script:** `src/models/validate_module4_split.py`

**Purpose:** Verify the frozen population, chronological split, post-cutoff regime, and customer continuity before model comparison.

### Output

| Population | Purpose | Dates | Rows | Delinquent rows | Delinquency rate |
|---|---|---|---:|---:|---:|
| Development | Training + time-aware CV | 2016-06-01 to 2016-07-13 | 122,119 | 20,209 | 16.55% |
| Final holdout | Untouched final evaluation | 2016-07-14 to 2016-07-23 | 28,648 | 5,953 | 20.78% |
| Post-23 July | Exploratory robustness/distribution checks only | 2016-07-24 to 2016-08-21 | 58,825 | 0 | 0.00% |

Customer continuity in the final holdout:

- seen-customer rows: **3,437**
- unseen-customer rows: **25,211**
- unseen-customer share: **88.00%**

### Interpretation

The final holdout has a materially higher delinquency rate than development and is compositionally different. It is therefore a meaningful short-term temporal generalisation test rather than a random sample of the development population.

The very high unseen-customer share strengthens the case for relying on behaviour/account features that are available to both new and returning customers. It also reinforces the decision not to rely on customer-history variables in the primary specification.

### Decision

**Accepted.** Freeze the chronological split and keep the post-23 July block outside supervised performance evaluation.

**Machine-readable output:** `reports/tables/module4_split_summary.json`

---

## 2. Feature and leakage review

**Script:** `src/models/review_module4_features.py`

**Purpose:** Confirm that the 12-feature working specification is suitable for Module 4 model development before fitting new models.

### Output

- rows reviewed: **150,767**
- candidate predictors: **12**
- `pdate` used as predictor: **False**
- identifier present in model-ready data: **False**
- source label present in model-ready data: **False**
- near-deterministic target flags: **None**
- exact duplicate feature pairs: **None**
- direct fairness-name flags: **None**
- maximum predictor missingness: **2.02%**
- status: **Pass**

### Interpretation

The primary feature set passed the structural leakage screen. No identifier, source target, temporal control, duplicate predictor, or suspicious near-deterministic target relationship was found. Missingness is low enough to be handled inside leakage-safe model pipelines.

The fairness-name check is only a safeguard. It does not establish whether a variable is or is not an indirect proxy; that question remains part of later modelling and interpretation.

### Decision

**Accepted.** Proceed to model benchmarking with the 12-feature primary specification.

**Machine-readable outputs:**
- `reports/tables/module4_feature_leakage_review.csv`
- `reports/tables/module4_feature_leakage_review.json`

---

## 3. Initial candidate-model benchmark

**Script:** `src/models/benchmark_module4_models.py`

**Purpose:** Compare a transparent baseline with two nonlinear challengers using development data only. The final holdout remained unopened.

### Models

- Logistic regression — interpretable baseline
- Random forest — nonlinear ensemble with moderate complexity
- XGBoost — stronger nonlinear challenger already supported by Module 3 evidence

### Development-period result

| Model | Mean ROC-AUC | Minimum ROC-AUC | Mean average precision | Mean Brier | Mean top-20% capture | Minimum top-20% capture |
|---|---:|---:|---:|---:|---:|---:|
| Random forest | **0.8688** | 0.8413 | **0.6292** | **0.1024** | **66.12%** | 58.18% |
| XGBoost | 0.8662 | 0.8329 | 0.6246 | 0.1065 | 65.00% | **58.78%** |
| Logistic regression | 0.8017 | 0.7055 | 0.5171 | 0.1270 | 60.20% | 51.74% |

### Interpretation

Both nonlinear models materially outperform logistic regression on discrimination, ranking, and calibration-related error. Logistic regression remains useful as the transparent baseline, but its weakest fold is substantially less stable.

Random forest is marginally strongest overall at this stage, while XGBoost remains close enough to justify further tuning.

### Decision

**Keep all three for comparison, tune Random Forest and XGBoost only.** Logistic regression remains the reference baseline rather than a tuning priority.

**Machine-readable outputs:**
- `reports/tables/module4_candidate_model_fold_metrics.csv`
- `reports/tables/module4_candidate_model_summary.csv`
- `reports/tables/module4_candidate_model_summary.json`

**MLflow experiment:** `module4_candidate_model_benchmark`

---

## 4. Chronological hyperparameter tuning

**Script:** `src/models/tune_module4_models.py`

**Purpose:** Test whether modest tuning improves the two nonlinear challengers without opening the final holdout or conducting an unnecessarily large parameter search.

### Best configuration by model

| Model | Configuration | Mean ROC-AUC | Minimum ROC-AUC | Mean AP | Mean Brier | Mean top-20% capture | Minimum top-20% capture |
|---|---|---:|---:|---:|---:|---:|---:|
| Random forest | max_depth=8, min_samples_leaf=20, max_features=sqrt | **0.8711** | 0.8417 | **0.6316** | **0.1012** | **66.41%** | 58.31% |
| XGBoost | max_depth=5, learning_rate=0.05, min_child_weight=5 | 0.8688 | 0.8417 | 0.6287 | 0.1062 | 64.96% | 58.12% |

### Interpretation

Tuning produces only modest gains, which is a useful result in itself: the earlier benchmark was already reasonably specified. Random forest remains the strongest overall development-period model and its ranking is reinforced rather than overturned by tuning.

The tuned Random Forest improves mean ROC-AUC from 0.8688 to 0.8711 and mean top-20% capture from 66.12% to 66.41%, while also slightly improving Brier score.

### Decision

**Freeze three candidates for final evaluation:**
- logistic regression baseline;
- tuned random forest;
- tuned XGBoost.

No further tuning is performed before the untouched holdout is opened.

**Machine-readable outputs:**
- `reports/tables/module4_tuning_fold_metrics.csv`
- `reports/tables/module4_tuning_summary.csv`
- `reports/tables/module4_tuning_summary.json`

**MLflow experiment:** `module4_model_tuning`

---

## 5. Final holdout evaluation

**Script:** `src/models/evaluate_module4_holdout.py`

**Status:** **Completed**

The final holdout was used for evaluation only. No tuning was performed on holdout results.

### Output

| Model | ROC-AUC | Average precision | Brier | Prevalence-baseline Brier | Top-20% capture | Mean predicted risk | Observed delinquency rate | ROC-AUC drop vs development | Core acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic regression | 0.8016 | 0.4825 | 0.1808 | 0.1664 | 50.71% | 39.66% | 20.78% | 0.0002 | Fail |
| Random forest | 0.8308 | 0.5807 | **0.1490** | 0.1664 | 54.19% | 36.57% | 20.78% | 0.0403 | Fail |
| XGBoost | **0.8316** | **0.5822** | 0.1548 | 0.1664 | **55.05%** | 37.17% | 20.78% | 0.0371 | Fail |

### Interpretation

All three models retain useful ranking ability on the later holdout. The nonlinear models remain clearly stronger than logistic regression on ROC-AUC, average precision, and top-20% capture.

Neither tuned nonlinear model meets the pre-agreed temporal-stability threshold of a maximum 0.03 ROC-AUC drop from development performance. Random forest falls by about 0.040 and XGBoost by about 0.037. This indicates a modest but material reduction in discrimination on the later period.

Calibration is the more visible weakness. Both nonlinear models improve on the prevalence-only Brier baseline, but their mean predicted risks (about 36.6%-37.2%) substantially exceed the observed delinquency rate of 20.78%. Logistic regression is worse on this criterion: its Brier score is poorer than the prevalence-only baseline and its mean predicted risk is about 39.7%.

The holdout therefore does not support declaring any candidate fully accepted under the criteria set before evaluation. At the same time, the results do not show model failure in the broader sense: random forest and XGBoost still exceed the minimum discrimination and operational-capture thresholds. The main unresolved issues are temporal stability and probability calibration.

### Decision

**Do not select a final production candidate yet.**

The next modelling step is to assess a leakage-safe calibration strategy using development data only, freeze that calibration approach independently of the holdout results, and then evaluate the calibrated probabilities without additional parameter tuning. Model ranking remains secondary until calibration and stability are considered together.

**Machine-readable outputs:**
- `reports/tables/module4_final_holdout_metrics.csv`
- `reports/tables/module4_final_holdout_summary.json`

**MLflow experiment:** `module4_final_holdout_evaluation`

---

## Supporting visuals

The running record will use generated figures under `reports/figures/module4/`. Figures are reproducible from the machine-readable tables rather than manually edited.

Planned figures:
- development vs final-holdout delinquency rate;
- initial candidate-model ROC-AUC comparison;
- initial candidate-model top-20% capture comparison;
- tuned nonlinear-model comparison;
- final holdout ROC-AUC comparison;
- final holdout top-20% capture comparison.

---

## Running decision summary

| Stage | Decision |
|---|---|
| Validation design | Use chronological development/final-holdout structure; do not claim long-term temporal stability |
| Calendar effects | Explore cautiously; do not place day-of-month effects at the forefront |
| Post-23 July data | Use only for exploratory robustness/distribution analysis |
| Customer history | Exclude from primary specification; retain only for sensitivity analysis |
| Feature set | 12-feature primary specification passed leakage review |
| Baseline | Retain logistic regression for interpretability/reference |
| Challengers | Random forest and XGBoost |
| Development-period leader | Tuned random forest |
| Final holdout | Completed: nonlinear models retain useful ranking but fail the pre-agreed stability threshold; calibration remains unresolved |


---

## 6. Development-only calibration assessment

**Script:** `src/models/assess_module4_calibration.py`

**Status:** **Pending local execution**

### Purpose

Assess whether probability calibration can be improved for the two tuned nonlinear challengers without using the 14–23 July final holdout for method selection.

The experiment compares:

- uncalibrated probabilities;
- Platt scaling;
- isotonic calibration.

For each development-period evaluation fold, the base model is trained on earlier observations, the calibrator is fitted only on the immediately preceding seven calendar days, and performance is measured on a later development block.

### Selection rule

Calibration method selection is based on development-only evidence:

1. mean Brier score;
2. mean expected calibration error;
3. mean absolute calibration-in-the-large gap.

ROC-AUC, average precision, and top-20% capture are monitored to ensure calibration does not materially damage ranking performance.

### Methodological caution

The need to revisit calibration became particularly visible after the final holdout evaluation. Therefore, if the selected calibration method is later applied to the already-opened holdout, that result will be treated as **confirmatory/sensitivity evidence**, not as a second independent final validation.

**Expected machine-readable outputs:**
- `reports/tables/module4_calibration_fold_metrics.csv`
- `reports/tables/module4_calibration_summary.csv`
- `reports/tables/module4_calibration_summary.json`

**MLflow experiment:** `module4_development_calibration`
