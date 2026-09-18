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
| Final holdout | Completed: nonlinear models retain useful ranking but fail the pre-agreed stability threshold |
| Calibration | Isotonic selected on development data and supported by holdout sensitivity analysis; probability quality improves materially with minimal ranking loss |
| Final model selection | Prefer tuned Random Forest + isotonic calibration; retain calibrated XGBoost as challenger |


---

## 6. Development-only calibration assessment

**Script:** `src/models/assess_module4_calibration.py`

**Status:** **Completed**

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

### Output

| Model | Method | Mean ROC-AUC | Minimum ROC-AUC | Mean AP | Mean Brier | Mean ECE | Mean abs. calibration gap | Mean top-20% capture |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Random forest | Isotonic | 0.8332 | 0.8226 | 0.5737 | **0.1005** | **0.0464** | 0.0384 | 61.83% |
| Random forest | Uncalibrated | **0.8340** | **0.8234** | **0.5873** | 0.1021 | 0.0481 | **0.0291** | **61.92%** |
| Random forest | Platt | 0.8340 | 0.8234 | 0.5873 | 0.1045 | 0.0533 | 0.0385 | 61.92% |
| XGBoost | Isotonic | 0.8414 | 0.8195 | 0.5744 | **0.1026** | 0.0588 | 0.0502 | **61.21%** |
| XGBoost | Platt | **0.8424** | **0.8199** | **0.5867** | 0.1042 | **0.0586** | **0.0487** | 61.06% |
| XGBoost | Uncalibrated | 0.8424 | 0.8199 | 0.5867 | 0.1065 | 0.0734 | 0.0535 | 61.06% |

### Interpretation

Under the pre-defined selection rule, isotonic calibration has the best mean Brier score for both nonlinear models. For random forest it also gives a small ECE improvement over the uncalibrated model, although the mean absolute calibration gap becomes slightly worse. For XGBoost, isotonic improves Brier score and ECE relative to the uncalibrated model, but Platt has a marginally lower ECE and calibration gap.

The calibration gains are therefore real but modest rather than decisive. Isotonic also slightly reduces ROC-AUC and average precision, which is consistent with the stepwise mapping creating probability ties. Top-20% capture is essentially preserved.

The random-forest result is particularly nuanced: the uncalibrated model already has the smallest mean absolute calibration gap, while isotonic wins on the primary Brier criterion and slightly on ECE. This means calibration is not uniformly better on every measure.

### Decision

**Freeze isotonic calibration as the development-selected calibration method for both nonlinear candidates**, because the selection rule specified mean Brier score as the primary criterion before the results were observed.

Do not reinterpret this as evidence that isotonic is universally superior. The development results show a trade-off: improved probability error on average, with small losses in ranking metrics and mixed effects on calibration-in-the-large.

Any application of isotonic calibration to the already-opened final holdout will be treated as **confirmatory/sensitivity evidence**, not as a second independent final validation.

**Machine-readable outputs:**
- `reports/tables/module4_calibration_fold_metrics.csv`
- `reports/tables/module4_calibration_summary.csv`
- `reports/tables/module4_calibration_summary.json`

**MLflow experiment:** `module4_development_calibration`


---

## 7. Calibrated holdout sensitivity analysis

**Script:** `src/models/confirm_module4_calibration_holdout.py`

**Status:** **Completed**

### Purpose

Apply the development-selected **isotonic calibration** method to the already-opened final holdout as confirmatory/sensitivity evidence.

Chronology is preserved:

- base model training: through **6 July 2016**;
- calibration window: **7-13 July 2016**;
- evaluation: **14-23 July 2016** final holdout.

The script compares uncalibrated and isotonic-calibrated probabilities from the **same base-model train/calibration split**. This isolates the effect of calibration as far as possible.

### Output

| Model | Method | ROC-AUC | Average precision | Brier | ECE (10 bins) | Top-20% capture | Mean predicted risk | Observed delinquency rate | Abs. calibration gap | Beats prevalence Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Random forest | Uncalibrated | 0.8291 | 0.5777 | 0.1424 | 0.1389 | 54.04% | 34.67% | 20.78% | 0.1389 | Yes |
| Random forest | Isotonic | 0.8287 | 0.5667 | **0.1230** | **0.0293** | 54.04% | 17.85% | 20.78% | **0.0293** | Yes |
| XGBoost | Uncalibrated | 0.8270 | 0.5752 | 0.1488 | 0.1463 | 54.58% | 35.41% | 20.78% | 0.1463 | Yes |
| XGBoost | Isotonic | 0.8271 | 0.5689 | **0.1234** | **0.0291** | 54.51% | 17.94% | 20.78% | **0.0284** | Yes |

### Interpretation

The sensitivity result supports the development-period calibration decision. Isotonic calibration materially improves probability quality for both nonlinear candidates on the already-opened holdout.

For random forest, Brier score improves from 0.1424 to 0.1230 and ECE falls from 0.1389 to 0.0293. Mean predicted risk moves from 34.67% to 17.85%, much closer to the observed 20.78% delinquency rate. Top-20% capture is unchanged at 54.04%, while ROC-AUC changes only marginally.

For XGBoost, Brier score improves from 0.1488 to 0.1234 and ECE falls from 0.1463 to 0.0291. Mean predicted risk moves from 35.41% to 17.94%, again much closer to the observed rate. Top-20% capture is essentially unchanged and ROC-AUC is effectively stable.

The calibrated probabilities slightly underpredict average risk, but the calibration gap is far smaller than before calibration. This is a material improvement in probability reliability without meaningful loss of ranking performance.

### Decision

**Retain isotonic calibration as the preferred probability-calibration approach for both nonlinear candidates.**

The result strengthens the case for calibration but does not create a new independent validation result. The original final-holdout assessment remains the formal independent holdout evaluation. These calibrated results are confirmatory/sensitivity evidence.

The remaining model-selection question is now primarily the trade-off between random forest and XGBoost. Their calibrated holdout performance is very close. Random forest has a slightly better Brier score; XGBoost has a marginally higher top-20% capture. Any final choice should therefore return to the agreed complexity-versus-value principle rather than rely on tiny metric differences.

**Machine-readable outputs:**
- `reports/tables/module4_calibrated_holdout_metrics.csv`
- `reports/tables/module4_calibrated_holdout_summary.json`

**MLflow experiment:** `module4_calibrated_holdout_sensitivity`


---

## 8. Final model selection

### Decision rule

Where competing models deliver materially similar business-relevant performance, prefer the simpler and more interpretable option unless the more complex model provides a meaningful gain that justifies additional implementation, maintenance, computational, and explanation burden.

### Evidence considered

After development-only tuning and isotonic calibration:

- Random forest and XGBoost show very similar discrimination and top-20% capture.
- Random forest has a slightly better calibrated-holdout Brier score (0.1230 vs 0.1234).
- XGBoost has a marginally higher top-20% capture (54.51% vs 54.04%).
- The differences are too small to justify choosing XGBoost on predictive performance alone.
- Random forest is operationally and conceptually simpler than gradient boosting and is easier to explain to non-technical stakeholders, while still supporting SHAP-based global and local explanation.

### Selection

**Primary Module 4 model: tuned Random Forest with isotonic calibration.**

Frozen base-model parameters:

- `n_estimators=300`
- `max_depth=8`
- `min_samples_leaf=20`
- `max_features="sqrt"`
- `class_weight="balanced_subsample"`
- `random_state=42`

Calibration:

- isotonic regression;
- fitted on a chronologically later calibration window than the base-model training data and earlier than the scored evaluation period.

### Challenger retained

**XGBoost with isotonic calibration** remains the documented challenger model. It is not rejected as ineffective; it is simply not preferred because its small metric advantage on one operational measure does not justify the additional model complexity.

### Validation caveat

The selected random forest still missed the pre-agreed independent-holdout temporal-stability criterion before calibration. Calibration materially improved probability quality but does not remove that stability finding, because calibration does not change the underlying ranking model in a way that restores the original independent holdout result.

The selected model should therefore be described as the **preferred candidate for this assignment**, not as a fully production-validated model. Longer historical coverage would be needed to establish stronger temporal stability.

