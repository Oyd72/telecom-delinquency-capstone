# Representation and bias assessment

## Purpose

This document explains how representation and bias are assessed for the telecom delinquency project using only attributes that genuinely exist in the source data.

The dataset does not provide usable demographic protected characteristics such as sex, age group, ethnicity, disability, or another legally protected group attribute. The project therefore does not infer, manufacture, or proxy such characteristics merely to make a conventional fairness analysis possible.

## What can be assessed

The automated suite focuses on representation diagnostics rather than claiming demographic fairness certification.

The primary operational slice is **first-time versus returning borrower**, derived from the point-in-time customer-history logic already used in the project. This slice is useful because borrower-history composition changes materially over the observation window and may affect model development or monitoring.

It is not treated as a protected characteristic.

The suite reports:

- number and share of records in each operational group;
- observed five-day delinquency rate by group;
- the difference in delinquency rate between returning and first-time borrowers;
- temporal group composition by month position;
- whether `pcircle` could support any meaningful group comparison.

## Left-censoring limitation

The first-time/returning distinction is affected by left-censoring because customer history before the dataset start is unavailable. A borrower appearing early in the extract may be classified as first-time even if earlier activity exists outside the observation window.

For that reason, representation changes in this slice are diagnostic rather than evidence of population bias. The project already observed that apparent repeat-customer share rises strongly with time as more prior history becomes visible.

## `pcircle`

`pcircle` is checked only to determine whether it contains enough variation to define groups. In the current dataset it is constant, so it cannot support comparative representation or fairness analysis.

## Use of Fairlearn

Fairlearn `MetricFrame` is used to calculate grouped metrics transparently and reproducibly. It is not used to produce Equalized Odds, demographic parity, or another protected-group fairness conclusion because the necessary protected characteristics are unavailable and this Module 3 check is applied to the data rather than a final production prediction stream.

This distinction matters: using a fairness library does not make an operational grouping a protected group, and the project does not present it as one.

## Automated interpretation rule

The suite does not automatically declare the dataset biased or unbiased. Differences in representation or delinquency prevalence are reported for review, together with the observation-window caveat.

A structural failure, such as a missing target or grouping field, should cause the diagnostic script to fail. Substantive differences between groups should not automatically fail the ETL pipeline because there is no defensible universal disparity threshold for this operational slice.

## Verified results

The full local run on 15 September 2026 analysed all 150,767 model-ready records and completed successfully inside the Prefect pipeline. The expanded unit-test suite also passed nine of nine tests.

The operational slice is highly imbalanced: 138,709 records (92.00%) are classified as first-time borrowers and 12,058 records (8.00%) as returning borrowers. The observed five-day delinquency rate is 18.44% for the first-time group and 4.83% for the returning group, a returning-minus-first-time difference of approximately -13.61 percentage points.

This difference is material as a descriptive pattern, but it is not treated as evidence of protected-group discrimination. The grouping is operational rather than demographic, and the returning-borrower classification is left-censored because customer history before the dataset start is unavailable.

The temporal diagnostics reinforce that limitation. Returning borrowers represent about 5.03% of June records but 11.65% of July records. At the same time, observed delinquency rates differ substantially between first-time and returning groups in both months. The rising returning-borrower share is therefore interpreted partly as a consequence of accumulating observable history rather than a stable population characteristic.

`pcircle` contains only one non-missing value and is therefore unusable for group comparison.

No automatic bias conclusion is generated from these results. The appropriate conclusion is narrower: the dataset contains a strong borrower-history composition effect that should remain visible in model development and monitoring, but the available data do not support a conventional protected-group fairness assessment.

## Pipeline integration

The diagnostic runs after the processed model-ready dataset has passed its blocking Great Expectations validation. In the Prefect flow it is a non-blocking analytical control: successful execution and report generation are required, but the magnitude of observed group differences is not converted automatically into a pipeline rejection decision.

Outputs are written to:

- `reports/tables/representation_bias_by_group.csv`
- `reports/tables/representation_bias_summary.json`

The privacy audit log also records that the diagnostic was executed, without storing row-level values or direct identifiers.

## Current status

The representation and bias suite is implemented in `src/monitoring/representation_bias_checks.py`, integrated into the Prefect flow, covered by unit tests, and verified against the full model-ready dataset. The Module 3 bias-detection requirement is therefore complete for the information actually available in this dataset.

The main limitation remains substantive rather than technical: protected demographic characteristics are not present, so this work cannot support claims about demographic fairness, Equalized Odds, or demographic parity.