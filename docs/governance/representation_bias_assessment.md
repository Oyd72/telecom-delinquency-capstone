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

## Pipeline integration

The diagnostic runs after the processed model-ready dataset has passed its blocking Great Expectations validation. In the Prefect flow it is a non-blocking analytical control: successful execution and report generation are required, but the magnitude of observed group differences is not converted automatically into a pipeline rejection decision.

Outputs are written to:

- `reports/tables/representation_bias_by_group.csv`
- `reports/tables/representation_bias_summary.json`

The privacy audit log also records that the diagnostic was executed, without storing row-level values or direct identifiers.

## Current status

The representation and bias suite is implemented in `src/monitoring/representation_bias_checks.py`, integrated into the Prefect flow, and covered by unit tests. Local execution still needs to be verified against the full project dataset before the Module 3 bias-detection requirement is marked complete.
