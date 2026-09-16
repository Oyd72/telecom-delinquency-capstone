# Representation and bias assessment

## Scope

The dataset does not contain usable demographic protected characteristics such as sex, age group, ethnicity, or disability. I therefore do not infer or manufacture them simply to make a conventional fairness analysis possible.

What can be assessed is narrower: whether the observed data are unevenly distributed across operational groups that actually exist in the dataset, and whether those differences matter for modelling.

## Operational slice used

The main comparison is **first-time versus returning borrower**, derived from the point-in-time customer-history logic already used elsewhere in the project.

This is useful because the borrower-history mix changes substantially over the observation window. It is not treated as a protected characteristic.

The automated check reports:

- record count and share by group;
- five-day delinquency rate by group;
- the difference in delinquency rate between returning and first-time borrowers;
- group composition over time;
- whether `pcircle` contains enough variation for any group comparison.

## Important limitation: left-censoring

Customer history before the start of the dataset is missing. A borrower who appears early in June may therefore be labelled as first-time even if earlier transactions exist outside the extract.

This matters because the apparent share of returning borrowers rises as the observation window gets longer. A change in group composition is therefore partly mechanical and should not be read as proof that the underlying borrower population changed in the same way.

## `pcircle`

`pcircle` is constant in the current dataset. Since it has only one non-missing value, it cannot support any meaningful regional or group comparison.

## Use of Fairlearn

Fairlearn `MetricFrame` is used to calculate grouped metrics in a reproducible way. It is not used to claim demographic parity, Equalized Odds, or another protected-group fairness result because the data needed for those claims are not available.

Using a fairness library does not turn an operational grouping into a protected class. The distinction is kept explicit throughout the project.

## How the automated check behaves

The script fails if the structure needed for the diagnostic is missing, for example if the target or grouping field is absent. It does not fail the ETL merely because the two operational groups differ. There is no defensible universal threshold that would allow this difference to be labelled automatically as “biased” or “unbiased.”

## Verified results

The full run on 15 September 2026 analysed all 150,767 model-ready records. The diagnostic also ran successfully inside the Prefect flow.

The groups are very uneven in size:

- first-time borrowers: 138,709 records, or about 92.00%;
- returning borrowers: 12,058 records, or about 8.00%.

Observed five-day delinquency was 18.44% for first-time borrowers and 4.83% for returning borrowers. The returning-minus-first-time difference is therefore about -13.61 percentage points.

That is a large descriptive difference, but it is not evidence of protected-group discrimination. The grouping is operational, and it is affected by the missing pre-June history.

The monthly breakdown reinforces the same caution. Returning borrowers account for about 5.03% of June records and 11.65% of July records. Their observed delinquency rate also differs from the first-time group in both months. At least part of the rise in returning-borrower share is expected simply because more customer history becomes visible as time passes.

The appropriate conclusion is therefore limited: borrower-history composition is an important modelling characteristic in this dataset, but the available data do not support a demographic fairness assessment.

## Pipeline integration

The representation diagnostic runs after the processed model-ready dataset has passed blocking validation. It is a non-blocking analytical control: the check itself must run successfully and produce its reports, but the size of the group difference does not automatically reject the pipeline.

Outputs are written to:

- `reports/tables/representation_bias_by_group.csv`
- `reports/tables/representation_bias_summary.json`

The privacy audit records that the diagnostic ran, without storing row-level values or direct identifiers.

## Status

The representation/bias suite is implemented in `src/monitoring/representation_bias_checks.py`, integrated into Prefect, covered by tests, and verified on the full model-ready dataset. For the information actually available in this dataset, the Module 3 bias-detection requirement is complete.

The main limitation is the data itself: protected demographic characteristics are not present, so the project cannot make claims about demographic fairness, Equalized Odds, or demographic parity.