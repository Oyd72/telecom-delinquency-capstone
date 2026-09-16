# Report evidence

This directory holds reproducible outputs from the pipeline and modelling scripts. The local project may contain more generated files than the GitHub repository.

## What can be committed

Outputs can be added to GitHub when they are aggregate or otherwise privacy-safe. Examples include:

- Great Expectations summaries;
- aggregate cleaning summaries;
- temporal and feature-selection summary tables;
- model-comparison and calibration summaries;
- aggregate representation/bias results;
- figures that do not expose individual records;
- privacy-audit evidence after checking that it contains no raw identifiers or row-level personal data.

For the Module 3 submission, the most useful evidence is likely to include:

- `tables/gx_raw_validation.json`
- `tables/gx_interim_validation.json`
- `tables/gx_processed_validation.json`
- `tables/cleaning_summary.json`
- `tables/model_dataset_summary.json`
- `tables/representation_bias_summary.json`
- selected temporal, feature-selection, and model-comparison summaries used in the presentation
- selected figures that help explain the pipeline or modelling choices

## What stays local

Do not commit:

- raw, interim, or processed row-level customer datasets;
- extracts containing `msisdn` or another direct identifier;
- ad hoc row-level debugging exports;
- screenshots or logs containing secrets, credentials, or unnecessary personal data.

The privacy audit log is generated under `reports/privacy/`. It is designed not to contain raw identifiers, but it should still be checked before it is included in a submission or committed to the repository.

## Source of truth

Generated reports are evidence of what happened in a run. The transformation logic itself lives under `src/`, while the reasoning, limitations, and governance decisions are documented under `docs/governance/`.