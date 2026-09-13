# Data dictionary change log

This file records material changes to `docs/data_dictionary.md` and the evidence supporting those changes. Git commit history remains the authoritative technical record of the exact file versions and diffs; this log provides the human-readable rationale for why the dictionary evolved.

## Change-management principles

- Preserve the original source meaning where documented.
- Distinguish source documentation from analytical inference.
- Prefer the Kaggle data description supplied for this dataset as the primary semantic source.
- Use the observed CSV structure and distributions to test whether documented semantics are consistent with the data.
- When source documentation and observed data conflict, record the ambiguity rather than forcing an interpretation.
- Do not convert dataset-specific observations into universal validation rules without a defensible business or semantic basis.
- Keep statistical anomaly detection diagnostic unless field semantics or cross-field evidence support a stronger conclusion.
- Record material exclusions, derived fields, privacy treatments, and leakage concerns explicitly.

## Change history

| Date | Field(s) | Change | Reason / evidence | Related commit |
|---|---|---|---|---|
| 11 Sep 2026 | All fields | Created the initial preliminary data dictionary. | Based mainly on column names, early exploratory work, and limited public documentation. Several field meanings remained provisional. | `fa54d7a6cddac6d446039b773e7980e8b8d71fa2` |
| 11 Sep 2026 | `msisdn` | Clarified that the field should not be used as a model predictor and should be retained only for grouping, deduplication, and split checks. | `msisdn` functions as an identifier-like field and is useful for customer-level analysis, but direct use as a predictive feature would create governance and leakage concerns. | `79891ceeafcd632322067eee427d6d7fed9566d1` |
| 13 Sep 2026 | All fields | Replaced preliminary interpretations with a domain-informed classification based on the supplied Kaggle data description and the actual CSV. | The Module 3 pipeline requires validation rules and transformations to be grounded in the data itself rather than in a descriptive schema alone. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | Schema | Corrected the active source schema from 37 to 36 columns. | The current CSV contains 36 fields and does not contain the former administrative `Unnamed: 0` column. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `pcircle` | Reclassified from a potential modelling / segmentation field to an excluded model feature. | All 209,593 records in the current dataset have the single value `UPW`, so the field carries no predictive variation and cannot support meaningful geographic fairness comparison. The value is retained in raw data for provenance only. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `aon` | Reclassified as a candidate only after cleaning, with hard validation for negative values and diagnostic treatment of the extreme upper regime. | The supplied definition confirms that `aon` means age on cellular network in days. Negative values are therefore semantically invalid. Very large values reach hundreds of thousands of days and show abrupt discontinuities and implausible within-customer jumps, supporting a likely contamination / miscoding interpretation. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `last_rech_date_ma`, `last_rech_date_da` | Reclassified as semantically testable elapsed-day fields with invalid negative values and diagnostic upper-tail review. | The supplied description identifies these as days associated with the last recharge. Negative values conflict with elapsed-day semantics; some upper-tail values also form an implausible extreme regime. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `cnt_ma_rech30`, `cnt_ma_rech90`, `cnt_da_rech30`, `cnt_da_rech90`, `cnt_loans30`, `cnt_loans90` | Standardised treatment as genuine count fields. | Count fields should be non-null, non-negative, and integer-valued. Large positive values are monitored but are not invalid solely because they are extreme. Fractional values or a discontinuous extreme regime are treated as possible contamination. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `fr_ma_rech30`, `fr_ma_rech90`, `fr_da_rech30`, `fr_da_rech90` | Retained the cautious description “frequency-related measure” rather than equating these fields with recharge counts. | The source documentation uses “frequency” but does not define the exact construction. The fields are empirically distinct from the corresponding `cnt_*` variables and therefore should not be treated as synonyms. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `maxamnt_loans30` | Reclassified from a straightforward maximum-loan amount field to a documentation / encoding ambiguity. | The source says loan options are 5 and 10 with repayment amounts 6 and 12, while the observed core values are 0, 6, and 12. The field is therefore not validated against `{5,10}`; core observed values are retained provisionally and extreme non-core values are flagged. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `maxamnt_loans90` | Reclassified as effectively derived / redundant, subject to confirming the observed 6/12 encoding. | In the clean portion of the data, the field can be reconstructed for almost all rows from `cnt_loans90` and `amnt_loans90`. It is better used as a cross-field consistency check than as an independent predictor. | Planned follow-up revision |
| 13 Sep 2026 | `medianamnt_loans30`, `medianamnt_loans90` | Marked as unresolved and excluded from modelling unless the encoding can be verified. | Observed values do not behave like straightforward medians of 6/12 loan amounts, and the public description does not explain the encoding. The meaning should not be inferred from the field name alone. | Planned follow-up revision |
| 13 Sep 2026 | `rental30`, `rental90`, `medianmarechprebal30`, `medianmarechprebal90` | Avoided treating negative values as automatic errors. | These are balance-like measures. The supplied documentation does not establish that negative balances are impossible; negative values may therefore be legitimate and require investigation rather than automatic deletion. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `daily_decr30`, `daily_decr90` | Negative values retained as suspicious but not automatically invalid. | The fields are described as average daily spend. Negative values may indicate refunds, adjustments, or undocumented encoding; no authoritative business rule confirms that they are impossible. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `payback30`, `payback90` | Retained only subject to point-in-time and leakage review. | Historical repayment measures may be predictive, but they must be confirmed to use only completed prior loans and information available before the current five-day outcome. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `msisdn` | Refined the privacy treatment from generic identifier handling to identifier minimisation / pseudonymisation. | The public dataset does not establish whether the values identify real individuals. The field is therefore treated proportionately as identifier-like: used only where necessary for grouping, chronology, repeat-customer analysis, and leakage-safe splitting, then removed or pseudonymised before model-ready output. | Planned follow-up revision |
| 13 Sep 2026 | Derived `prior_tx_count`, `is_repeat_customer` | Added as proposed derived features, subject to chronology controls and explicit selection-bias qualification. | Returning borrowers show a materially higher five-day repayment rate, but the underlying credit-approval process is unknown. Repeat borrowers may therefore be a selected population subject to different approval criteria. The association must not be interpreted causally. | Planned follow-up revision |

## Open items

The following points remain intentionally unresolved and should be updated only when further evidence is available:

- exact construction of the `fr_*` frequency-related fields;
- encoding and modelling suitability of `medianamnt_loans30` and `medianamnt_loans90`;
- final treatment of the contamination regime in `aon`, recharge-date fields, and affected count/frequency variables;
- point-in-time availability and leakage risk for `payback30` and `payback90`;
- final confirmation of the 6/12 encoding used in `maxamnt_loans30` and `maxamnt_loans90`;
- final pipeline treatment and imputation strategy for values flagged as contaminated.

## Audit relationship with Git history

This log is explanatory documentation. Git history remains the authoritative record for exact file changes. Each future material update to `docs/data_dictionary.md` should therefore be accompanied by:

1. a meaningful Git commit message;
2. a corresponding entry in this change log describing the affected field(s), decision, and evidence;
3. where relevant, a reference to the analysis output or validation result that triggered the change.
