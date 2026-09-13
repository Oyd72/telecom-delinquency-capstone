# Data dictionary change log

This file records material changes to `docs/data_dictionary.md` and the evidence supporting those changes. Git commit history remains the authoritative technical record of the exact file versions and diffs; this log provides the human-readable rationale for why the dictionary evolved.

## Change-management principles

- Preserve the original source meaning where documented.
- Distinguish source documentation from analytical inference.
- Prefer the Kaggle data description supplied for this dataset as the primary semantic source.
- Use the observed CSV structure and distributions to test whether documented semantics are consistent with the data.
- Use cross-field and within-customer consistency checks as additional evidence where helpful.
- When source documentation and observed data conflict, record the ambiguity rather than forcing an interpretation.
- Do not convert dataset-specific observations into universal validation rules without a defensible business or semantic basis.
- Keep statistical anomaly detection diagnostic unless field semantics or cross-field evidence support a stronger conclusion.
- Record material exclusions, derived fields, privacy treatments, and leakage concerns explicitly.
- Keep the dictionary, cleaning policy, and executable validation logic aligned; any material change to one should be reflected in the others where relevant.

## Change history

| Date | Field(s) | Change | Reason / evidence | Related commit |
|---|---|---|---|---|
| 11 Sep 2026 | All fields | Created the initial preliminary data dictionary. | Based mainly on column names, early exploratory work, and limited public documentation. Several field meanings remained provisional. | `fa54d7a6cddac6d446039b773e7980e8b8d71fa2` |
| 11 Sep 2026 | `msisdn` | Clarified that the field should not be used as a model predictor and should be retained only for grouping, deduplication, and split checks. | `msisdn` functions as an identifier-like field and is useful for customer-level analysis, but direct use as a predictive feature would create governance and leakage concerns. | `79891ceeafcd632322067eee427d6d7fed9566d1` |
| 13 Sep 2026 | All fields | Replaced preliminary interpretations with a domain-informed classification based on the supplied Kaggle data description and the actual CSV. | The Module 3 pipeline requires validation rules and transformations to be grounded in the data itself rather than in a descriptive schema alone. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | Schema | Corrected the active source schema from 37 to 36 columns. | The current CSV contains 36 fields and does not contain the former administrative `Unnamed: 0` column. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `pcircle` | Reclassified from a potential modelling / segmentation field to an excluded model feature. | All 209,593 records in the current dataset have the single value `UPW`, so the field carries no predictive variation and cannot support meaningful geographic fairness comparison. The value is retained in raw data for provenance only. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `aon` | Reclassified as a candidate only after cleaning, with hard validation for negative values and treatment of the separated upper contamination regime. | The supplied definition confirms that `aon` means age on cellular network in days. Negative values are semantically invalid. Very large values reach hundreds of thousands of days and show abrupt discontinuities and implausible within-customer jumps. | `1fc65ab5076f3462a338a745a6a04e28372bb734`, `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `last_rech_date_ma`, `last_rech_date_da` | Reclassified as semantically testable elapsed-day fields with invalid negative values and treatment of the separated contamination regime. | The supplied description identifies these as days associated with the last recharge. Negative values conflict with elapsed-day semantics; the upper tail also forms a distinct implausible regime. | `1fc65ab5076f3462a338a745a6a04e28372bb734`, `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `cnt_ma_rech30`, `cnt_ma_rech90`, `cnt_da_rech30`, `cnt_da_rech90`, `cnt_loans30`, `cnt_loans90` | Standardised treatment as genuine count fields. | Count fields must be non-null, non-negative, and integer-valued. Large positive integers are monitored but are not invalid solely because they are extreme. Fractional values are treated as contamination. | `1fc65ab5076f3462a338a745a6a04e28372bb734`, `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `fr_ma_rech30`, `fr_ma_rech90`, `fr_da_rech30`, `fr_da_rech90` | Retained the cautious description “frequency-related measure” and excluded all `fr_*` fields from the approved model feature set unless their encoding is clarified. | The source documentation uses “frequency” but does not define the exact construction. The fields are empirically distinct from corresponding `cnt_*` variables, and the 30-day fields also contain a separated contamination regime. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `maxamnt_loans30` | Reclassified from an independent candidate predictor to a derived/redundant consistency check. | The source says principal options are 5 and 10 with repayment amounts 6 and 12, while observed core values are 0/6/12. The supplied field can be reconstructed from `cnt_loans30` and `amnt_loans30` for about 99.5% of records; mismatches occur in the non-core anomalous regime. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `maxamnt_loans90` | Reclassified as a derived/redundant consistency check rather than an independent predictor. | In the clean portion of the data, the field is reconstructible from `cnt_loans90` and `amnt_loans90`; mismatches concentrate in contaminated count records. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `medianamnt_loans30`, `medianamnt_loans90` | Marked as unresolved and excluded from modelling unless the encoding can be verified. | Observed values do not behave like straightforward medians of 6/12 loan amounts, and the public description does not explain the encoding. The meaning is not inferred from the field name alone. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `rental30`, `rental90`, `medianmarechprebal30`, `medianmarechprebal90` | Avoided treating negative values as automatic errors. | These are balance-like measures. The supplied documentation does not establish that negative balances are impossible; negative values may therefore be legitimate and require investigation rather than automatic deletion. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `daily_decr30`, `daily_decr90` | Negative values retained as suspicious but not automatically invalid. | The fields are described as average daily spend. Negative values may indicate refunds, adjustments, or undocumented encoding; no authoritative business rule confirms that they are impossible. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `payback30`, `payback90` | Excluded from the approved model feature set pending point-in-time and leakage review. | Historical repayment measures may be predictive, but they must be confirmed to use only completed prior loans and information available before the current five-day outcome. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `msisdn` | Refined the privacy treatment to identifier minimisation / pseudonymisation. | The public dataset does not establish whether the values identify real individuals. The field is therefore treated proportionately as identifier-like: used only where necessary for grouping, chronology, repeat-customer analysis, and leakage-safe splitting, then removed or pseudonymised before model-ready output. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | Derived `prior_tx_count`, `is_repeat_customer` | Added as proposed derived features, subject to chronology controls and explicit selection-bias qualification. | Returning borrowers show a materially higher five-day repayment rate, but the underlying credit-approval process is unknown. Repeat borrowers may therefore be a selected population subject to different approval criteria. The association must not be interpreted causally. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | Governance / audit | Aligned the dictionary with the formal cleaning policy. | `data/raw/` remains immutable; transformations occur in `data/interim/`; altered or removed records/values are recorded automatically in the pipeline audit log, while Git history records changes to rules, code, configuration, and governance decisions. | `9dd57e36733ae5d32272d93b79cbbc4385dd26c1`, `a63766797bbf4b737d18015e1088011fe054e757` |

## Open items

The following points remain intentionally unresolved and should be updated only when further evidence is available:

- exact construction of the `fr_*` frequency-related fields;
- encoding of `medianamnt_loans30` and `medianamnt_loans90`;
- final imputation strategy for values converted to missing after contamination/semantic-invalid checks;
- point-in-time availability and leakage risk for `payback30` and `payback90`;
- final confirmation and implementation of the 6/12 reconstruction logic used for `maxamnt_loans30` and `maxamnt_loans90`;
- any additional contamination treatment justified by reproducible cross-field or longitudinal evidence.

## Audit relationship with Git history

This log is explanatory documentation. Git history remains the authoritative record for exact file changes. Each future material update to `docs/data_dictionary.md` or `docs/governance/data_cleaning_policy.md` should therefore be accompanied by:

1. a meaningful Git commit message;
2. a corresponding entry in this change log describing the affected field(s), decision, and evidence;
3. where relevant, a reference to the analysis output or validation result that triggered the change.

During pipeline execution, the runtime audit log separately records which records or values were altered, removed, quarantined, or flagged and the rule/treatment applied. This execution log is distinct from Git history.
