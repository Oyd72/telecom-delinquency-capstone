# Data dictionary change log

This file explains the material changes made to `docs/data_dictionary.md` and why they were made. Git history remains the technical record of the exact versions and diffs; this log is the readable record of the reasoning behind them.

## Working rules

- Start from the meaning provided with the dataset.
- Keep source documentation separate from our own analytical interpretation.
- Use the Kaggle description as the main semantic reference for these fields.
- Check that description against the actual CSV structure and observed values.
- Use cross-field and within-customer checks where they add useful evidence.
- If the documentation and the data conflict, record the uncertainty rather than forcing an answer.
- Do not turn a pattern seen in this dataset into a universal validation rule without a sound basis.
- Treat statistical anomaly detection as diagnostic unless stronger semantic or cross-field evidence supports action.
- Record important exclusions, derived fields, privacy treatments, and leakage concerns explicitly.
- Keep the dictionary, cleaning policy, and executable validation logic aligned.

## Change history

| Date | Field(s) | Change | Reason / evidence | Related commit |
|---|---|---|---|---|
| 11 Sep 2026 | All fields | Created the initial preliminary data dictionary. | Based mainly on column names, early exploratory work, and limited public documentation. Several field meanings remained provisional. | `fa54d7a6cddac6d446039b773e7980e8b8d71fa2` |
| 11 Sep 2026 | `msisdn` | Clarified that the field should not be used as a model predictor and should be retained only for grouping, deduplication, and split checks. | `msisdn` functions as an identifier-like field and is useful for customer-level analysis, but direct use as a predictive feature would create governance and leakage concerns. | `79891ceeafcd632322067eee427d6d7fed9566d1` |
| 13 Sep 2026 | All fields | Replaced preliminary interpretations with a domain-informed classification based on the supplied Kaggle data description and the actual CSV. | The Module 3 pipeline needs rules and transformations that are grounded in the observed data rather than in a descriptive schema alone. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | Schema | Corrected the active source schema from 37 to 36 columns. | The current CSV contains 36 fields and does not contain the former administrative `Unnamed: 0` column. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `pcircle` | Reclassified from a possible modelling / segmentation field to an excluded model feature. | All 209,593 records contain `UPW`, so the field has no predictive variation and cannot support geographic group comparison. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `aon` | Reclassified as a candidate only after cleaning, with hard validation for negative values and treatment of the separated upper contamination regime. | The definition confirms that `aon` means age on cellular network in days. Negative values are invalid, and the very large values show clear discontinuities and implausible within-customer jumps. | `1fc65ab5076f3462a338a745a6a04e28372bb734`, `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `last_rech_date_ma`, `last_rech_date_da` | Reclassified as elapsed-day fields that can be checked semantically, with negative values invalid and the separated upper regime treated as contamination. | Negative values conflict with elapsed-day semantics, and the upper tail forms a distinct implausible regime. | `1fc65ab5076f3462a338a745a6a04e28372bb734`, `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `cnt_ma_rech30`, `cnt_ma_rech90`, `cnt_da_rech30`, `cnt_da_rech90`, `cnt_loans30`, `cnt_loans90` | Standardised treatment as count fields. | Counts must be non-null, non-negative, and integer-valued. Rare large positive integers are monitored rather than rejected automatically. Fractional values are treated as contamination. | `1fc65ab5076f3462a338a745a6a04e28372bb734`, `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `fr_ma_rech30`, `fr_ma_rech90`, `fr_da_rech30`, `fr_da_rech90` | Kept the cautious description “frequency-related measure” and excluded all `fr_*` fields from the approved model feature set unless their encoding is clarified. | The source uses “frequency” without defining the calculation. These fields are not equivalent to the corresponding count fields. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `maxamnt_loans30` | Reclassified from an independent candidate predictor to a derived/redundant consistency check. | The source says principal options are 5 and 10 with repayment amounts 6 and 12, while observed core values are 0/6/12. The field can be reconstructed from count and amount for about 99.5% of records. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `maxamnt_loans90` | Reclassified as a derived/redundant consistency check rather than an independent predictor. | In the clean portion of the data, the field is reconstructible from `cnt_loans90` and `amnt_loans90`; mismatches concentrate in contaminated count records. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `medianamnt_loans30`, `medianamnt_loans90` | Marked as unresolved and excluded from modelling unless the encoding can be verified. | Observed values do not behave like straightforward medians of 6/12 loan amounts, and the public description does not explain the encoding. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `rental30`, `rental90`, `medianmarechprebal30`, `medianmarechprebal90` | Avoided treating negative values as automatic errors. | These are balance-like measures, and the source does not establish that negative balances are impossible. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `daily_decr30`, `daily_decr90` | Kept negative values as suspicious but not automatically invalid. | The fields are described as average daily spend, but refunds, adjustments, or undocumented encoding cannot be ruled out. | `1fc65ab5076f3462a338a745a6a04e28372bb734` |
| 13 Sep 2026 | `payback30`, `payback90` | Excluded from the approved model feature set pending point-in-time and leakage review. | These fields are usable only if they rely on completed prior loans and information available before the current five-day outcome. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | `msisdn` | Refined the privacy treatment to identifier minimisation / pseudonymisation. | The public dataset does not establish whether the values identify real individuals. The field is therefore used only where necessary for grouping and chronology, then removed or pseudonymised before model-ready output. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | Derived `prior_tx_count`, `is_repeat_customer` | Added as proposed derived features, subject to chronology controls and a selection-bias caveat. | Returning borrowers show a materially higher five-day repayment rate, but the approval process is unknown. They may therefore be a selected population. | `a63766797bbf4b737d18015e1088011fe054e757` |
| 13 Sep 2026 | Governance / audit | Aligned the dictionary with the formal cleaning policy. | `data/raw/` remains unchanged; transformations happen in `data/interim/`; runtime changes are logged, while Git history records changes to rules, code, configuration, and governance decisions. | `9dd57e36733ae5d32272d93b79cbbc4385dd26c1`, `a63766797bbf4b737d18015e1088011fe054e757` |

## Open items

The following points are still unresolved and should only be changed when better evidence is available:

- exact construction of the `fr_*` fields;
- encoding of `medianamnt_loans30` and `medianamnt_loans90`;
- final imputation strategy for values set to missing after contamination or semantic-invalid checks;
- point-in-time availability and leakage risk for `payback30` and `payback90`;
- final confirmation of the 6/12 reconstruction logic for `maxamnt_loans30` and `maxamnt_loans90`;
- any further contamination treatment supported by reproducible cross-field or longitudinal evidence.

## Relationship with Git history

This log explains the decisions. Git history remains the exact technical record. A future material change to the data dictionary or cleaning policy should have a meaningful commit message, a short entry here explaining the decision and evidence, and, where useful, a reference to the analysis or validation result that prompted it.

The runtime audit log serves a different purpose: it records which values or records were changed, removed, quarantined, or flagged during a pipeline run and which rule was applied.