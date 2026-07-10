# SOXX Constituent Alpha Attribution Design

## Goal

Determine whether lagged single-name semiconductor signals contain stable information for SOXX before evaluating their use in SOXL, and measure how much the current probability-budget execution policy censors the model signal.

## Research Layers

1. Constituent attribution: build lagged 20-day raw and SOXX-residual momentum for each qlib-managed SOXX constituent.
2. SOXX aggregation: compare current-weight, equal-weight, residual-weighted, and breadth signals against future SOXX open-to-open returns.
3. Leveraged mapping: compare actual SOXL five-day open-to-open returns with the exact daily-reset path generated from realized SOXX open-to-open returns.
4. Execution censoring: read the current hardware-cycle position artifact through the qlib recorder and measure raw-zero ratio, floor ratio, probability ranking, and trade compression.

## Validation

All signal directions are fitted on the train segment only. Validation and test report oriented Rank correlation, top-minus-bottom quartile return spread, directional AUC, and sign stability. Five-day labels also use five non-overlapping calendar offsets; the card reports their positive-sign ratio and correlation range and does not promote on raw overlapping correlation alone.

## Promotion Boundary

The current SOXX constituent store backfills current holdings through history and is therefore survivorship-biased. The card is diagnostic and must return `not_promoted_missing_pit_holdings` until lagged point-in-time holdings exist. Execution constraints are not relaxed by this experiment.

## Artifacts

The notebook, workflow, summary, conclusion, and compact metric tables remain in the card root. The daily attribution panel and sourced execution positions are recorder-only.
