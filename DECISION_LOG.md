# Decision Log

This document records the main methodological decisions made during the development of the project, including the alternatives considered, the reasons for selecting each approach, and the outcomes of the final analyses.

The project investigates whether AI can detect changes in a person's well-being by learning their daily behavioral patterns and identifying deviations from their usual behavior.

## Central Research Question

> **Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?**

Behavioral deviations are interpreted as potential signals of changes in well-being, not as diagnoses or proof of a health condition.

---

# 1. Dataset Selection

## Decision

Use **PMData** as the primary dataset for the final analysis.

## Why?

The project requires both:

1. measurable daily behavioral and sleep-related information, and
2. self-reported well-being information.

PMData provides daily measurements related to activity, exercise, sleep, and heart rate, together with self-reported well-being indicators.

This combination supports the investigation of whether changes in an individual's observable daily behavior are associated with changes in their reported well-being.

## Alternative Considered

**CASAS Smart Home dataset.**

## Why Was It Not Selected?

CASAS provides valuable sensor-based information about activity and movement within the home. However, much of the behavioral information is indirect, such as room presence and movement between areas.

For this project, behavioral change needed to be represented using quantitative measures such as steps, exercise, and sleep characteristics.

## Outcome

PMData was selected as the primary dataset.

---

# 2. Behavioral Variable Selection

## Decision

Retain all **13 behavioral variables** as candidate variables throughout the project.

## Behavioral Variables

1. Steps
2. Exercise_Count
3. Exercise_Duration
4. Exercise_Distance
5. Exercise_Calories
6. Exercise_Avg_HR
7. Sleep_Hours
8. Sleep_Duration_Score
9. Deep_Sleep_Minutes
10. Sleep_Restlessness
11. Sleep_Composition
12. Sleep_Revitalization
13. Sleep_Score

## Well-Being Indicators

The available well-being-related indicators are:

- fatigue
- mood
- readiness
- sleep_quality
- stress

These variables represent the self-reported well-being information available in PMData. They are not treated as a complete measurement of all dimensions of well-being.

## Why?

Different individuals may show different relationships between their daily behavior and well-being.

Pre-selecting only a small subset of behavioral variables could remove potentially informative individual patterns.

## Outcome

All 13 behavioral variables remain part of the initial candidate feature set.

The machine-learning stage evaluates feature availability at the participant level and may deactivate a variable when the predefined missing-data rule requires it. The original 13-variable candidate framework is nevertheless preserved.

---

# 3. Personalized Baseline

## Decision

Construct a separate behavioral baseline for each participant.

## Why?

The project focuses on identifying behavior that is unusual **for the individual**, rather than unusual relative to a population-wide average.

Individuals naturally differ in:

- activity levels
- exercise patterns
- sleep duration
- sleep characteristics
- daily routines

Therefore, the same behavioral value may represent a normal day for one participant and a substantial deviation for another.

## Alternative Considered

Use population-level averages or universal thresholds.

## Why Was It Not Selected?

A universal baseline could fail to account for individual differences and could classify normal personal behavior as unusual.

## Outcome

Behavioral deviations are evaluated relative to each participant's own historical pattern.

---

# 4. Chronological 50/50 Baseline Split

## Decision

Use the first **50% of each participant's chronological observations** to construct the personalized baseline and the second 50% as the main analysis period.

## Why?

The research question concerns changes over time.

A chronological split allows the analysis to:

1. learn an individual's earlier behavioral pattern;
2. establish their baseline;
3. examine later behavior relative to that baseline.

It also prevents later observations from being used to define the earlier baseline.

## Alternative Considered

Randomly divide observations into baseline and analysis sets.

## Why Was It Not Selected?

A random split could mix earlier and later observations and weaken the temporal interpretation of behavioral change.

## Outcome

The first chronological half is used for baseline construction and the second half for the main personalized deviation analyses.

---

# 5. Personalized Deviation Measures

## Decision

Represent behavioral changes relative to each participant's baseline using personalized deviation measures, including Z-scores.

## Why?

The behavioral variables have different units, scales, and levels of natural variability.

For example, steps, exercise duration, and sleep measures cannot be directly compared using raw differences alone.

Standardized deviation measures allow behavioral changes to be interpreted relative to the individual's own typical variability.

## Baseline Outputs

The baseline stage produces measures including:

- deviation from baseline median
- personalized Z-score
- absolute Z-score

## Outcome

These personalized deviation measures form the basis of the behavioral change analyses and contribute to the later machine-learning feature construction.

---

# 6. Primary Behavioral Deviation Threshold

## Decision

Use **|Z| >= 2** as the primary threshold for identifying relatively large behavioral deviations.

## Why?

A two-standard-deviation threshold provides a relatively conservative criterion for identifying larger departures from an individual's baseline variability.

## Alternative Considered

Use a lower threshold such as **|Z| >= 1**.

## Why Was It Not Selected as the Primary Threshold?

A lower threshold captures smaller fluctuations and therefore produces a broader set of observations classified as deviations.

## Outcome

**|Z| >= 2** is retained as the primary behavioral deviation threshold.

The threshold-based deviation analysis is distinct from the later full-period correlation analyses. The correlation analyses retain continuous personalized deviation information rather than restricting all observations to threshold-defined deviation events.

---

# 7. Sensitivity Analysis

## Decision

Conduct an additional sensitivity analysis using **|Z| >= 1**.

## Why?

The definition of a behavioral deviation can influence the number of observations considered unusual.

Using a lower threshold provides a broader alternative definition and allows the robustness of the findings to be examined.

## Outcome

The |Z| >= 1 analysis is retained as a separate sensitivity-analysis family.

The final sensitivity analysis produced:

- 1,040 total relationship tests
- 887 valid tests with N >= 10
- 182 raw significant relationships
- 47 FDR-significant relationships

These results are interpreted separately from the primary same-day analysis because the sensitivity analysis uses a different deviation threshold.

---

# 8. Handling Incomplete and Unequal Data Availability

## Decision

Use available paired observations for each participant-level relationship rather than requiring complete observations across all participants, variables, and days.

## Why?

Data availability differs between participants and variables.

A complete-case strategy would discard usable observations simply because another variable or participant had missing data.

## Alternative Considered

Require complete observations across the entire dataset.

## Why Was It Not Selected?

This would substantially reduce the available information and would not reflect the actual structure of the PMData dataset.

## Outcome

Available paired observations are used for participant-level relationships, subject to the minimum sample-size requirement.

Missing behavioral observations are **not automatically interpreted as zero activity**.

---

# 9. Minimum Sample Size for Correlation Analysis

## Decision

Require at least **N >= 10 usable paired observations** for participant-level correlation analysis.

## Why?

Correlation estimates based on very small numbers of observations can be unstable.

The minimum-N rule provides a basic reliability criterion while still allowing incomplete participant-level datasets to contribute information.

## Outcome

Relationships with fewer than 10 usable paired observations are excluded from inferential correlation analysis.

---

# 10. Pearson Correlation

## Decision

Use participant-level **Pearson correlation** to examine behavioral-well-being relationships.

## Why?

The statistical stage aims to examine the direction and strength of linear association between personalized behavioral deviations and well-being indicators within individuals.

Pearson correlation provides a direct measure of this association.

## Alternative Considered

Begin directly with machine-learning prediction.

## Why Was It Not Selected Initially?

Before developing predictive models, it was necessary to characterize the within-person relationships between behavior and well-being.

The statistical analysis therefore preceded the machine-learning stage.

## Outcome

Participant-level Pearson correlations are calculated for behavioral × well-being combinations across the relevant analysis periods.

---

# 11. Initial Anomaly-Only Analysis

## Decision

Initially investigate behavioral-well-being relationships using only observations classified as meaningful behavioral deviations.

## Why?

This approach directly reflected the early-warning concept:

> When behavior becomes unusual for an individual, does their well-being also change?

## Outcome

The anomaly-only analysis resulted in too few usable observations for many participant-level relationships.

## Interpretation

This was not interpreted as evidence that behavioral deviations have no relationship with well-being.

Instead, it demonstrated an important methodological limitation: restricting the analysis to deviation events can substantially reduce the available sample for participant-level statistical analysis.

## Decision After Evaluation

Expand the primary statistical analysis to the full analysis period while retaining personalized deviation measures.

---

# 12. Full-Period Personalized Deviation Analysis

## Decision

Analyze personalized behavioral deviations across the full analysis period rather than restricting the primary analysis to threshold-defined anomaly days.

## Why?

The full-period approach preserves more observations while maintaining the personalized-baseline framework.

It allows the analysis to examine whether relatively higher or lower behavior compared with an individual's own baseline is associated with changes in well-being.

## Outcome

The full-period personalized deviation approach was retained for the main statistical analyses.

---

# 13. Same-Day Analysis

## Decision

Examine behavioral deviations and well-being measured on the same day.

## Why?

Same-day analysis provides the initial assessment of within-person behavioral-well-being associations before introducing temporal ordering.

## Final Outcome

The final same-day analysis produced:

- 1,040 total relationship tests
- 900 valid tests with N >= 10
- 180 raw significant relationships
- 43 FDR-significant relationships

The FDR-adjusted results are treated as the primary significance results.

---

# 14. Multiple-Testing Correction

## Decision

Apply the **Benjamini-Hochberg False Discovery Rate (FDR)** correction.

## Why?

The project evaluates many combinations of:

- participants
- behavioral variables
- well-being indicators
- analysis periods

Without multiple-testing correction, some statistically significant findings could occur by chance.

## Alternative Considered

Use uncorrected p-values as the primary criterion.

## Why Was It Not Selected?

Uncorrected significance does not adequately account for the number of simultaneous statistical tests.

## Outcome

FDR-adjusted results are used as the primary significance criterion.

The correction is applied separately within each analysis family.

---

# 15. One-Day Lagged Analysis

## Decision

Conduct a one-day lagged analysis in which behavior on day **t** is compared with well-being on day **t+1**.

## Why?

The project investigates the possibility that behavioral changes may provide signals before a subsequent change in reported well-being.

The lagged analysis therefore introduces temporal ordering:

```text
Behavior on day t
        |
        v
Well-being on day t+1
