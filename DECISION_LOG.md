# Decision Log

This document records the main methodological decisions made during the development of the project and the reasons for those decisions.

The project investigates whether AI can detect changes in a person's well-being by learning their daily habits and identifying unusual behavioral patterns.

The central research question is:

> **Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?**

Behavioral deviations are interpreted as potential signals of changes in well-being, not as diagnoses or proof of a health condition.

---

## 1. Dataset Selection

### Decision

Use **PMData** as the primary dataset.

### Why?

The project aims to detect meaningful changes in an individual's daily behavioral patterns and investigate whether these changes are associated with changes in self-reported well-being.

The **CASAS Smart Home dataset** mainly provides indirect behavioral information based on sensor events and presence in different areas of the home. For example, it can indicate that a person was present in the kitchen or bedroom, but it does not directly provide quantitative measures such as steps, exercise duration, or sleep characteristics.

PMData provides directly measurable daily behavioral and sleep-related indicators collected through wearable devices, including:

- steps
- exercise duration
- exercise distance
- exercise calories
- exercise average heart rate
- sleep duration
- deep sleep
- sleep restlessness
- sleep composition
- sleep revitalization
- sleep scores

It also provides self-reported well-being indicators for the same participants.

Therefore, PMData provides a strong match to the research question and supports personalized behavioral baselines.

### Alternative Considered

**CASAS Smart Home dataset.**

### Why Was It Not Selected?

CASAS provides valuable information about daily activity and movement within the home, but many measures are indirect indicators such as room presence or movement between locations.

For this project, behavioral changes needed to be represented using measurable quantitative indicators such as steps, exercise, and sleep characteristics.

### Outcome

PMData was selected as the primary dataset.

---

## 2. Behavioral Variable Selection

### Decision

Retain all **13 behavioral variables** as candidate variables throughout the project.

### Behavioral Variables

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

### Well-Being Indicators

The available well-being-related indicators are:

- fatigue
- mood
- readiness
- sleep_quality
- stress

### Why?

The project is intended to investigate whether different aspects of daily behavior and sleep may provide individualized signals of changes in well-being.

Pre-selecting a small number of behavioral variables could remove potentially informative individual patterns.

### Outcome

All 13 behavioral variables are retained as initial candidate variables.

The machine-learning stage may deactivate a variable for an individual when the predefined missing-data rule requires it, but the original 13-variable candidate set is preserved.

---

## 3. Personalized Baseline Instead of a Universal Baseline

### Decision

Construct a separate behavioral baseline for each participant.

### Why?

The project focuses on detecting unusual behavior **for an individual**, rather than identifying behavior that is unusual for the population as a whole.

People naturally differ in:

- activity level
- exercise patterns
- sleep duration
- sleep characteristics
- daily routines

Therefore, the same absolute value may be normal for one participant but unusual for another.

### Alternative Considered

Use a population-level average or universal threshold.

### Why Was It Not Selected?

A universal threshold could incorrectly classify normal individual differences as unusual behavior.

### Outcome

Behavioral deviations are evaluated relative to each participant's own baseline.

---

## 4. Chronological 50/50 Baseline Split

### Decision

Use the first 50% of each participant's chronological observations to establish the personalized baseline and the second 50% as the main analysis period.

### Why?

The project is concerned with detecting changes over time.

A chronological split allows the analysis to:

1. learn the person's earlier normal pattern, and
2. examine whether later observations deviate from that learned pattern.

This also avoids using later observations to define the earlier baseline.

### Alternative Considered

Randomly split observations into baseline and analysis sets.

### Why Was It Not Selected?

A random split could allow later observations to contribute to the baseline, weakening the temporal interpretation.

### Outcome

The first half is used for baseline construction and the second half for the main analysis period.

---

## 5. Personalized Deviation Measures

### Decision

Represent behavioral deviations using measures relative to each participant's baseline, including personalized Z-scores.

### Why?

The behavioral variables have different units, scales, and natural variability.

For example, steps, exercise duration, and sleep measures cannot be directly compared using raw differences alone.

Standardization allows deviations to be interpreted relative to the individual's typical variability.

### Outcome

The baseline stage produces:

- deviation from baseline median
- personalized Z-score
- absolute Z-score

These measures are used as part of the personalized behavioral analysis and later machine-learning feature construction.

---

## 6. Primary Behavioral Deviation Threshold

### Decision

Use **|Z| ≥ 2** as the primary threshold for identifying relatively large behavioral deviations.

### Why?

A two-standard-deviation threshold provides a relatively conservative criterion for distinguishing larger departures from ordinary day-to-day variation.

### Alternative Considered

Use a lower threshold such as **|Z| ≥ 1**.

### Why Was It Not Selected as the Primary Threshold?

A lower threshold identifies substantially more observations and may classify smaller ordinary fluctuations as meaningful deviations.

### Outcome

**|Z| ≥ 2** remains the primary change-detection threshold.

The initial change-detection analysis identified **558 deviations** across the 16 participants.

Importantly, the threshold-based event analysis is distinct from the later full-period correlation analyses, which retain continuous personalized deviation information rather than restricting all observations to the 558 detected events.

---

## 7. Sensitivity Analysis with |Z| ≥ 1

### Decision

Run an additional sensitivity analysis using **|Z| ≥ 1**.

### Why?

The choice of deviation threshold can influence the number of observations classified as meaningful deviations.

Testing a less conservative threshold allows the robustness of the findings to be examined under a broader definition of deviation.

### Outcome

The **|Z| ≥ 1** sensitivity analysis was retained as a separate analysis family.

It produced:

- 955 total relationship tests
- 603 valid tests
- 108 raw significant relationships
- **21 FDR-significant relationships**

The sensitivity analysis is interpreted separately because it uses a selected subset of observations.

---

## 8. Handling Incomplete and Unequal Data Availability

### Decision

Use available paired observations for each participant-level relationship rather than requiring complete data across all participants, variables, and days.

### Why?

Not all participants have complete observations for every behavioral and well-being variable.

The number of usable observations therefore differs between participant × behavioral × well-being relationships.

A complete-case approach would discard substantial usable information.

### Alternative Considered

Require complete data for all participants, variables, and days.

### Why Was It Not Selected?

Missingness differs across participants and variables. Requiring complete data would unnecessarily reduce the available sample.

### Outcome

Available paired observations are used for each relationship, subject to the minimum sample-size rule.

Missing behavioral observations are not automatically interpreted as zero activity.

---

## 9. Minimum Sample Size: N ≥ 10

### Decision

Require at least **10 usable paired observations** for participant-level correlation analysis.

### Why?

Correlation estimates based on very small samples can be unstable and difficult to interpret.

A minimum-N rule provides a basic reliability criterion while allowing incomplete datasets to remain usable.

### Outcome

Relationships with fewer than 10 usable paired observations are excluded from inferential correlation analysis.

---

## 10. Pearson Correlation

### Decision

Use participant-level Pearson correlation to examine behavioral–well-being relationships.

### Why?

The statistical objective is to examine whether measurable behavioral or sleep-related changes are associated with changes in well-being within an individual.

Pearson correlation provides a direct measure of the strength and direction of linear association.

### Alternative Considered

Immediately build a machine-learning prediction model.

### Why Was It Not Selected at This Stage?

Before developing predictive models, the project first needed to characterize within-person behavioral–well-being relationships.

The initial stage was therefore relationship-focused rather than prediction-focused.

### Outcome

Participant-level Pearson correlations are calculated across behavioral × well-being pairs.

---

## 11. Initial Anomaly-Only Analysis

### Decision

Initially examine behavioral–well-being relationships only on days classified as meaningful behavioral deviations.

### Why?

This directly matched the early-warning idea:

> When behavior becomes unusual for a person, does their well-being also change?

### Outcome

The anomaly-only analysis produced very few usable observations for many participant-level relationships.

The limited number of observations reduced statistical power.

### Interpretation

The anomaly-only result was not interpreted as evidence that behavioral deviations have no relationship with well-being.

Instead, the result demonstrated that restricting analysis to anomaly days substantially reduces the number of usable observations.

### Decision After Evaluation

Expand the primary statistical analysis to the full analysis period while retaining personalized deviation measures.

---

## 12. Full-Period Personalized Deviation Analysis

### Decision

Analyze personalized behavioral deviations across the full analysis period rather than using only detected anomaly days.

### Why?

The full-period approach preserves more observations while maintaining the personalized-deviation framework.

It allows the analysis to ask:

> When a person's behavior is relatively higher or lower than their own usual pattern, is that deviation associated with their well-being?

### Alternative Considered

Continue using anomaly days only.

### Why Was It Not Selected as the Primary Analysis?

The anomaly-only approach produced too few observations for many participant-level relationships.

### Outcome

The full-period approach was retained for the main participant-level statistical analyses.

---

## 13. Same-Day Analysis

### Decision

Examine behavioral variables and well-being measured on the same day.

### Why?

Same-day analysis provides an initial assessment of whether behavioral deviations and well-being are associated before introducing temporal ordering.

### Outcome

The final same-day analysis produced:

- 1,040 total relationship tests
- 900 valid tests
- 180 raw significant relationships
- **43 FDR-significant relationships**

---

## 14. Benjamini–Hochberg FDR Correction

### Decision

Apply the **Benjamini–Hochberg False Discovery Rate (FDR)** correction to multiple statistical tests.

### Why?

The project examines many combinations of:

- participants
- behavioral variables
- well-being indicators

With many simultaneous tests, some statistically significant results can occur by chance.

### Alternative Considered

Use uncorrected p-values only.

### Why Was It Not Selected?

Uncorrected p-values do not adequately account for multiple testing.

### Outcome

FDR-adjusted results are used as the primary significance criterion.

FDR correction is applied separately within each analysis family.

---

## 15. One-Day Lagged Analysis

### Decision

Add a one-day lagged analysis in which behavior on day **t** is compared with well-being on day **t+1**.

### Why?

The project aims to investigate potential early-warning signals, not only contemporaneous associations.

A lagged analysis introduces temporal ordering:

```text
Behavior on day t
        ↓
Well-being on day t+1
