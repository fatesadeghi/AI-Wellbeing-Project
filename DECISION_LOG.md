# Decision Log

## Project

**AI-Based Early Detection of Changes in Well-Being and Quality of Life Among People Living Alone**

### Research Question

> **Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?**

---

## 1. Dataset Selection

### Decision

Use **PMData** as the primary dataset.

### Why?

The project requires both:

* observable behavioral/sensor measurements, and
* self-reported well-being measurements.

PMData provides both types of information for the same participants, making it suitable for investigating whether changes in observable daily behavior are associated with changes in self-reported well-being.

### Alternative considered

**CASAS Smart Home dataset.**

### Why was it not selected?

CASAS provides useful behavioral and sensor information, but it did not provide enough direct self-reported well-being measures for the intended analysis.

### Outcome

PMData was selected as the main dataset.

---

## 2. Personalized Baseline Instead of a Universal Baseline

### Decision

Construct a separate behavioral baseline for each participant.

### Why?

The project focuses on detecting **unusual behavior for an individual**, rather than identifying behavior that is unusual for the population as a whole.

People naturally differ in:

* activity level,
* exercise patterns,
* sleep duration,
* sleep quality,
* and other daily behaviors.

Therefore, the same absolute value may be normal for one participant but unusual for another.

A personalized baseline better matches the project's goal of individualized well-being monitoring.

### Alternative considered

Use a population-level average or universal threshold.

### Why was it not selected?

A universal threshold could incorrectly classify normal individual differences as abnormal behavior.

### Outcome

All behavioral deviations are evaluated relative to each participant's own baseline.

---

## 3. Chronological 50/50 Baseline Split

### Decision

Use the first 50% of each participant's observations to establish the personalized baseline and the second 50% as the analysis period.

### Why?

The project is concerned with detecting changes over time.

A chronological split allows the system to:

1. learn the person's earlier normal pattern, and
2. examine whether later observations deviate from that learned pattern.

This better represents the logic of an early-warning system than randomly mixing observations from the entire period.

### Alternative considered

Randomly split observations into baseline and analysis sets.

### Why was it not selected?

A random split could allow observations from the later period to contribute to the baseline, weakening the temporal interpretation of the analysis.

### Outcome

The first half is treated as the personalized baseline period and the second half as the analysis period.

---

## 4. Standardizing Behavioral Deviations with Z-scores

### Decision

Represent daily behavioral deviations using personalized Z-scores.

### Why?

Different behavioral variables have very different scales and levels of variability.

For example, steps, exercise duration, and sleep measures cannot be directly compared using their raw differences.

Standardization allows the deviation to be interpreted relative to the individual's typical variability.

### Alternative considered

Use raw differences from the baseline only.

### Why was it not selected?

Raw differences are difficult to compare across variables because each variable has a different scale.

### Outcome

Daily deviations are standardized relative to the participant's baseline distribution.

---

## 5. Primary Deviation Threshold: |Z| ≥ 2

### Decision

Use `|Z| ≥ 2` as the primary threshold for identifying meaningful behavioral deviations.

### Why?

The project needs to distinguish potentially meaningful departures from ordinary day-to-day fluctuations.

A two-standard-deviation threshold provides a relatively conservative criterion, reducing the number of small fluctuations classified as meaningful changes.

### Alternative considered

Use a lower threshold such as `|Z| ≥ 1`.

### Why was it not selected as the primary threshold?

A lower threshold identifies substantially more observations and may classify ordinary variation as meaningful deviation.

### Outcome

`|Z| ≥ 2` was retained as the primary change-detection criterion.

Using this threshold, **558 meaningful deviations** were detected across the 16 participants.

---

## 6. Sensitivity Analysis with |Z| ≥ 1

### Decision

Run an additional sensitivity analysis using `|Z| ≥ 1`.

### Why?

The choice of deviation threshold can influence the number of detected behavioral changes.

Testing a less conservative threshold allows us to determine whether the detection results are highly dependent on the selected threshold.

### Alternative considered

Use only the primary threshold.

### Why was this not sufficient?

Using only one threshold would provide no information about how sensitive the detected changes are to the threshold choice.

### Outcome

`|Z| ≥ 1` produced **2,743 deviations**, compared with 558 using `|Z| ≥ 2`.

The `|Z| ≥ 2` threshold remains the primary analysis, while `|Z| ≥ 1` is treated as a sensitivity analysis.

---

## 7. Minimum Sample Size: N ≥ 10

### Decision

Require at least 10 paired observations for participant-level correlation analysis.

### Why?

Correlation estimates based on very few observations can be unstable and difficult to interpret.

A minimum sample-size rule was therefore introduced to avoid interpreting extremely small samples as reliable participant-level relationships.

### Alternative considered

Allow correlations with very small N.

### Why was it not selected?

Very small samples can produce large correlation coefficients by chance and provide weak statistical evidence.

### Outcome

Relationships with fewer than 10 usable paired observations were excluded from inferential correlation analysis.

---

## 8. Pearson Correlation for Behavioral–Well-Being Associations

### Decision

Use participant-level Pearson correlation to examine the association between behavioral deviations and well-being measures.

### Why?

The analysis aims to determine whether higher or lower behavioral deviation scores are associated with higher or lower well-being scores within each participant.

Pearson correlation provides a straightforward measure of the strength and direction of a linear association.

### Alternative considered

Immediately build a machine-learning prediction model.

### Why was it not selected at this stage?

Before building a predictive model, it was necessary to establish whether meaningful within-person behavioral–well-being relationships could be observed in the dataset.

The initial goal was therefore exploratory and relationship-focused rather than predictive.

### Outcome

Participant-level correlations were calculated for behavioral × wellness variable pairs.

---

## 9. Initial Anomaly-Only Analysis

### Decision

Initially test behavioral–well-being relationships only on days classified as meaningful behavioral deviations.

### Why?

This directly matched the early-warning concept:

> When behavior becomes unusual for a person, does their well-being also change?

### Outcome

The anomaly-only analysis produced:

* 795 relationship tests,
* only 17 relationships meeting `N ≥ 10`,
* 1 raw significant relationship,
* 0 FDR-significant relationships.

### Interpretation

This was not interpreted as evidence that behavioral deviations have no relationship with well-being.

Instead, restricting the analysis to anomaly days substantially reduced the number of available observations and therefore reduced statistical power.

### Decision after evaluation

Expand the relationship analysis to the full analysis period while retaining personalized deviations.

---

## 10. Full-Period Personalized Deviation Analysis

### Decision

Analyze personalized behavioral deviations across the full analysis period rather than using only detected anomaly days.

### Why?

The full-period approach preserves more observations while maintaining the personalized-deviation framework.

It allows the analysis to ask a broader question:

> When a person's behavior is relatively higher or lower than their own usual pattern, is that deviation associated with their well-being?

### Alternative considered

Continue using anomaly days only.

### Why was it not selected as the primary analysis?

The anomaly-only approach produced too few observations for many participant-level relationships.

### Outcome

The full-period analysis provided substantially greater statistical power and produced multiple FDR-significant relationships.

---

## 11. Same-Day Analysis

### Decision

First examine behavioral deviation and well-being measured on the same day.

### Why?

Same-day analysis provides an initial test of whether departures from a person's normal behavioral pattern are associated with their current self-reported well-being.

It is also a straightforward first step before testing temporal relationships.

### Alternative considered

Start directly with lagged analysis.

### Why was same-day analysis performed first?

The same-day analysis establishes whether there is an observable relationship at all before asking whether behavioral deviation precedes a subsequent well-being change.

### Outcome

The final daily personalized-deviation analysis produced **44 FDR-significant relationships**.

---

## 12. Benjamini–Hochberg FDR Correction

### Decision

Apply Benjamini–Hochberg False Discovery Rate correction to the multiple relationship tests.

### Why?

The analysis examines many combinations of:

* participants,
* behavioral variables,
* and wellness variables.

When many statistical tests are performed, some relationships may appear significant simply by chance.

FDR correction was therefore used to control the expected proportion of false discoveries among the relationships identified as significant.

### Alternative considered

Use uncorrected p-values only.

### Why was it not selected?

Using only raw p-values would increase the risk of reporting false-positive relationships due to the large number of simultaneous tests.

### Outcome

FDR-adjusted p-values were used when determining the primary set of statistically significant relationships.

---

## 13. Interpretation of Significant Relationships

### Decision

Interpret statistically significant correlations as **associations**, not causal effects.

### Why?

The data are observational, and the correlation analysis does not establish that a behavioral change causes a change in well-being.

Other factors may contribute to the observed relationships.

### Alternative considered

Describe significant relationships as causal effects.

### Why was this rejected?

The study design and analysis do not provide sufficient evidence for causal inference.

### Outcome

Results are reported as behavioral–well-being associations.

---

## 14. Individual-Level Interpretation

### Decision

Retain participant-level results instead of relying only on an overall pooled relationship.

### Why?

The central concept of the project is personalized monitoring.

The analyses showed that relationships between behavioral deviations and well-being can differ substantially between individuals.

This individual heterogeneity is itself relevant to the research question.

### Alternative considered

Combine all participants into one population-level correlation.

### Why was it not selected as the sole analysis?

A pooled relationship could hide meaningful individual differences and would be less consistent with the personalized-baseline approach.

### Outcome

Participant-level relationships are retained and interpreted individually.

---

## 15. Focus on Sleep and Physical Activity

### Decision

Emphasize sleep and physical-activity variables when summarizing repeated significant relationships.

### Why?

These domains produced the most repeated significant associations with well-being in the participant-level analyses.

### Outcome

Sleep-related and physical-activity measures emerged as the main behavioral domains associated with well-being, while individual differences remained substantial.

---

## 16. One-Day Lagged Analysis

### Decision

Add a one-day lagged analysis in which behavioral deviation on day `t` is compared with well-being on day `t+1`.

### Why?

The project's goal includes **early detection**.

A same-day relationship does not establish that a behavioral deviation occurs before a later well-being change.

The lagged analysis introduces temporal ordering:

**Behavioral deviation → following-day well-being**

This provides a more direct test of whether behavioral changes could potentially act as an early-warning signal.

### Alternative considered

Use only same-day relationships.

### Why was this not sufficient?

Same-day associations cannot distinguish whether behavior precedes, follows, or changes simultaneously with the well-being measurement.

### Outcome

A one-day lagged analysis was performed across the participant-level behavioral and wellness variables.

---

## 17. Choice of One-Day Lag

### Decision

Use a one-day lag as the initial temporal window.

### Why?

The choice was informed by prior research in community/healthy adult populations showing prospective relationships between daily behavioral factors such as sleep or physical activity and next-day affect or well-being.

The one-day window also matches the daily resolution of PMData and provides a clear, interpretable temporal relationship without introducing a long or arbitrary prediction horizon.

### Alternative considered

Use multiple-day lags without prior justification.

### Why was this not selected?

Choosing a longer lag without a clear theoretical or empirical basis would introduce an additional methodological assumption.

### Outcome

The first temporal analysis uses:

**Behavior on day t → well-being on day t+1**

---

## 18. Lagged FDR Analysis

### Decision

Apply the same minimum-N criterion and FDR correction to the lagged analysis.

### Why?

The lagged analysis also involves many participant × behavioral × wellness tests, so the same protection against unstable estimates and multiple-comparison false positives is required.

### Outcome

The lagged analysis produced:

* 1,040 total relationship tests
* 871 relationships with `N ≥ 10`
* 64 raw significant relationships
* **4 FDR-significant relationships**

The four FDR-significant relationships were all observed for participant **p01** and next-day **readiness**.

---

## 19. Interpretation of the Lagged Findings

### Decision

Interpret the lagged significant relationships as preliminary evidence of potential next-day early-warning associations.

### Why?

The behavioral deviation was measured before the well-being measurement, providing temporal ordering.

However, the analysis is still correlational and participant-specific.

### Outcome

The four significant lagged relationships are treated as evidence supporting the **potential** of personalized behavioral monitoring as an early-warning approach, not as proof of a validated prediction system.

---

## 20. Early Detection vs. Prediction

### Decision

Do not claim that the current analysis constitutes a fully validated predictive AI model.

### Why?

The current work demonstrates:

* personalized baseline learning,
* behavioral deviation detection,
* association analysis,
* and temporal/lagged relationships.

However, it does not yet include a trained predictive model evaluated on unseen future outcomes using formal predictive performance metrics.

### Outcome

The project's current claim is limited to:

> **Personalized behavioral deviations show associations with well-being and provide preliminary evidence for potential early-warning signals.**

The project does not claim clinical prediction or diagnosis.

---

## 21. Handling Individual Heterogeneity

### Decision

Treat participant heterogeneity as an important finding rather than as noise to be removed.

### Why?

The project's central premise is that well-being and behavioral patterns are individual.

If different participants show different behavioral–well-being relationships, this supports the need for personalized monitoring.

### Outcome

The results are not interpreted as evidence that one behavioral variable has the same effect for everyone.

---

## 22. Final Methodological Framework

### Decision

Use the following overall framework for the current project:

**Personal data → Personalized baseline → Daily behavioral deviation → Change detection → Same-day association analysis → One-day lagged analysis → FDR correction → Individual-level interpretation**

### Why?

This sequence follows the research question from:

1. learning what is normal for an individual,
2. detecting departures from that normal pattern,
3. examining whether those departures relate to well-being,
4. and finally testing whether some deviations precede next-day well-being changes.

### Outcome

The framework provides a reproducible personalized early-warning analysis while avoiding unsupported claims of causality or clinical prediction.

---

## 23. Overall Research Conclusion from the Current Analysis

The analysis supports the following methodological conclusion:

> A personalized baseline can be used to identify deviations in an individual's daily behavioral patterns. These personalized deviations show associations with self-reported well-being, with sleep and physical activity emerging as the most repeatedly associated behavioral domains. A one-day lagged analysis identified a smaller set of next-day relationships, providing preliminary evidence that personalized behavioral deviations may have potential as early-warning signals.

At the same time, the findings demonstrate substantial individual heterogeneity and do not establish causality or validated predictive performance.

---

## 24. Limitations Recorded in the Decision Log

The following limitations are explicitly retained:

* The dataset contains 16 participants.
* The analysis is observational.
* Correlation does not establish causality.
* A one-day lag provides temporal ordering but is not equivalent to predictive modeling.
* The anomaly-only analysis had limited statistical power.
* Significant relationships varied across participants.
* The current framework does not constitute a clinical diagnostic system.

### Completed

* Dataset selection and preparation
* Personalized baseline construction
* Chronological baseline/analysis split
* Personalized behavioral deviation calculation
* Z-score standardization
* Primary change detection
* Sensitivity analysis
* Minimum-N filtering
* Same-day personalized deviation analysis
* Benjamini–Hochberg FDR correction
* One-day lagged analysis
* Lagged FDR correction
* Participant-level interpretation
* Reproducible Git-based project structure

### Documentation to Complete

* Finalize this Decision Log
* Document the methods, formulas, and implementation details in the research report
* Consolidate the final findings, limitations, and implications
