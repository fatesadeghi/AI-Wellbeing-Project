# Decision Log

## 1. Dataset Selection

### Decision

Use **PMData** as the primary dataset.

### Why?

The project requires both:

* observable behavioral/sensor measurements, and
* self-reported well-being measurements.

PMData provides both types of information for the same participants, making it suitable for investigating whether changes in observable daily behavior are associated with changes in self-reported well-being.

### Alternative Considered

**CASAS Smart Home dataset.**

### Why Was It Not Selected?

CASAS provides useful behavioral and sensor information, but it did not provide enough direct self-reported well-being measures for the intended analysis.

### Outcome

PMData was selected as the main dataset.

---

## 2. Personalized Baseline Instead of a Universal Baseline

### Decision

Construct a separate behavioral baseline for each participant.

### Why?

The project focuses on detecting unusual behavior **for an individual**, rather than identifying behavior that is unusual for the population as a whole.

People naturally differ in:

* activity level,
* exercise patterns,
* sleep duration,
* sleep characteristics,
* and other daily behaviors.

Therefore, the same absolute value may be normal for one participant but unusual for another.

A personalized baseline better matches the project's goal of individualized well-being monitoring.

### Alternative Considered

Use a population-level average or universal threshold.

### Why Was It Not Selected?

A universal threshold could incorrectly classify normal individual differences as abnormal behavior.

### Outcome

Behavioral deviations are evaluated relative to each participant's own baseline.

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

### Alternative Considered

Randomly split observations into baseline and analysis sets.

### Why Was It Not Selected?

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

Standardization allows deviations to be interpreted relative to the individual's typical variability.

### Alternative Considered

Use raw differences from the baseline only.

### Why Was It Not Selected?

Raw differences are difficult to compare across variables because each variable has a different scale and variability.

### Outcome

Daily behavioral deviations are standardized relative to the participant's baseline distribution.

---

## 5. Primary Deviation Threshold: |Z| ≥ 2

### Decision

Use **|Z| ≥ 2** as the primary threshold for identifying meaningful behavioral deviations.

### Why?

The project needs to distinguish potentially meaningful departures from ordinary day-to-day fluctuations.

A two-standard-deviation threshold provides a relatively conservative criterion and reduces the number of small fluctuations classified as meaningful changes.

### Alternative Considered

Use a lower threshold such as **|Z| ≥ 1**.

### Why Was It Not Selected as the Primary Threshold?

A lower threshold identifies substantially more observations and may classify ordinary variation as meaningful deviation.

### Outcome

**|Z| ≥ 2** was retained as the primary change-detection criterion.

Using this threshold, **558 meaningful deviations** were detected across the 16 participants.

---

## 6. Sensitivity Analysis with |Z| ≥ 1

### Decision

Run an additional sensitivity analysis using **|Z| ≥ 1**.

### Why?

The choice of deviation threshold can influence the number of detected behavioral changes.

Testing a less conservative threshold allows us to determine whether the detected changes are highly dependent on the selected threshold.

### Alternative Considered

Use only the primary threshold.

### Why Was This Not Sufficient?

Using only one threshold would provide no information about how sensitive the detected changes are to the threshold choice.

### Outcome

**|Z| ≥ 1** produced **2,743 deviations**, compared with 558 using |Z| ≥ 2.

The **|Z| ≥ 2** threshold remains the primary analysis, while **|Z| ≥ 1** is treated as a sensitivity analysis.

---

## 7. Minimum Sample Size: N ≥ 10

### Decision

Require at least **10 paired observations** for participant-level correlation analysis.

### Why?

Correlation estimates based on very few observations can be unstable and difficult to interpret.

A minimum sample-size rule was therefore introduced to avoid interpreting extremely small samples as reliable participant-level relationships.

### Alternative Considered

Allow correlations with very small N.

### Why Was It Not Selected?

Very small samples can produce large correlation coefficients by chance and provide weak statistical evidence.

### Outcome

Relationships with fewer than 10 usable paired observations were excluded from inferential correlation analysis.

---

## 8. Pearson Correlation for Behavioral–Well-Being Associations

### Decision

Use participant-level Pearson correlation to examine associations between behavioral deviations and well-being measures.

### Why?

The analysis aims to determine whether higher or lower behavioral deviation scores are associated with higher or lower well-being scores within each participant.

Pearson correlation provides a straightforward measure of the strength and direction of a linear association.

### Alternative Considered

Immediately build a machine-learning prediction model.

### Why Was It Not Selected at This Stage?

Before building a predictive model, it was necessary to establish whether meaningful within-person behavioral–well-being relationships could be observed in the dataset.

The initial goal was therefore exploratory and relationship-focused rather than predictive.

### Outcome

Participant-level correlations were calculated for behavioral × wellness variable pairs.

---

## 9. Initial Anomaly-Only Analysis

### Decision

Initially test behavioral–well-being relationships only on days classified as meaningful behavioral deviations.

### Why?

This approach directly matched the early-warning concept:

> When behavior becomes unusual for a person, does their well-being also change?

### Outcome

The anomaly-only analysis produced:

* 795 relationship tests,
* only 17 relationships meeting N ≥ 10,
* 1 raw significant relationship,
* 0 FDR-significant relationships.

### Interpretation

This result was not interpreted as evidence that behavioral deviations have no relationship with well-being.

Instead, restricting the analysis to anomaly days substantially reduced the number of available observations and therefore reduced statistical power.

### Decision After Evaluation

Expand the relationship analysis to the full analysis period while retaining personalized deviations.

---

## 10. Full-Period Personalized Deviation Analysis

### Decision

Analyze personalized behavioral deviations across the full analysis period rather than using only detected anomaly days.

### Why?

The full-period approach preserves more observations while maintaining the personalized-deviation framework.

It allows the analysis to ask a broader question:

> When a person's behavior is relatively higher or lower than their own usual pattern, is that deviation associated with their well-being?

### Alternative Considered

Continue using anomaly days only.

### Why Was It Not Selected as the Primary Analysis?

The anomaly-only approach produced too few observations for many participant-level relationships.

### Outcome

The full-period analysis provided substantially greater statistical power and produced multiple FDR-significant relationships.

---

## 11. Same-Day Analysis

### Decision

First examine behavioral deviation and well-being measured on the same day.

### Why?

Same-day analysis provides an initial test of whether departures from a person's normal behavioral pattern are associated with their current self-reported well-being.

It also provides a baseline for comparison with the later temporal analysis.

### Alternative Considered

Start directly with lagged analysis.

### Why Was Same-Day Analysis Performed First?

The same-day analysis establishes whether there is an observable relationship before asking whether behavioral deviation precedes a subsequent well-being change.

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

FDR correction was therefore used to reduce the risk of reporting false discoveries among the relationships identified as significant.

### Alternative Considered

Use uncorrected p-values only.

### Why Was It Not Selected?

Using only raw p-values would increase the risk of reporting false-positive relationships due to the large number of simultaneous tests.

### Outcome

FDR-adjusted results were used when determining the primary set of statistically significant relationships.

---

## 13. Interpretation of Significant Relationships

### Decision

Interpret statistically significant correlations as **associations**, not causal effects.

### Why?

The data are observational, and the correlation analysis does not establish that a behavioral change causes a change in well-being.

Other factors may contribute to the observed relationships.

### Alternative Considered

Describe significant relationships as causal effects.

### Why Was This Rejected?

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

### Alternative Considered

Combine all participants into one population-level correlation.

### Why Was It Not Selected as the Sole Analysis?

A pooled relationship could hide meaningful individual differences and would be less consistent with the personalized-baseline approach.

### Outcome

Participant-level relationships are retained and interpreted individually.

---

## 15. Interpreting Repeated Behavioral Patterns

### Decision

Emphasize behavioral domains that show repeated significant associations across participants when summarizing the findings.

### Why?

The goal is not to assume in advance that one specific behavioral variable is most important.

Instead, the analysis first identifies which behavioral domains repeatedly show associations with well-being.

Sleep

