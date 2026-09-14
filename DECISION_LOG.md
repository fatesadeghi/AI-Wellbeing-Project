# Decision Log

## 1. Dataset Selection

### Decision

Use **PMData** as the primary dataset.

### Why?

The project aims to detect meaningful changes in an individual's daily behavioral patterns and investigate whether these changes are associated with changes in self-reported well-being.

The **CASAS Smart Home dataset** mainly provides indirect behavioral information based on sensor events and presence in different areas of the home. For example, it can indicate that a person was present in the kitchen or bedroom, but it does not directly provide quantitative measures such as the number of steps taken, exercise duration, or sleep depth.

PMData provides more directly measurable daily behavioral and sleep-related indicators collected through wearable devices, including:

* steps,
* exercise duration,
* exercise distance,
* exercise calories,
* sleep duration,
* deep sleep,
* sleep restlessness,
* sleep composition,
* and sleep scores.

It also provides self-reported well-being measures for the same participants.

Therefore, PMData was a better fit for constructing personalized behavioral baselines and investigating whether measurable changes in daily behavior and sleep are associated with changes in well-being.

### Alternative Considered

**CASAS Smart Home dataset.**

### Why Was It Not Selected?

CASAS provides valuable information about daily activity and movement within the home, but many of its behavioral measures are indirect indicators, such as presence in a room or movement between locations.

For this project, we wanted behavioral changes to be represented by measurable quantitative indicators such as steps, exercise, sleep duration, and deep sleep.

PMData therefore provided a stronger match to the research question.

### Outcome

PMData was selected as the primary dataset.

---

## 2. Main Statistical Objective

### Decision

Use statistical analysis to investigate whether different behavioral and sleep-related elements have meaningful relationships with self-reported well-being.

### Why?

The main statistical question was:

> **Are changes in different measurable behavioral or sleep elements associated with changes in an individual's well-being?**

The project contains multiple behavioral variables and multiple well-being variables.

Therefore, the analysis needed to examine these relationships systematically rather than focusing on only one behavioral measure.

### Alternative Considered

Focus on a single behavioral variable or select specific variables before analyzing the data.

### Why Was It Not Selected?

Pre-selecting a small number of variables could overlook other behavioral or sleep-related elements that may show meaningful individual relationships with well-being.

### Outcome

The analysis examined multiple behavioral × wellness relationships at the participant level.

---

## 3. Personalized Baseline Instead of a Universal Baseline

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

A personalized baseline better matches the project's goal of individualized monitoring.

### Alternative Considered

Use a population-level average or universal threshold.

### Why Was It Not Selected?

A universal threshold could incorrectly classify normal individual differences as abnormal behavior.

### Outcome

Behavioral deviations are evaluated relative to each participant's own baseline.

---

## 4. Chronological 50/50 Baseline Split

### Decision

Use the first 50% of each participant's observations to establish the personalized baseline and the second 50% as the analysis period.

### Why?

The project is concerned with detecting changes over time.

A chronological split allows the analysis to:

1. learn the person's earlier normal pattern, and
2. examine whether later observations deviate from that learned pattern.

This also avoids using later observations to define what was considered normal earlier in time.

### Alternative Considered

Randomly split observations into baseline and analysis sets.

### Why Was It Not Selected?

A random split could allow observations from the later period to contribute to the baseline, weakening the temporal interpretation of the analysis.

### Outcome

The first half was used for personalized baseline construction and the second half for the main analysis.

---

## 5. Standardizing Behavioral Deviations with Z-scores

### Decision

Represent daily behavioral deviations using personalized Z-scores.

### Why?

The behavioral variables have different scales and different levels of natural variability.

For example, steps, exercise duration, and sleep measures cannot be directly compared using their raw differences from baseline.

Standardization allows each deviation to be interpreted relative to the individual's typical variability.

### Alternative Considered

Use raw differences from the baseline only.

### Why Was It Not Selected?

Raw differences are difficult to compare across variables because the variables have different units and scales.

### Outcome

Daily behavioral deviations were standardized relative to each participant's baseline distribution.

---

## 6. Primary Deviation Threshold: |Z| ≥ 2

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

## 7. Sensitivity Analysis with |Z| ≥ 1

### Decision

Run an additional sensitivity analysis using **|Z| ≥ 1**.

### Why?

The choice of deviation threshold can influence the number of detected behavioral changes.

Testing a less conservative threshold allows us to examine whether the detected changes are highly dependent on the selected threshold.

### Alternative Considered

Use only the primary threshold.

### Why Was This Not Sufficient?

Using only one threshold would provide no information about how sensitive the detected changes are to the threshold choice.

### Outcome

**|Z| ≥ 1** produced **2,743 deviations**, compared with 558 using |Z| ≥ 2.

The **|Z| ≥ 2** threshold remains the primary analysis, while **|Z| ≥ 1** is treated as a sensitivity analysis.

---

## 8. Handling Incomplete and Unequal Data Availability

### Decision

Account for incomplete data when performing participant-level statistical analyses.

### Why?

Not all participants had complete observations for every behavioral and well-being variable across the entire study period.

As a result, the number of usable paired observations was different for different participant × behavioral × wellness relationships.

The analysis therefore needed to use the available paired observations without discarding an entire participant simply because some observations were missing.

### Alternative Considered

Require complete data for all participants, variables, and days.

### Why Was It Not Selected?

A complete-case approach would discard a substantial amount of usable information because missing observations were not identical across participants and variables.

This would unnecessarily reduce the amount of data available for analysis.

### Outcome

Available paired observations were used for each relationship, while a minimum sample-size criterion was introduced to avoid relying on extremely small samples.

---

## 9. Minimum Sample Size: N ≥ 10

### Decision

Require at least **10 usable paired observations** for participant-level correlation analysis.

### Why?

Because data availability differed across participants and variables, some relationships had very few usable observations.

Correlation estimates based on very small samples can be unstable and difficult to interpret.

A minimum-N rule therefore provided a basic level of statistical reliability while allowing incomplete datasets to remain usable.

### Alternative Considered

Allow correlations with very small N.

### Why Was It Not Selected?

Very small samples can produce large correlation coefficients by chance and provide weak statistical evidence.

### Outcome

Relationships with fewer than 10 usable paired observations were excluded from inferential correlation analysis.

---

## 10. Pearson Correlation

### Decision

Use participant-level Pearson correlation to examine behavioral–well-being relationships.

### Why?

The main statistical objective was to determine whether changes in measurable behavioral or sleep-related elements were associated with changes in well-being within an individual.

Pearson correlation provides a straightforward measure of the strength and direction of a linear association.

### Alternative Considered

Immediately build a machine-learning prediction model.

### Why Was It Not Selected at This Stage?

Before developing a predictive model, it was necessary to establish whether meaningful within-person behavioral–well-being relationships could be observed in the dataset.

The first stage was therefore relationship-focused rather than predictive.

### Outcome

Participant-level correlations were calculated across behavioral × wellness variable pairs.

---

## 11. Initial Anomaly-Only Analysis

### Decision

Initially examine behavioral–well-being relationships only on days classified as meaningful behavioral deviations.

### Why?

This directly matched the early-warning idea:

> When behavior becomes unusual for a person, does their well-being also change?

### Outcome

The anomaly-only analysis produced:

* 795 relationship tests,
* only 17 relationships meeting N ≥ 10,
* 1 raw significant relationship,
* 0 FDR-significant relationships.

### Interpretation

This result was not interpreted as evidence that behavioral deviations have no relationship with well-being.

Instead, restricting the analysis to anomaly days substantially reduced the number of observations and therefore reduced statistical power.

### Decision After Evaluation

Expand the analysis to the full analysis period while retaining personalized behavioral deviations.

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

The full-period analysis provided substantially greater statistical power and produced multiple FDR-significant relationships.

---

## 13. Same-Day Analysis

### Decision

First examine behavioral deviation and well-being measured on the same day.

### Why?

The first statistical step was to determine whether measurable behavioral deviations were associated with well-being at all.

Same-day analysis provides a direct initial assessment of these relationships before introducing a temporal lag.

### Alternative Considered

Start directly with a lagged analysis.

### Why Was Same-Day Analysis Performed First?

Before asking whether behavior precedes a later well-being change, it was useful to first establish whether behavioral deviations showed any observable association with well-being.

### Outcome

The final same-day personalized-deviation analysis produced:

* 1,040 total relationship tests,
* 935 usable tests after data and validity filtering,
* 185 raw significant relationships,
* **44 FDR-significant relationships**.

---

## 14. Benjamini–Hochberg FDR Correction

### Decision

Apply the **Benjamini–Hochberg False Discovery Rate (FDR)** correction to the multiple statistical tests.

### Why?

The analysis examines many combinations of:

* participants,
* behavioral variables,
* and wellness variables.

When a large number of statistical tests are performed, some relationships may appear significant simply by chance.

Therefore, relying only on raw p-values would increase the risk of reporting false-positive findings.

### Alternative Considered

Use uncorrected p-values only.

### Why Was It Not Selected?

Uncorrected p-values do not adequately account for the large number of simultaneous tests performed in the analysis.

### Outcome

FDR-adjusted results were used to identify the primary statistically significant relationships.

The same multiple-testing principle was also applied to the lagged analysis.

---

## 15. Individual-Level Interpretation

### Decision

Retain participant-level results rather than relying only on a pooled population-level relationship.

### Why?

The project is based on personalized monitoring.

The analyses showed that behavioral–well-being relationships can differ substantially between individuals.

This heterogeneity is itself relevant to the research question.

### Alternative Considered

Combine all participants into one population-level correlation.

### Why Was It Not Selected as the Sole Analysis?

A pooled relationship could hide meaningful individual differences and would be less consistent with the personalized-baseline approach.

### Outcome

Participant-level relationships are retained and interpreted individually.

---

## 16. Interpreting Repeated Behavioral Patterns

### Decision

Emphasize behavioral domains that show repeated significant associations across participants when summarizing the findings.

### Why?

The goal was not to assume in advance that one specific behavioral variable was the most important.

Instead, the analysis was used to identify which behavioral and sleep-related domains repeatedly showed associations with well-being.

Sleep-related and physical-activity measures appeared most repeatedly among the significant relationships.

### Alternative Considered

Pre-select sleep or physical activity as the main variables before analyzing the results.

### Why Was It Not Selected?

Pre-selecting specific domains before examining the results could introduce an unnecessary assumption.

### Outcome

Sleep and physical activity were emphasized based on the observed results, while individual differences were retained.

---

## 17. One-Day Lagged Analysis for Early Detection

### Decision

Add a one-day lagged analysis in which behavioral deviation on day **t** is compared with well-being on day **t+1**.

### Why?

The project is not only interested in whether behavior and well-being are related. It also aims to investigate the possibility of **early detection**.

A same-day relationship does not show whether a behavioral deviation occurs before a later change in well-being.

The lagged analysis introduces temporal ordering:

**Behavioral deviation → following-day well-being**

This makes the analysis more directly relevant to the early-warning objective.

### Alternative Considered

Use only same-day relationships.

### Why Was This Not Sufficient?

Same-day associations cannot establish whether the behavioral change occurred before the well-being measurement.

### Outcome

A one-day lagged analysis was performed across the participant-level behavioral and wellness variables.

---

## 18. Choice of One-Day Lag

### Decision

Use a one-day lag as the initial temporal window.

### Why?

The PMData observations are organized at a daily level, making a one-day window a natural and interpretable temporal unit.

In addition, previous research in community and healthy adult populations has examined prospective relationships between daily behavioral factors such as sleep or physical activity and next-day affect or well-being.

Therefore, a one-day lag provided a theoretically and practically justified first temporal window rather than introducing an arbitrary longer prediction horizon.

### Alternative Considered

Use multiple-day lags without prior justification.

### Why Was This Not Selected?

A longer lag would introduce an additional methodological assumption without a clear justification from the current research design.

### Outcome

The temporal analysis uses:

**Behavior on day t → well-being on day t+1**

---

## 19. Lagged FDR Analysis

### Decision

Apply the same minimum-N criterion and FDR correction to the lagged analysis.

### Why?

The lagged analysis also involves many participant × behavioral × wellness tests.

Therefore, the same approach to incomplete data, minimum sample size, and multiple testing was retained to make the temporal analysis comparable with the same-day analysis.

### Outcome

The lagged analysis produced:

* 1,040 total relationship tests,
* 871 relationships with N ≥ 10,
* 64 raw significant relationships,
* **4 FDR-significant relationships**.

All four FDR-significant relationships were observed for participant **p01** and next-day **readiness**.

---

## 20. Interpretation of Lagged Findings

### Decision

Interpret the lagged significant relationships as preliminary evidence of potential next-day early-warning associations.

### Why?

The behavioral deviation was measured one day before the well-being measurement, providing temporal ordering.

However, the analysis remains correlational and participant-specific.

### Outcome

The four significant lagged relationships are treated as evidence supporting the **potential** of personalized behavioral monitoring as an early-warning approach.

They are not interpreted as proof of a validated prediction system.

---

## 21. Early Detection vs. Prediction

### Decision

Do not claim that the current analysis constitutes a fully validated predictive AI model.

### Why?

The current work demonstrates:

* personalized baseline learning,
* behavioral deviation detection,
* statistical association analysis,
* and temporal/lagged relationships.

However, it does not include a trained predictive model evaluated on unseen future outcomes using formal predictive performance metrics.

### Outcome

The project's current claim is limited to:

> **Personalized behavioral deviations show associations with well-being and provide preliminary evidence for potential early-warning signals.**

The project does not claim clinical prediction or diagnosis.

---

## 22. Handling Individual Heterogeneity

### Decision

Treat participant heterogeneity as an important finding rather than as noise to be removed.

### Why?

The project's central premise is that behavioral patterns and well-being can differ between individuals.

If different participants show different behavioral–well-being relationships, this supports the need for personalized monitoring.

### Outcome

The results are not interpreted as evidence that one behavioral variable has the same relationship with well-being for everyone.

---

## 23. Final Methodological Framework

### Decision

Use the following framework for the current project:

**Personal data → Personalized baseline → Daily behavioral deviation → Change detection → Same-day association analysis → FDR correction → One-day lagged analysis → FDR correction → Individual-level interpretation**

### Why?

This sequence follows the research question step by step:

1. identify what is normal for each individual,
2. detect departures from that normal pattern,
3. examine whether behavioral deviations are associated with well-being,
4. test same-day relationships,
5. and finally investigate whether behavioral deviations precede next-day well-being changes.

### Outcome

The framework provides a reproducible personalized early-warning analysis while avoiding unsupported claims of causality or validated prediction.

---

## 24. Overall Research Conclusion

The current analysis supports the following methodological conclusion:

> **A personalized baseline can be used to identify deviations in an individual's daily behavioral patterns. These personalized deviations show associations with self-reported well-being, with sleep and physical activity emerging as the most repeatedly associated behavioral domains. The one-day lagged analysis identified a smaller set of next-day relationships, providing preliminary evidence that personalized behavioral deviations may have potential as early-warning signals.**

At the same time, the findings demonstrate substantial individual heterogeneity and do not establish causality or validated predictive performance.

---

## 25. Limitations

The following limitations are explicitly retained:

* The dataset contains 16 participants.
* Some participants had incomplete observations across variables and days.
* The number of usable observations differed between relationships.
* The analysis is observational.
* Correlation does not establish causality.
* A one-day lag provides temporal ordering but is not equivalent to predictive modeling.
* The anomaly-only analysis had limited statistical power.
* Significant relationships varied across participants.
* Results may depend on the selected baseline period and deviation threshold.
* The current framework does not constitute a clinical diagnostic system.

---

## 26. Project Status

### Completed

* Dataset selection and preparation
* Personalized baseline construction
* Chronological baseline/analysis split
* Personalized behavioral deviation calculation
* Z-score standardization
* Primary change detection
* Sensitivity analysis
* Handling of incomplete data
* Minimum-N filtering
* Same-day personalized deviation analysis
* Pearson correlation analysis
* Benjamini–Hochberg FDR correction
* One-day lagged analysis
* Lagged FDR correction
* Participant-level interpretation
* Reproducible Git-based project structure

### Documentation to Complete

* Finalize the research report
* Document methods, formulas, and implementation details in the report
* Consolidate final findings, limitations, and implications
* Finalize the project README

