# Decision Log

## Project

**AI-Based Early Detection of Changes in Well-Being and Quality of Life Among People Living Alone**

---

## 1. Dataset Selection

**Decision:** Use PMData as the main dataset.

**Alternative considered:** CASAS Smart Home.

**Reason:** CASAS provides rich sensor and behavioral data, but it does not provide enough direct self-reported well-being measures for the main research question. PMData contains both objective behavioral/sleep measures and self-reported wellness variables.

**Outcome:** PMData was selected. The dataset contains 16 participants with daily behavioral and wellness measurements.

---

## 2. Personalized Baseline

**Decision:** Use an individual-level personalized baseline rather than one universal baseline for all participants.

**Alternative considered:** A population-level threshold or universal baseline.

**Reason:** Normal behavior differs substantially between individuals. A universal threshold could incorrectly classify normal individual behavior as abnormal.

**Outcome:** A separate baseline was calculated for each participant using the first 50% of their chronological data.

---

## 3. Chronological Train/Baseline Split

**Decision:** Use the first 50% of each participant's timeline to establish the baseline and the later 50% for analysis.

**Alternative considered:** Random train/test splitting.

**Reason:** The project focuses on detecting changes from an individual's previous normal behavior. A chronological split preserves the temporal direction and avoids using future observations to define the baseline.

**Outcome:** The first half was used for personalized baseline estimation and the second half for deviation and well-being analysis.

---

## 4. Deviation Measurement

**Decision:** Standardize daily behavioral measurements using participant-specific Z-scores relative to the personalized baseline.

**Reason:** Raw differences are not directly comparable across variables with different scales. Z-scores express how unusual a daily observation is relative to that participant's normal behavior.

**Outcome:** Daily personalized deviations were calculated for behavioral and sleep variables.

---

## 5. Change Detection Threshold

**Decision:** Use |Z| ≥ 2 as the primary threshold for meaningful behavioral deviation.

**Alternative considered:** |Z| ≥ 1.

**Reason:** |Z| ≥ 2 provides a more conservative definition of an unusual observation and reduces the number of minor fluctuations classified as meaningful changes.

**Outcome:** The |Z| ≥ 2 threshold was used as the primary change-detection criterion. |Z| ≥ 1 was evaluated as a sensitivity analysis.

---

## 6. Minimum Sample Size

**Decision:** Require at least N = 10 observations for a participant-level correlation.

**Reason:** Very small sample sizes can produce unstable correlation estimates and unreliable significance tests.

**Outcome:** Relationships with fewer than 10 valid observations were excluded from inferential analysis.

---

## 7. Initial Anomaly-Only Analysis

**Decision:** Initially test the relationship between detected anomaly days and well-being.

**Outcome:** This approach produced very small sample sizes for many participant-level relationships, limiting statistical power.

**Revision:** Instead of restricting the correlation analysis only to anomaly days, the analysis was expanded to the full analysis period while using personalized behavioral deviations as the behavioral variables.

**Reason:** This preserves the personalized-deviation concept while providing enough observations to evaluate the relationship with well-being.

---

## 8. Full-Period Personalized Deviation Analysis

**Decision:** Analyze daily personalized behavioral deviation scores across the full analysis period and correlate them with same-day self-reported well-being.

**Reason:** This approach provides substantially more observations than anomaly-only analysis and allows the relationship between the degree of behavioral deviation and well-being to be examined.

**Outcome:** 1,040 behavioral × well-being relationships were evaluated; 935 had N ≥ 10.

---

## 9. Same-Day Relationship

**Decision:** Examine same-day associations between behavioral deviation and self-reported well-being.

**Alternative considered:** Lagged relationships.

**Reason:** Same-day analysis provides a direct and interpretable first test of whether behavioral deviations are associated with changes in well-being.

**Outcome:** Same-day Pearson correlations were used for the main analysis.

---

## 10. Multiple-Testing Correction

**Decision:** Apply Benjamini–Hochberg False Discovery Rate (FDR) correction.

**Reason:** The analysis includes many participant-level behavioral × well-being tests. Using raw p-values alone would increase the risk of false-positive findings.

**Outcome:** 185 relationships had p < 0.05 before correction, while 44 remained significant after FDR correction at q < 0.05.

---

## 11. Interpretation of Correlations

**Decision:** Interpret significant correlations as associations rather than causal effects.

**Reason:** The observational analysis does not establish that behavioral changes cause changes in well-being.

**Outcome:** Results are reported as evidence of associations between personalized behavioral deviations and self-reported well-being.

---

## 12. Main Finding So Far

**Outcome:** Sleep- and activity-related behavioral deviations showed the most repeated significant associations with well-being, while the strength and direction of relationships varied across individuals.

**Interpretation:** The results support the usefulness of personalized behavioral baselines and suggest that changes in sleep and physical activity may be informative signals for changes in individual well-being.
