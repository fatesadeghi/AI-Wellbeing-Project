# Decision Log

## Project

**AI-Based Early Detection of Changes in Well-Being and Quality of Life Among People Living Alone**

## Research Question

**Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?**

---

## 1. Dataset Selection

**Decision:** Use PMData as the primary dataset.

**Alternative considered:** CASAS Smart Home dataset.

**Reason:**
CASAS provides useful behavioral and sensor data, but it does not provide sufficient direct self-reported well-being measures for the intended analysis. PMData contains both objective behavioral/sleep measures and self-reported wellness variables.

**Outcome:**
PMData was selected because it allows behavioral changes to be examined in relation to self-reported well-being.

---

## 2. Personalized Baseline

**Decision:** Build a separate behavioral baseline for each participant rather than using a universal population-level baseline.

**Alternative considered:** Use the same threshold or average pattern for all participants.

**Reason:**
Individuals have different normal levels of activity, sleep, and other behaviors. A behavior that is unusual for one person may be normal for another.

**Outcome:**
The analysis uses each participant's own historical behavior to define their normal pattern.

---

## 3. Chronological Train/Baseline Split

**Decision:** Use the first 50% of each participant's data as the personalized baseline period and the second 50% as the analysis period.

**Alternative considered:** Randomly split observations into training and analysis sets.

**Reason:**
The project concerns behavioral change and potential early detection over time. A chronological split better preserves the temporal structure and avoids using future observations to define the earlier baseline.

**Outcome:**
The first half is used to establish the personal baseline; the second half is used to evaluate deviations and their relationship with well-being.

---

## 4. Deviation Measurement

**Decision:** Measure behavioral deviation relative to the participant's personalized baseline and standardize the deviation using a Z-score.

**Reason:**
Raw differences are not directly comparable across variables with different scales. Z-scores express how far an observation is from the individual's typical level relative to their baseline variability.

**Outcome:**
Daily behavioral deviations are represented using personalized Z-scores.

---

## 5. Primary Change-Detection Threshold

**Decision:** Use `|Z| ≥ 2` as the primary threshold for detecting meaningful behavioral deviations.

**Alternative considered:** Lower thresholds such as `|Z| ≥ 1`.

**Reason:**
A threshold of two standard deviations provides a more conservative definition of an unusual observation and reduces the number of minor fluctuations classified as meaningful changes.

**Outcome:**
Using `|Z| ≥ 2`, 558 meaningful deviations were detected across 16 participants.

---

## 6. Sensitivity Analysis

**Decision:** Also evaluate a less conservative threshold of `|Z| ≥ 1`.

**Reason:**
The choice of deviation threshold can affect the number of detected changes. A sensitivity analysis helps determine whether the findings depend strongly on the primary threshold.

**Outcome:**
Using `|Z| ≥ 1`, 2,743 deviations were detected. The `|Z| ≥ 2` threshold was retained as the primary analysis, while `|Z| ≥ 1` was treated as a sensitivity analysis.

---

## 7. Minimum Sample Size

**Decision:** Require at least 10 paired observations (`N ≥ 10`) for a participant-level correlation analysis.

**Reason:**
Very small samples can produce unstable and difficult-to-interpret correlation estimates.

**Outcome:**
Relationships with fewer than 10 usable observations were excluded from inferential correlation analysis.

---

## 8. Initial Anomaly-Only Analysis

**Decision:** Initially examine the relationship between detected behavioral anomalies and well-being only on days classified as meaningful deviations.

**Reason:**
This directly tests whether unusually large behavioral changes coincide with changes in well-being.

**Outcome:**
The anomaly-only analysis produced very limited statistical power: among 795 relationship tests, only 17 met the minimum N requirement, with 1 raw significant relationship and 0 FDR-significant relationships.

**Interpretation:**
This result was not interpreted as evidence that behavioral deviations are unrelated to well-being. The anomaly-only filter substantially reduced the available sample size.

---

## 9. Full-Period Personalized Deviation Analysis

**Decision:** Extend the primary relationship analysis to the full analysis period rather than restricting it only to detected anomaly days.

**Reason:**
Using all valid daily personalized deviations preserves substantially more observations and allows the analysis to examine whether within-person behavioral deviations are associated with well-being across the analysis period.

**Outcome:**
The full-period analysis provided substantially greater statistical power and identified multiple significant relationships after FDR correction.

---

## 10. Same-Day Behavioral Deviation and Well-Being

**Decision:** First examine same-day relationships between personalized behavioral deviations and self-reported well-being.

**Reason:**
Same-day analysis provides an initial test of whether departures from a person's normal behavioral pattern are associated with their well-being on the same day.

**Outcome:**
The final daily personalized-deviation analysis identified 44 FDR-significant participant-level relationships.

**Interpretation:**
Sleep and physical activity showed the most repeated associations with well-being, although relationships varied substantially between participants.

---

## 11. Multiple-Comparison Correction

**Decision:** Use Benjamini-Hochberg False Discovery Rate (FDR) correction across the relationship tests.

**Reason:**
The analysis examines many behavioral-variable × wellness-variable × participant relationships. Testing many relationships increases the probability of false-positive findings.

**Outcome:**
FDR-adjusted results were used as the primary basis for identifying statistically significant relationships.

---

## 12. Interpretation of Correlations

**Decision:** Interpret significant correlations as associations rather than causal effects or definitive predictions.

**Reason:**
The correlation-based analysis is observational and does not establish causality. Other factors may contribute to the observed relationships.

**Outcome:**
Results are described as behavioral–well-being associations and, where appropriate, as preliminary evidence for potential early-warning signals rather than causal or predictive effects.

---

## 13. One-Day Lagged Analysis

**Decision:** Add a one-day lagged analysis in which behavioral deviation on day `t` is compared with self-reported well-being on day `t+1`.

**Reason:**
Same-day association alone does not demonstrate that behavioral changes could provide an early signal of subsequent well-being changes. A one-day lag provides a temporal ordering in which the behavioral deviation occurs before the well-being measurement.

The one-day lag was chosen based on prior research in community and healthy adult samples reporting prospective relationships between sleep/physical activity and next-day affect or well-being.

**Outcome:**
The lagged analysis was completed across 1,040 possible relationships. 871 met the minimum N requirement. 64 were significant before multiple-comparison correction, but only 4 remained significant after FDR correction.

---

## 14. Final Lagged Findings

**Decision:** Treat the FDR-significant lagged relationships as preliminary evidence of potential next-day early-warning associations.

**Outcome:**
Four FDR-significant relationships were identified, all for participant p01 and next-day readiness:

* Steps → next-day readiness: `r = -0.551`, `q = 0.001146`
* Exercise_Calories → next-day readiness: `r = -0.549`, `q = 0.001146`
* Exercise_Distance → next-day readiness: `r = -0.491`, `q = 0.011039`
* Exercise_Duration → next-day readiness: `r = -0.454`, `q = 0.035658`

**Interpretation:**
For p01, deviations in physical-activity measures were associated with readiness on the following day. Because these relationships were observed in only one participant, they should not be generalized to all participants.

---

## 15. Early Detection Claim

**Decision:** Frame the current result as evidence of potential early-warning capability rather than a completed predictive AI system.

**Reason:**
The current analysis demonstrates personalized baseline construction, deviation detection, and temporal associations with subsequent well-being. However, it does not yet constitute a trained predictive model evaluated on future unseen outcomes.

**Outcome:**
The project can support the claim that personalized behavioral monitoring has potential for early detection/early warning, while explicitly distinguishing this from validated prediction or diagnosis.

---

## 16. Individual-Level Heterogeneity

**Decision:** Preserve participant-level results rather than relying only on pooled relationships.

**Reason:**
The project is based on personalized well-being and behavioral patterns. The analyses show that relationships differ between individuals.

**Outcome:**
The findings support the importance of individualized baselines and participant-specific interpretation rather than assuming one universal behavioral–well-being relationship.

---

## 17. Main Findings So Far

**Decision:** Emphasize repeated patterns across participants while retaining individual-level differences.

**Outcome:**
Sleep-related and physical-activity variables produced the most repeated significant associations with well-being. Same-day analysis produced substantially more FDR-significant relationships than the one-day lagged analysis, while the lagged analysis identified a smaller set of potentially meaningful next-day relationships.

---

## 18. Limitations to Preserve in Interpretation

The following limitations are retained as part of the interpretation of the results:

* The analysis is observational and does not establish causality.
* Correlation does not demonstrate that behavioral deviation causes a change in well-being.
* The lagged analysis provides temporal ordering but is not equivalent to predictive modeling.
* The number of participants is limited to 16.
* Significant relationships are heterogeneous across participants.
* The anomaly-only analysis had limited statistical power because filtering to anomaly days substantially reduced the number of observations.
* The current work does not constitute a clinical diagnostic system.

---

## 19. Current Methodological Position

The final methodological framework is:

**Personal data → Personalized baseline → Daily behavioral deviation → Change detection → Same-day behavioral/well-being association → One-day lagged analysis → FDR correction → Individual-level interpretation**

The framework is intended as a personalized early-warning approach rather than a universal threshold-based or diagnostic system.

---

## 20. Current Project Status

### Completed

* Dataset selection and preparation
* Personalized baseline construction
* Daily deviation calculation
* Z-score standardization
* Primary change detection
* Sensitivity analysis
* Same-day personalized deviation analysis
* FDR correction
* One-day lagged analysis
* Lagged FDR correction
* Participant-level interpretation
* Git repository and analysis outputs

### Remaining

* Finalize and commit this decision log
* Consolidate the final research findings and limitations
* Prepare the final project report/walkthrough materials

