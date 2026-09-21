````markdown
# Research Decision Log

## Project
**AI-Based Early Detection of Changes in Well-Being and Quality of Life Among People Living Alone**

## Central Research Question

> Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?

This decision log documents the main methodological choices made during the project, why they were made, and how they were implemented.

The purpose of the log is to support transparency, reproducibility, and understanding of the research process.

---

# 1. Dataset Selection

### Decision
Use the **PMData** dataset as the main dataset for the final analysis.

### Reason
The project requires both objective behavioral measurements and well-being-related information.

The initial exploration considered the CASAS smart-home dataset. However, it did not provide sufficient direct self-reported well-being information for the intended analysis.

PMData provided a more suitable combination of:

- behavioral and activity measurements,
- sleep-related measurements,
- and self-reported well-being variables.

### Outcome
PMData became the final dataset used for the statistical and machine-learning analyses.

---

# 2. Behavioral Variable Selection

### Decision
Retain the full set of **13 behavioral variables** rather than reducing the analysis to a smaller subset.

### Behavioral variables

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

### Reason
Well-being is multidimensional, and changes in different aspects of daily behavior may provide different information.

Reducing the behavioral set before analysis could remove potentially informative patterns.

### Outcome
All 13 variables were retained as candidate behavioral predictors.

---

# 3. Personalized Baseline

### Decision
Construct a separate behavioral baseline for each participant.

### Reason
The project is based on the idea that the same behavior may have different meanings for different people.

For example, a particular number of daily steps may be normal for one participant but unusual for another.

Therefore, a population-wide threshold was not considered sufficient for detecting individual changes.

### Implementation
For each participant, the first half of their available chronological observations was used to establish their personal baseline.

Baseline statistics included:

- mean
- standard deviation

These statistics were calculated separately for each behavioral variable.

### Outcome
Each participant received an individual behavioral baseline.

---

# 4. Chronological 50/50 Baseline Split

### Decision
Use a chronological **50/50 split** for baseline construction.

### Reason
The baseline should represent an individual's earlier normal behavior rather than information from the future.

A chronological split also preserves the temporal structure of the data.

### Implementation

For each participant:

- the earlier 50% of observations formed the baseline period;
- the later 50% formed the analysis period.

### Outcome
The baseline was established before the main deviation analysis.

---

# 5. Personalized Deviation Measures

### Decision
Measure behavioral deviation using a participant-specific standardized score.

### Formula

\[
Z = \frac{X-\mu_{baseline}}{\sigma_{baseline}}
\]

where:

- \(X\) = observed value,
- \(\mu_{baseline}\) = participant's baseline mean,
- \(\sigma_{baseline}\) = participant's baseline standard deviation.

### Interpretation

A positive Z-score indicates that the observed value is above the participant's usual level.

A negative Z-score indicates that it is below the participant's usual level.

The absolute value indicates how unusual the observation is relative to the participant's baseline variability.

### Important interpretation

The deviation measure identifies **unusual behavior**, not whether that behavior is inherently good or bad.

For example:

- unusually low activity may be relevant;
- unusually high activity may also be relevant;
- unusually long sleep may be relevant;
- unusually short sleep may also be relevant.

The method does not determine the health meaning of the deviation by itself.

---

# 6. Primary Behavioral Deviation Threshold

### Decision
Use **|Z| ≥ 2** as the primary deviation threshold.

### Reason
A threshold of 2 identifies relatively large deviations from an individual's baseline.

This provides a more conservative definition of unusual behavior and reduces the number of smaller fluctuations classified as deviations.

### Interpretation

- \(|Z| < 2\): not classified as a primary deviation
- \(|Z| \geq 2\): classified as a primary deviation

Both unusually high and unusually low values are included.

### Outcome
The |Z| ≥ 2 threshold was used as the primary deviation criterion.

---

# 7. Sensitivity Analysis with |Z| ≥ 1

### Decision
Perform a sensitivity analysis using **|Z| ≥ 1**.

### Reason
A threshold of 1 is more sensitive than a threshold of 2.

It captures smaller deviations that may be missed by the primary threshold.

The purpose was to determine whether the general findings were substantially affected by the choice of deviation threshold.

### Outcome
The |Z| ≥ 1 analysis was retained as a sensitivity analysis rather than replacing the primary |Z| ≥ 2 criterion.

---

# 8. Handling Incomplete and Unequal Data Availability

### Decision
Analyze each participant based on the observations that are actually available, while explicitly tracking missing data.

### Reason
Participants did not necessarily have identical amounts of data for every variable.

Removing all participants with any missing information would unnecessarily reduce the usable sample.

### Implementation
Data availability was assessed at the participant and feature level.

For the ML pipeline, feature availability was explicitly classified, including:

- Available
- Available_With_Missing
- Available_With_Long_Missing_Streak
- Available_Sleep_Exempt
- Missing_Column
- Excluded_First_10_Days_Missing

### Outcome
Missingness was documented rather than silently ignored.

The feature-availability analysis identified:

- 115 `Available_With_Missing`
- 69 `Available`
- 14 `Available_Sleep_Exempt`
- 9 `Available_With_Long_Missing_Streak`
- 1 `Missing_Column`
- 0 `Excluded_First_10_Days_Missing`

---

# 9. Minimum Sample Size N ≥ 10

### Decision
Require at least **10 usable paired observations** for a participant-level behavior × well-being relationship.

### Reason
Correlation estimates based on very few observations can be unstable and difficult to interpret.

A minimum of 10 observations was therefore used as a practical inclusion criterion.

### Implementation
For each participant × behavioral variable × well-being variable relationship:

\[
N \geq 10
\]

was required for the relationship to be included in the statistical analysis.

### Outcome
Relationships with fewer than 10 usable paired observations were not treated as valid statistical relationships.

---

# 10. Pearson Correlation

### Decision
Use **Pearson correlation** to examine linear relationships between behavioral measures and well-being variables.

### Reason
The primary statistical question was whether changes in behavioral variables were associated with changes in well-being-related measures.

Pearson's \(r\) provides:

- direction of association,
- strength of linear association,
- and a corresponding statistical significance test.

### Interpretation

\[
-1 \leq r \leq 1
\]

- positive \(r\): higher values of one variable are associated with higher values of the other;
- negative \(r\): higher values of one variable are associated with lower values of the other;
- values closer to zero indicate weaker linear association.

### Important limitation
Correlation does not establish causation.

---

# 11. Initial Anomaly-Only Analysis

### Decision
Begin the deviation-based analysis by examining unusual behavioral observations.

### Reason
The central research question is concerned with whether changes in daily behavior can provide signals of changes in well-being.

Therefore, identifying departures from an individual's normal behavior was an important first analytical step.

### Outcome
Personalized deviations were generated from the baseline and used as the basis for subsequent relationship analyses.

---

# 12. Full-Period Personalized Deviation Analysis

### Decision
Use the complete analysis period after baseline construction rather than restricting the analysis to a small number of manually selected anomaly days.

### Reason
The research question concerns whether behavioral changes over time are related to well-being.

Using the complete analysis period allows the statistical analysis to evaluate the broader relationship rather than only a small subset of observations.

### Outcome
Daily personalized deviations were retained throughout the analysis period.

---

# 13. Same-Day Analysis

### Decision
Perform a same-day analysis between behavioral deviations and well-being measurements from the same day.

### Reason
A same-day analysis tests whether unusual behavior and well-being measurements are associated within the same temporal window.

### Analysis

Behavior at day \(t\)

\[
\rightarrow
\]

Well-being at day \(t\)

### Outcome

The final same-day analysis contained:

- 1,040 tested relationships
- 900 relationships with \(N \geq 10\)
- 180 raw significant relationships
- 43 FDR-significant relationships

---

# 14. Benjamini–Hochberg FDR Correction

### Decision
Apply the **Benjamini–Hochberg False Discovery Rate (FDR)** correction to the multiple statistical tests.

### Reason
The analysis evaluates many participant × behavior × well-being relationships.

When many statistical tests are performed, some small p-values can occur by chance.

Using uncorrected p-values alone would increase the risk of false-positive findings.

### Implementation
The Benjamini–Hochberg procedure was applied separately to each analysis family.

The adjusted significance criterion was:

\[
q < 0.05
\]

### Interpretation

- raw \(p < 0.05\): statistically significant before multiple-testing correction;
- FDR \(q < 0.05\): statistically significant after multiple-testing correction.

### Outcome
FDR correction was used for the final interpretation of statistical significance.

---

# 15. One-Day Lagged Analysis

### Decision
Perform a one-day lagged analysis.

### Reason
A behavioral change may occur before a change in self-reported well-being.

Therefore, same-day associations alone may not capture temporal relationships.

### Analysis

Behavior at day \(t\)

\[
\rightarrow
\]

Well-being at day \(t+1\)

### Outcome

The final lagged analysis contained:

- 1,040 tested relationships
- 860 relationships with \(N \geq 10\)
- 63 raw significant relationships
- 4 FDR-significant relationships

### Interpretation
This analysis examines temporal ordering but does not by itself establish causality.

---

# 16. Seven-Day Historical Analysis

### Decision
Analyze the relationship between recent behavioral history and same-day well-being.

### Reason
A person's well-being may reflect accumulated behavioral patterns rather than only the behavior observed on the same day.

The seven-day analysis therefore examined whether behavior during the previous seven days was associated with well-being on the current day.

### Analysis

Previous seven days of behavior

\[
\rightarrow
\]

Current-day well-being

### Outcome

The final seven-day analysis contained:

- 1,040 tested relationships
- 794 relationships with \(N \geq 10\)
- 103 raw significant relationships
- 6 FDR-significant relationships

---

# 17. Final Statistical Results

### Decision
Use the final FDR-corrected results from the four completed analysis families as the statistical results of the project.

### Final summary

| Analysis | Total Tests | Valid N ≥ 10 | Raw Significant | FDR Significant |
|---|---:|---:|---:|---:|
| Same-day | 1,040 | 900 | 180 | 43 |
| One-day lagged | 1,040 | 860 | 63 | 4 |
| Seven-day history | 1,040 | 794 | 103 | 6 |
| Sensitivity, \|Z\| ≥ 1 | 1,040 | 887 | 182 | 47 |
| **Total** | **4,160** | **3,443** | **528** | **100** |

### Interpretation
The analyses identified statistically significant relationships after FDR correction.

However, these findings should be interpreted as associations between behavioral patterns and well-being-related measures.

They do not demonstrate that a behavioral change causes a change in well-being.

---

# 18. Machine-Learning Strategy / System

### Decision
Develop a personalized machine-learning system to examine whether daily behavioral information can model **next-day changes in well-being**.

### Reason
The statistical analyses evaluate relationships between variables.

The research question also asks whether AI can learn individual patterns and detect or model changes in well-being.

A machine-learning component was therefore added to evaluate predictive modeling at the individual level.

### System design

The ML pipeline:

1. prepares participant-level daily data;
2. evaluates feature availability;
3. creates behavioral-derived features;
4. predicts next-day well-being changes;
5. trains participant-specific models;
6. evaluates model performance;
7. analyzes feature importance.

### Feature space
The system retained the 13 behavioral variables as the underlying behavioral feature set.

Derived features resulted in **91 candidate feature columns**.

### Outcome
The personalized ML system was implemented and executed on the PMData participants.

---

# 19. ML Feature Availability and Missing Data

### Decision
Evaluate feature availability separately for each participant before interpreting ML results.

### Reason
Real-world longitudinal behavioral data can contain:

- missing days,
- incomplete variables,
- long missing periods,
- and participant-specific differences in data coverage.

A single global assumption about data availability would therefore be inappropriate.

### Implementation

The feature-availability analysis examined:

- whether a variable exists;
- whether observations are missing;
- whether missingness occurs in the initial ten calendar days;
- whether long missing streaks occur;
- whether sleep variables require different treatment.

### Outcome
The analysis produced a dedicated feature-availability table and summary.

No feature was excluded because all observations in the first ten calendar days were missing.

One original behavioral variable was identified as having a missing column for one participant.

---

# 20. Personalized ML Models

### Decision
Train separate models for each participant and each well-being target rather than training one universal model for all participants.

### Reason
The central research framework is personalized.

Different participants can have different:

- normal activity levels,
- sleep patterns,
- variability,
- behavioral relationships,
- and responses over time.

A single population-level model could therefore obscure individual patterns.

### Targets

Five well-being-related targets were modeled:

1. fatigue
2. mood
3. readiness
4. sleep_quality
5. stress

### Model count

With:

- 16 participants
- 5 targets

the pipeline produced:

\[
16 \times 5 = 80
\]

participant-specific models.

### Outcome
80 personalized models were trained and evaluated.

---

# 21. ML Model Evaluation

### Decision
Evaluate the personalized models using:

- \(R^2\)
- MAE
- RMSE

### Reason
A single performance measure does not fully describe predictive performance.

The three metrics provide complementary information.

### Results

Across the 80 models:

- Positive \(R^2\): 14
- Zero \(R^2\): 1
- Negative \(R^2\): 65
- Mean \(R^2\): -0.1147
- Median \(R^2\): -0.0903
- Mean MAE: 0.6871
- Mean RMSE: 0.9236

### Interpretation
The results show that predictive performance varied substantially between participant-specific models.

Most models did not achieve positive \(R^2\).

Therefore, the current dataset and modeling setup do not provide evidence of strong, general predictive performance across participants.

The ML system should be viewed as an implemented and evaluated research model rather than a validated real-world prediction system.

---

# 22. Feature Importance

### Decision
Analyze feature importance to identify which behavioral patterns contributed most frequently to participant-specific ML models.

### Reason
Prediction performance alone does not explain which aspects of daily behavior the models use.

Feature-importance analysis provides an additional view of which behavioral variables may contain useful information.

### Implementation

The ML pipeline produced:

- 7,280 feature-importance rows;
- 91 derived feature columns;
- mapping of derived features back to the original 13 behavioral variables.

All 91 derived features were successfully mapped to the original behavioral variables.

### Outcome
Feature importance was aggregated at the behavioral-variable level and examined:

- overall,
- by participant,
- by well-being target,
- and among the top three behaviors for each participant × target combination.

---

# 23. Interpretation of ML Results

### Decision
Interpret feature-importance results as indicators of model usage rather than causal importance.

### Top-3 behavior frequency

| Behavioral variable | Top-3 appearances |
|---|---:|
| Steps | 49 |
| Exercise_Calories | 30 |
| Sleep_Restlessness | 28 |
| Deep_Sleep_Minutes | 24 |
| Sleep_Hours | 24 |
| Exercise_Duration | 23 |
| Sleep_Score | 20 |
| Sleep_Duration_Score | 14 |
| Exercise_Avg_HR | 13 |
| Exercise_Distance | 6 |
| Sleep_Revitalization | 6 |
| Sleep_Composition | 2 |
| Exercise_Count | 1 |

### Interpretation
Steps appeared most frequently among the top three behavioral variables across participant × well-being models.

Other variables, including exercise and sleep measures, also appeared frequently.

However, feature importance does not demonstrate that a variable causes changes in well-being.

It indicates that the variable contributed to the model's predictions under the implemented modeling procedure.

---

# 24. Relationship Between Statistical and ML Analyses

### Decision
Treat the statistical and ML analyses as complementary rather than interchangeable.

### Reason

The statistical analysis asks:

> Are behavioral measures associated with well-being-related measures?

The ML analysis asks:

> Can participant-specific behavioral information be used to model changes in well-being?

These are related but different questions.

A variable can show a statistical association without providing strong predictive performance.

Similarly, a feature can contribute to a model without establishing a statistically significant relationship in the correlation analysis.

### Outcome
The final project uses both approaches to provide complementary evidence.

The statistical analysis provides association-based evidence, while the ML analysis evaluates individual predictive modeling.

---

# 25. Reproducibility

### Decision
Maintain the complete analysis pipeline in the GitHub repository.

### Repository structure

```text
Code/
├── 01_build_baseline.py
├── 02_same_day_analysis.py
├── 03_lagged_analysis.py
├── 04_seven_day_analysis.py
├── 05_sensitivity_analysis.py
├── 06_fdr_correction.py
└── 07_final_summary.py

ML/
└── Code/
    ├── 08_prepare_ml_data.py
    ├── 09_feature_availability.py
    ├── 10_personalized_ml.py
    └── 11_analyze_ml_results.py
````

### Reason

The project should be reproducible and auditable.

The repository therefore contains:

* data-processing code;
* statistical analysis code;
* ML code;
* generated results;
* decision documentation;
* and version-controlled commit history.

### Outcome

The final analysis pipeline was committed and pushed to GitHub.

---

# 26. AI Use and Research Transparency

### Decision

Use AI assistance for coding support, wording, organization, and methodological discussion while retaining human responsibility for the research decisions and interpretation.

### Reason

AI tools were used during the project as a research and coding assistant.

However, AI-generated suggestions were not treated as independent scientific evidence.

### AI-supported activities included

* code development and debugging;
* restructuring analysis scripts;
* explaining statistical concepts;
* improving documentation;
* improving English wording;
* organizing research reports;
* discussing methodological alternatives.

### Research responsibility

The final methodological choices, interpretation of results, and research conclusions remain the responsibility of the researcher.

The project also follows the supervisor's requirement to document AI use and maintain understanding of the implemented analyses.

---

# 27. Ethical Interpretation / Limitations

### Decision

Interpret behavioral deviations as **potential signals**, not diagnoses or proof of health problems.

### Reason

An unusual behavioral pattern can have many possible explanations.

For example:

* unusually low activity could reflect illness, fatigue, schedule changes, or other circumstances;
* unusually high activity could also reflect different personal or situational factors;
* unusual sleep duration may have multiple possible explanations.

The current analysis cannot determine the underlying cause.

### Limitations

Important limitations include:

* small participant sample (\(N=16\));
* unequal data availability;
* missing observations;
* participant-specific models with limited observations;
* observational rather than experimental data;
* correlation does not establish causality;
* current ML performance was generally weak;
* no clinical validation;
* no external validation cohort;
* well-being was represented by five self-reported indicators rather than a complete multidimensional measurement of well-being.

### Ethical position

The system should therefore be described as an early-warning or research framework, not as a diagnostic system.

A detected deviation should be considered a signal for further investigation rather than evidence of a health condition.

---

# 28. Final Methodological Position

### Decision

The final project combines personalized behavioral baselines, deviation detection, temporal statistical analysis, multiple-testing correction, and personalized machine learning.

### Final methodological pipeline

```text
PMData
   ↓
Participant-level data preparation
   ↓
13 behavioral variables
   ↓
Personalized chronological baseline
   ↓
Daily personalized deviations
   ↓
Primary threshold |Z| ≥ 2
   ↓
Sensitivity threshold |Z| ≥ 1
   ↓
Statistical analyses
   ├── Same-day
   ├── One-day lagged
   └── Seven-day historical
   ↓
Pearson correlations
   ↓
Benjamini–Hochberg FDR correction
   ↓
Final statistical results
   ↓
Personalized ML preparation
   ↓
91 derived candidate features
   ↓
16 participants × 5 well-being targets
   ↓
80 personalized ML models
   ↓
Performance evaluation
   ↓
Feature-importance analysis
   ↓
Interpretation of behavioral signals
```

### Final research position

The project provides a framework for studying whether changes in an individual's daily behavioral patterns can be associated with changes in well-being.

The statistical analyses identified FDR-significant relationships across same-day, lagged, seven-day, and sensitivity analyses.

The machine-learning system was successfully implemented and evaluated at the participant level. However, the overall predictive performance was limited, with most models producing non-positive \(R^2\) values.

Therefore, the current results support the **feasibility of investigating personalized behavioral signals of well-being**, but they do not establish a clinically validated prediction system.

The central concept remains:

> **Learn the individual's normal behavior → detect unusual deviations → examine their relationship with well-being → evaluate whether these patterns can support personalized early detection.**

Future work should focus on larger datasets, stronger temporal validation, improved predictive modeling, external validation, and integration of additional well-being information before considering real-world deployment.

