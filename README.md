# AI-Based Early Detection of Changes in Well-Being Among People Living Alone

## Project Overview

This project investigates whether AI can detect changes in a person's well-being by learning their daily behavioral patterns and identifying unusual deviations from their individual baseline.

The central research question is:

> **Can AI detect changes in a person's well-being by learning their daily habits and identifying unusual behavioral patterns?**

The project follows a **personalized approach**. Instead of assuming that the same behavioral threshold applies to everyone, the analysis first learns each participant's usual behavioral pattern and then examines deviations from that individual baseline.

Behavioral deviations are treated as **potential signals of changes in well-being**, not as diagnoses or proof of a health condition.

The overall research framework is:

```text
Learn individual baseline
        ↓
Understand usual behavioral patterns
        ↓
Detect meaningful deviations
        ↓
Examine relationships with well-being
        ↓
Develop and evaluate personalized AI models
```

---

## Dataset

The project uses the **PMData** dataset.

PMData combines daily behavioral and sleep-related measurements collected from wearable devices with self-reported well-being indicators.

The final analysis includes **16 participants**.

### Behavioral Variables

The project retains all 13 behavioral variables as candidate variables throughout the analysis:

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

* fatigue
* mood
* readiness
* sleep_quality
* stress

These indicators are used as measures of reported well-being-related states. They do not represent the full multidimensional construct of well-being.

---

## Research Pipeline

The project follows a personalized research pipeline:

```text
Daily behavioral data
        ↓
Personalized baseline
        ↓
Daily behavioral deviations
        ↓
Same-day / lagged / 7-day analyses
        ↓
Sensitivity analysis
        ↓
FDR correction
        ↓
Personalized machine learning
        ↓
Participant-specific behavioral importance
        ↓
Personalized well-being monitoring framework
```

---

## Personalized Baseline

A central methodological decision in this project is the use of a **participant-specific baseline**.

For each participant, the available data are divided chronologically into two equal parts:

* the first 50% is used to estimate the participant's baseline;
* the remaining 50% is used as the analysis period.

For each behavioral variable, the baseline mean and standard deviation are calculated from the participant's own baseline data.

Daily behavioral deviation is then expressed using a standardized score:

```text
Z = (Daily value - Personal baseline mean)
    / Personal baseline standard deviation
```

This measures how unusual a daily behavior is relative to the individual's own usual pattern.

Both unusually high and unusually low values can be detected because the analysis uses the absolute value of the standardized deviation.

The primary deviation threshold is:

```text
|Z| >= 2
```

A sensitivity analysis using:

```text
|Z| >= 1
```

was also performed to examine the effect of using a more sensitive deviation threshold.

Importantly, a large deviation does not automatically mean that the behavior is harmful or unhealthy. It indicates that the behavior differs substantially from the individual's established baseline.

---

## Statistical Analysis

The statistical analysis examines whether personalized behavioral deviations are associated with reported well-being.

### Same-Day Analysis

The same-day analysis examines relationships between behavioral deviations and well-being measured on the same day:

```text
Behavioral deviation on day t
        ↕
Well-being on day t
```

Pearson correlations were calculated for participant-specific combinations of behavioral variables and well-being indicators.

### One-Day Lagged Analysis

The lagged analysis examines whether behavioral patterns on one day are associated with well-being on the following day:

```text
Behavioral deviation on day t
        ↓
Well-being on day t+1
```

This provides a temporal perspective without implying causality.

### Seven-Day Historical Analysis

The seven-day analysis examines whether behavioral history during the previous seven days is associated with well-being on the current day:

```text
Behavioral history during previous 7 days
        ↓
Well-being on current day
```

This analysis was included because changes in well-being may potentially relate to behavioral patterns accumulated over several days rather than a single day's behavior.

### Minimum Sample Size

A participant-specific relationship was considered valid for statistical testing when at least:

```text
N >= 10
```

usable paired observations were available.

### Pearson Correlation

Pearson correlation was used to quantify the direction and strength of linear associations between behavioral deviations and well-being indicators.

The correlation coefficient ranges from:

```text
-1 to +1
```

where the sign indicates the direction of association and the magnitude indicates the strength of the linear relationship.

### Multiple-Testing Correction

Because many participant-specific relationships were tested, raw p-values were adjusted using the **Benjamini-Hochberg False Discovery Rate (FDR)** procedure.

The primary significance criterion after correction was:

```text
FDR-adjusted p < 0.05
```

FDR correction was applied separately to the major analysis families.

---

## Statistical Results

The final statistical analysis included:

* **16 participants**
* **4,160 total tested relationships**
* **3,443 relationships with N >= 10**
* **528 raw statistically significant relationships**
* **100 FDR-significant relationships**

### Results by Analysis

| Analysis              | Total Tests | Valid N >= 10 | Raw Significant | FDR Significant |
| --------------------- | ----------: | ------------: | --------------: | --------------: |
| Same-day              |       1,040 |           900 |             180 |              43 |
| One-day lagged        |       1,040 |           860 |              63 |               4 |
| Seven-day history     |       1,040 |           794 |             103 |               6 |
| Sensitivity, |Z| >= 1 |       1,040 |           887 |             182 |              47 |
| **Total**             |   **4,160** |     **3,443** |         **528** |         **100** |

The results show that statistically detectable relationships were present in the dataset, while their occurrence varied across participants, behavioral variables, well-being indicators, and analysis windows.

These findings represent statistical associations. They do not establish causality and do not indicate that a behavioral deviation represents a specific health condition.

---

## Machine Learning

Following the statistical analysis, a personalized machine-learning system was developed and evaluated to investigate whether behavioral information could be used to model next-day changes in well-being at the individual level.

The machine-learning analysis focused on **next-day well-being change prediction**.

The system was designed around participant-specific models rather than assuming that one model would represent all participants equally.

### ML Pipeline

```text
Daily behavioral data
        ↓
Feature preparation
        ↓
Feature availability assessment
        ↓
Participant-specific feature set
        ↓
Personalized machine-learning model
        ↓
Next-day well-being change prediction
        ↓
Model evaluation
        ↓
Feature importance
        ↓
Behavior-level interpretation
```

### ML Targets

Five well-being targets were modeled:

* fatigue
* mood
* readiness
* sleep_quality
* stress

With 16 participants and 5 targets, the final analysis evaluated:

```text
16 × 5 = 80 personalized models
```

### Candidate Features

The ML pipeline generated **91 candidate feature columns** from the behavioral variables.

Feature availability was assessed separately for each participant.

The feature-availability analysis identified:

* 115 features available with some missing observations
* 69 fully available features
* 14 available sleep features under the sleep-specific missingness rule
* 9 features with longer missing streaks that remained available for analysis
* 1 missing-column case
* 0 features excluded because all of their first 10 calendar days were missing

Missing-data patterns were explicitly documented rather than silently ignored.

---

## ML Model Evaluation

The final ML analysis produced:

* **16 participants**
* **5 well-being targets**
* **80 personalized models**
* **91 candidate feature columns**
* **7,280 feature-importance records**

Model performance was evaluated using:

* R²
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### Overall ML Performance

| Metric             |  Result |
| ------------------ | ------: |
| Total models       |      80 |
| Positive R² models |      14 |
| Zero R² models     |       1 |
| Negative R² models |      65 |
| Mean R²            | -0.1147 |
| Median R²          | -0.0903 |
| Mean MAE           |  0.6871 |
| Mean RMSE          |  0.9236 |

The ML results show substantial variation across participant-specific models.

Most models did not achieve positive R² values. Therefore, the current results should not be interpreted as evidence of a validated predictive system.

Instead, the results demonstrate that a complete personalized ML pipeline was developed and evaluated, while also showing that predictive performance remains a major area for further research.

---

## Feature Importance

Feature importance was analyzed at both the derived-feature level and the original behavioral-variable level.

All 91 derived ML features were successfully mapped back to the original 13 behavioral variables.

No feature-importance rows remained unmapped.

The following table shows how frequently each behavioral variable appeared among participant-specific top-three feature sets:

| Behavioral Variable  | Top-3 Frequency |
| -------------------- | --------------: |
| Steps                |              49 |
| Exercise_Calories    |              30 |
| Sleep_Restlessness   |              28 |
| Deep_Sleep_Minutes   |              24 |
| Sleep_Hours          |              24 |
| Exercise_Duration    |              23 |
| Sleep_Score          |              20 |
| Sleep_Duration_Score |              14 |
| Exercise_Avg_HR      |              13 |
| Exercise_Distance    |               6 |
| Sleep_Revitalization |               6 |
| Sleep_Composition    |               2 |
| Exercise_Count       |               1 |

These frequencies describe how often a behavioral variable appeared among participant-specific top-three feature sets.

They should not be interpreted as universal predictors of well-being or as evidence that one behavior causes changes in a well-being outcome.

---

## Relationship Between Statistical and ML Analyses

The statistical and machine-learning components serve complementary purposes.

The statistical analyses ask:

> **Are personalized behavioral deviations associated with well-being indicators?**

The machine-learning analysis asks:

> **Can behavioral information be used to model next-day changes in well-being for individual participants?**

The statistical analysis provides interpretable participant-specific associations across multiple temporal windows.

The ML analysis extends this framework by combining behavioral features to evaluate individual-level predictive performance and identify important behavioral inputs.

The two analyses should therefore not be treated as interchangeable evidence.

Statistical association does not establish prediction, and feature importance in a predictive model does not establish causality.

---

## Repository Structure

```text
AI-Wellbeing-Project/
│
├── Code/
│   ├── 01_build_baseline.py
│   ├── 02_same_day_analysis.py
│   ├── 03_lagged_analysis.py
│   ├── 04_seven_day_analysis.py
│   ├── 05_sensitivity_analysis.py
│   ├── 06_fdr_correction.py
│   └── 07_final_summary.py
│
├── ML/
│   ├── Code/
│   │   ├── 08_prepare_ml_data.py
│   │   ├── 09_feature_availability.py
│   │   ├── 10_personalized_ml.py
│   │   └── 11_analyze_ml_results.py
│   └── README.md
│
├── data/
│   └── pmdata/
│
└── results/
    ├── baseline/
    ├── same_day/
    ├── lagged/
    ├── seven_day/
    ├── sensitivity/
    ├── fdr/
    ├── Final_Summary/
    └── ml/
```

---

## Reproducibility

The project maintains the analysis code and generated result files in the Git repository.

The main workflow is divided into numbered scripts:

```text
01 → Build personalized baseline and daily deviations
02 → Same-day analysis
03 → One-day lagged analysis
04 → Seven-day historical analysis
05 → Sensitivity analysis
06 → FDR correction
07 → Final statistical summary
08 → ML data preparation
09 → ML feature availability assessment
10 → Personalized ML modeling
11 → ML results analysis
```

Important methodological decisions are documented in the project's decision log.

Git commit history is maintained to provide a transparent record of changes to the analysis and implementation.

---

## Decision Framework

The major methodological decisions in the project include:

1. Use PMData as the final dataset.
2. Retain all 13 behavioral variables as candidate variables.
3. Use participant-specific baselines rather than universal thresholds.
4. Split baseline and analysis periods chronologically using a 50/50 approach.
5. Measure deviations using participant-specific standardized scores.
6. Use `|Z| >= 2` as the primary deviation threshold.
7. Use `|Z| >= 1` as a sensitivity analysis.
8. Require at least `N >= 10` usable observations for participant-specific statistical relationships.
9. Use Pearson correlation for linear association analysis.
10. Examine same-day, one-day lagged, and seven-day historical relationships.
11. Apply Benjamini-Hochberg FDR correction to control for multiple testing.
12. Develop participant-specific ML models for next-day well-being change.
13. Evaluate predictive performance using R², MAE, and RMSE.
14. Analyze feature importance at both derived-feature and original behavioral-variable levels.
15. Interpret deviations as potential signals rather than diagnoses.

---

## Limitations

### Small Sample Size

The final dataset contains 16 participants. This limits statistical power and the generalizability of the findings.

### Participant-Specific Analysis

The personalized approach is designed to capture individual differences, but participant-specific results may vary substantially and should not automatically be generalized to other populations.

### Well-Being Measurement

The five available indicators represent selected reported well-being-related states. They do not constitute a complete measurement of the multidimensional concept of well-being.

### Association Does Not Imply Causation

Observed correlations and predictive feature importance do not demonstrate causal relationships.

### ML Performance

Most personalized ML models did not achieve positive R² values. The current ML results therefore do not establish a validated predictive system.

### Dataset Characteristics

The conclusions are based on PMData and its available behavioral and self-reported measures. Results may differ in other datasets, populations, or real-world monitoring environments.

### No Clinical Validation

The system has not been clinically validated and should not be used to diagnose illness or determine an individual's health status.

### No Real-Time Deployment

The current project develops and evaluates the analytical and machine-learning framework. It does **not** deploy the model as a real-time application or clinical monitoring system.

---

## Ethical Interpretation

The purpose of the framework is to identify **potential changes in behavioral patterns that may be relevant to well-being**.

A detected deviation should therefore be interpreted as a signal for further investigation rather than a diagnosis, health warning, or proof of a problem.

The personalized approach recognizes that an unusual behavior for one person may be normal for another.

Any future real-world implementation would also need to consider:

* privacy and data protection;
* informed consent;
* responsible handling of wearable data;
* false positives and false alarms;
* individual differences;
* transparency of model outputs;
* and appropriate human interpretation.

---

## AI Use and Research Transparency

AI tools were used during the project to support tasks such as:

* improving wording and clarity;
* organizing technical documentation;
* reviewing code structure;
* explaining methodological concepts;
* supporting literature-search organization.

The researcher remained responsible for the research design, methodological decisions, computational workflow, interpretation of results, and final conclusions.

The substantive analyses were implemented through the project's reproducible Python code and documented in the repository.

---

## Future Work

Future work can extend the current framework in several directions:

1. Validate the approach using larger and more diverse datasets.
2. Evaluate the models using stronger longitudinal validation strategies.
3. Improve personalized prediction performance.
4. Investigate more robust temporal and sequential machine-learning methods.
5. Examine whether combinations of behavioral deviations provide stronger signals than individual variables.
6. Study how detected deviations could support early investigation of changes in well-being.
7. Explore integration with additional behavioral or symptom-related data for health-risk early-warning research.
8. Investigate possible applications of personalized behavioral monitoring in other contexts, including companion-animal adaptation.

These directions represent future research rather than capabilities demonstrated by the current system.

---

## Conclusion

This project developed a personalized framework for investigating whether AI can detect changes in well-being by learning individual behavioral patterns and identifying unusual deviations.

The project combines:

* personalized behavioral baselines;
* standardized daily deviations;
* same-day statistical analysis;
* one-day lagged analysis;
* seven-day historical analysis;
* sensitivity analysis;
* multiple-testing correction;
* participant-specific machine-learning models;
* predictive performance evaluation;
* and behavioral feature-importance analysis.

The final statistical analysis identified **100 FDR-significant participant-specific relationships** across the analysis families.

The machine-learning analysis evaluated **80 personalized models**, with **14 models achieving positive R²**.

The project therefore provides a reproducible research framework for **personalized well-being monitoring and early-detection research**.

The current findings provide a basis for further investigation, but they do not constitute clinical validation, diagnosis, causal inference, or a deployed real-time monitoring system.

