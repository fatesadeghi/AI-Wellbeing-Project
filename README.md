# AI-Based Early Detection of Changes in Well-Being and Quality of Life Among People Living Alone

**Personalized behavioral analysis and machine learning using the PMData dataset**

---

## Overview

This project investigates whether changes in an individual's daily behavioral patterns can be associated with changes in well-being, and whether personalized machine-learning models can use behavioral information to model subsequent changes in well-being.

The project is based on a **personalized monitoring framework**:

> **Learn an individual's normal behavior → detect unusual deviations → examine their relationship with well-being → evaluate whether these patterns can support personalized early detection.**

Instead of applying the same behavioral threshold to everyone, the project first establishes a behavioral baseline for each participant. Daily observations are then compared with that individual's own baseline to identify unusual behavioral deviations.

These deviations are subsequently examined using statistical analyses and personalized machine-learning models.

**Important:** The system is a research and early-warning framework. Behavioral deviations are treated as potential signals and **not as diagnoses or proof of a health condition**.

---

## Research Question

> **Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?**

The project addresses this question through two complementary analytical approaches:

1. **Statistical analysis**
   Examines associations between personalized behavioral deviations and well-being-related measures.

2. **Personalized machine learning**
   Evaluates whether participant-specific behavioral information can be used to model next-day changes in well-being.

The statistical and machine-learning analyses answer related but different questions and are therefore interpreted separately.

---

# Dataset

The project uses the **PMData dataset**, which contains longitudinal behavioral, activity, sleep, and self-reported well-being information collected from participants using wearable devices and daily questionnaires.

The final analysis includes:

* **16 participants**
* **13 behavioral variables**
* **5 well-being indicators**

The project retains the complete set of behavioral variables as candidate predictors rather than selecting a smaller subset before analysis.

---

## Behavioral Variables

The 13 behavioral variables used throughout the analysis are:

1. `Steps`
2. `Exercise_Count`
3. `Exercise_Duration`
4. `Exercise_Distance`
5. `Exercise_Calories`
6. `Exercise_Avg_HR`
7. `Sleep_Hours`
8. `Sleep_Duration_Score`
9. `Deep_Sleep_Minutes`
10. `Sleep_Restlessness`
11. `Sleep_Composition`
12. `Sleep_Revitalization`
13. `Sleep_Score`

These variables represent daily activity and sleep-related behavior.

---

## Well-Being Indicators

Five self-reported well-being-related indicators are modeled:

* `fatigue`
* `mood`
* `readiness`
* `sleep_quality`
* `stress`

These variables represent specific reported aspects of well-being rather than the complete multidimensional construct of well-being.

---

# Methodology

## 1. Participant-Level Data Preparation

The raw PMData information is transformed into participant-level daily datasets.

The analysis preserves the longitudinal structure of the data so that earlier observations can be used to establish a personal baseline and later observations can be evaluated against it.

---

## 2. Personalized Baseline

A separate behavioral baseline is constructed for every participant.

For each participant, the available observations are ordered chronologically and divided into two periods:

```text
Earlier 50% of observations
        ↓
Personal baseline
        ↓
Later 50% of observations
        ↓
Analysis period
```

The baseline is calculated separately for every behavioral variable using:

* mean
* standard deviation

This chronological split prevents future observations from being used to define the earlier behavioral baseline.

---

## 3. Personalized Behavioral Deviation

For each participant and behavioral variable, daily observations in the analysis period are standardized relative to that participant's own baseline.

The deviation score is:

$$
Z = \frac{X-\mu_{baseline}}{\sigma_{baseline}}
$$

where:

* \(X\) = observed daily value
* \(\mu_{baseline}\) = participant-specific baseline mean
* \(\sigma_{baseline}\) = participant-specific baseline standard deviation

Interpretation:

* positive \(Z\) → value is above the participant's usual level
* negative \(Z\) → value is below the participant's usual level
* larger \(|Z|\) → greater deviation from the participant's usual variability

The deviation score identifies **unusual behavior**, not whether the behavior is inherently positive or negative.

---

## 4. Primary Deviation Threshold

The primary analysis uses:

$$
|Z| \geq 2
$$

as the definition of a substantial behavioral deviation.

Therefore:

```text
|Z| < 2    → not classified as a primary deviation
|Z| ≥ 2    → classified as a primary deviation
```

Both unusually high and unusually low behavioral values are retained.

---

## 5. Sensitivity Analysis

To examine whether the results depend strongly on the deviation threshold, a second analysis uses:

$$
|Z| \geq 1
$$

The `|Z| ≥ 2` analysis remains the primary analysis, while `|Z| ≥ 1` is treated as a sensitivity analysis.

This provides a more sensitive test that captures smaller deviations.

---

# Statistical Analysis

The statistical component examines participant-level relationships between behavioral measures and well-being-related variables.

A minimum of:

$$
N \geq 10
$$

usable paired observations is required for a participant × behavioral-variable × well-being relationship to be included.

Relationships with fewer than 10 usable observations are excluded from statistical interpretation.

---

## Pearson Correlation

Pearson correlation is used to evaluate linear associations.

$$
-1 \leq r \leq 1
$$

where:

* positive \(r\) indicates a positive linear association
* negative \(r\) indicates a negative linear association
* values closer to zero indicate weaker linear association

Correlation measures association and **does not establish causality**.

---

# Temporal Analyses

The project evaluates behavioral information at multiple temporal scales.

## 1. Same-Day Analysis

The first analysis examines whether behavioral information and well-being are associated on the same day.

```text
Behavior at day t
        ↓
Well-being at day t
```

Final results:

| Metric                       | Result |
| ---------------------------- | -----: |
| Total tested relationships   |  1,040 |
| Valid relationships (N ≥ 10) |    900 |
| Raw significant              |    180 |
| FDR-significant              |     43 |

---

## 2. One-Day Lagged Analysis

The second analysis examines whether behavioral information is associated with well-being on the following day.

```text
Behavior at day t
        ↓
Well-being at day t+1
```

Final results:

| Metric                       | Result |
| ---------------------------- | -----: |
| Total tested relationships   |  1,040 |
| Valid relationships (N ≥ 10) |    860 |
| Raw significant              |     63 |
| FDR-significant              |      4 |

This analysis introduces temporal ordering, but temporal ordering alone does **not establish causality**.

---

## 3. Seven-Day Historical Analysis

The third temporal analysis examines whether behavioral history during the previous seven days is associated with well-being on the current day.

```text
Behavior during previous 7 days
        ↓
Current-day well-being
```

Final results:

| Metric                       | Result |
| ---------------------------- | -----: |
| Total tested relationships   |  1,040 |
| Valid relationships (N ≥ 10) |    794 |
| Raw significant              |    103 |
| FDR-significant              |      6 |

---

# Multiple-Testing Correction

Because the project evaluates many participant × behavior × well-being relationships, multiple-testing correction is required.

The project uses the **Benjamini–Hochberg False Discovery Rate (FDR)** procedure.

The final significance criterion is:

$$
q < 0.05
$$

The distinction between raw and corrected significance is maintained throughout the results.

```text
Raw significance:
p < 0.05

FDR significance:
q < 0.05
```

The FDR-corrected results are used for the final statistical interpretation.

---

# Final Statistical Summary

The completed statistical analyses produced the following results:

| Analysis             | Total Tests | Valid N ≥ 10 | Raw Significant | FDR Significant |
| -------------------- | ----------: | -----------: | --------------: | --------------: |
| Same-day             |       1,040 |          900 |             180 |              43 |
| One-day lagged       |       1,040 |          860 |              63 |               4 |
| Seven-day history    |       1,040 |          794 |             103 |               6 |
| Sensitivity, |Z| ≥ 1 |       1,040 |          887 |             182 |              47 |
| **Total**            |   **4,160** |    **3,443** |         **528** |         **100** |

The statistical analyses therefore identified FDR-significant associations across the completed analysis families.

These results represent **associations**, not causal effects.

---

# Machine Learning

The project includes a personalized machine-learning component to investigate whether daily behavioral information can be used to model **next-day changes in well-being**.

The ML pipeline was designed around the same personalized framework used in the statistical analysis.

---

## ML Pipeline

```text
Participant-level daily data
            ↓
Feature availability analysis
            ↓
Behavioral feature preparation
            ↓
Derived behavioral features
            ↓
Next-day well-being target
            ↓
Participant-specific model
            ↓
Model evaluation
            ↓
Feature importance
            ↓
Behavior-level interpretation
```

---

## ML Feature Space

The original behavioral feature space contains the same 13 behavioral variables used in the statistical analysis.

The ML preprocessing pipeline generates:

**91 candidate derived feature columns**

These derived features are mapped back to the original behavioral variables to allow interpretation at the behavioral level.

All 91 derived features were successfully mapped to their corresponding original behavioral variables.

---

# Missing Data and Feature Availability

Because longitudinal wearable data contain incomplete observations, feature availability is evaluated separately for each participant.

The ML pipeline explicitly tracks categories including:

* `Available`
* `Available_With_Missing`
* `Available_With_Long_Missing_Streak`
* `Available_Sleep_Exempt`
* `Missing_Column`
* `Excluded_First_10_Days_Missing`

The feature-availability analysis identified:

| Availability Category              | Count |
| ---------------------------------- | ----: |
| Available_With_Missing             |   115 |
| Available                          |    69 |
| Available_Sleep_Exempt             |    14 |
| Available_With_Long_Missing_Streak |     9 |
| Missing_Column                     |     1 |
| Excluded_First_10_Days_Missing     |     0 |

Missingness is therefore documented explicitly rather than silently ignored.

---

# Personalized ML Models

Instead of training one universal model for the entire population, the project trains separate models for each participant and each well-being target.

The five targets are:

1. `fatigue`
2. `mood`
3. `readiness`
4. `sleep_quality`
5. `stress`

With:

* 16 participants
* 5 well-being targets

the complete ML pipeline produces:

$$
16 \times 5 = 80
$$

**participant-specific models**.

---

# ML Evaluation

The personalized models are evaluated using:

* \(R^2\)
* MAE
* RMSE

These metrics provide complementary information about predictive performance.

## Results Across 80 Models

| Metric           |  Result |
| ---------------- | ------: |
| Models           |      80 |
| Positive \(R^2\) |      14 |
| Zero \(R^2\)     |       1 |
| Negative \(R^2\) |      65 |
| Mean \(R^2\)     | -0.1147 |
| Median \(R^2\)   | -0.0903 |
| Mean MAE         |  0.6871 |
| Mean RMSE        |  0.9236 |

The results show substantial variation between participant-specific models.

Most models produced non-positive \(R^2\) values. Therefore, the current dataset and modeling configuration do not provide evidence of strong general predictive performance across participants.

The ML component should consequently be interpreted as an **implemented and evaluated research model**, rather than a validated real-world prediction system.

---

# Feature Importance

Prediction performance alone does not indicate which behavioral patterns were used by the models.

The ML pipeline therefore also performs feature-importance analysis.

The pipeline produced:

* **7,280 feature-importance rows**
* **91 derived feature columns**
* a complete mapping from derived features to the original 13 behavioral variables

Feature importance was examined:

* overall
* by participant
* by well-being target
* among the top three behaviors for each participant × target combination

---

## Top-3 Behavioral Frequency

The frequency with which each behavioral variable appeared among the top three model features across participant × target combinations was:

| Behavioral Variable  | Top-3 Appearances |
| -------------------- | ----------------: |
| Steps                |                49 |
| Exercise_Calories    |                30 |
| Sleep_Restlessness   |                28 |
| Deep_Sleep_Minutes   |                24 |
| Sleep_Hours          |                24 |
| Exercise_Duration    |                23 |
| Sleep_Score          |                20 |
| Sleep_Duration_Score |                14 |
| Exercise_Avg_HR      |                13 |
| Exercise_Distance    |                 6 |
| Sleep_Revitalization |                 6 |
| Sleep_Composition    |                 2 |
| Exercise_Count       |                 1 |

These frequencies describe **model feature usage** under the implemented modeling procedure.

They should not be interpreted as causal importance.

---

# Statistical Analysis vs. Machine Learning

The two analytical components address different questions.

### Statistical analysis

> **Are behavioral measures associated with well-being-related measures?**

This component focuses on:

* participant-level associations
* behavioral deviations
* temporal relationships
* correlation
* statistical significance
* multiple-testing correction

### Machine learning

> **Can participant-specific behavioral information be used to model changes in well-being?**

This component focuses on:

* prediction
* participant-specific models
* next-day well-being targets
* model performance
* feature importance

A behavioral variable can therefore:

* show a statistical association without producing strong predictive performance, or
* contribute to a model without being statistically significant in the correlation analysis.

The two approaches are complementary rather than interchangeable.

---

# Complete Research Pipeline

The complete methodology can be summarized as:

```text
                         PMData
                           │
                           ▼
               Participant-level preparation
                           │
                           ▼
                  13 behavioral variables
                           │
                           ▼
             Personalized chronological baseline
                           │
                           ▼
               Daily personalized deviations
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
          |Z| ≥ 2 primary        |Z| ≥ 1 sensitivity
                │                     │
                └──────────┬──────────┘
                           ▼
                  Statistical analyses
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Same-day     One-day lag   Seven-day
              │            │            │
              └────────────┼────────────┘
                           ▼
                   Pearson correlation
                           │
                           ▼
              Benjamini–Hochberg FDR
                           │
                           ▼
                  Statistical results
                           │
                           ▼
                  Personalized ML
                           │
                           ▼
                 91 derived features
                           │
                           ▼
                 16 participants
                         ×
                  5 well-being targets
                           │
                           ▼
                    80 ML models
                           │
                           ▼
              R² / MAE / RMSE evaluation
                           │
                           ▼
                  Feature importance
                           │
                           ▼
             Personalized behavioral signals
```

---

# Repository Structure

The repository is organized into separate statistical-analysis and machine-learning components.

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
│   └── Code/
│       ├── 08_prepare_ml_data.py
│       ├── 09_feature_availability.py
│       ├── 10_personalized_ml.py
│       └── 11_analyze_ml_results.py
│
├── data/
│   └── pmdata/
│
├── results/
│   ├── baseline/
│   └── ...
│
├── DECISION_LOG.md
├── README.md
└── .gitignore
```

The exact generated result files may change as the analysis pipeline is updated, while the methodological structure is documented in `DECISION_LOG.md`.

---

# Analysis Code

The main statistical pipeline is organized sequentially:

| Script                       | Purpose                                                                          |   |                            |
| ---------------------------- | -------------------------------------------------------------------------------- | - | -------------------------- |
| `01_build_baseline.py`       | Construct participant-specific chronological baselines and behavioral deviations |   |                            |
| `02_same_day_analysis.py`    | Run same-day behavioral/well-being analysis                                      |   |                            |
| `03_lagged_analysis.py`      | Run one-day lagged analysis                                                      |   |                            |
| `04_seven_day_analysis.py`   | Analyze previous seven days of behavioral history                                |   |                            |
| `05_sensitivity_analysis.py` | Repeat analysis using the `                                                      | Z | ≥ 1` sensitivity threshold |
| `06_fdr_correction.py`       | Apply Benjamini–Hochberg FDR correction                                          |   |                            |
| `07_final_summary.py`        | Generate final statistical summaries                                             |   |                            |

---

# Machine-Learning Code

The ML pipeline is organized as:

| Script                       | Purpose                                                |
| ---------------------------- | ------------------------------------------------------ |
| `08_prepare_ml_data.py`      | Prepare participant-level ML data and derived features |
| `09_feature_availability.py` | Evaluate missingness and feature availability          |
| `10_personalized_ml.py`      | Train and evaluate participant-specific ML models      |
| `11_analyze_ml_results.py`   | Analyze model performance and feature importance       |

---

# Reproducibility

The project is designed to maintain a reproducible analysis workflow.

The repository contains:

* data-processing code
* statistical-analysis code
* machine-learning code
* generated analysis results
* methodological documentation
* a research decision log
* version-controlled project history

The methodological decisions and their rationale are documented in:

**`DECISION_LOG.md`**

This includes decisions concerning:

* dataset selection
* behavioral-variable selection
* personalized baselines
* chronological splitting
* deviation thresholds
* missing-data handling
* minimum sample size
* correlation analysis
* temporal analyses
* FDR correction
* ML design
* model evaluation
* feature importance
* ethical interpretation
* research limitations

---

# How to Run the Project

The analysis is designed to be executed sequentially.

### Step 1 — Prepare the dataset

Place the PMData data in the expected project data directory.

```text
data/
└── pmdata/
```

### Step 2 — Run the statistical pipeline

Run the scripts in numerical order:

```bash
python Code/01_build_baseline.py
python Code/02_same_day_analysis.py
python Code/03_lagged_analysis.py
python Code/04_seven_day_analysis.py
python Code/05_sensitivity_analysis.py
python Code/06_fdr_correction.py
python Code/07_final_summary.py
```

### Step 3 — Run the ML pipeline

Then execute:

```bash
python ML/Code/08_prepare_ml_data.py
python ML/Code/09_feature_availability.py
python ML/Code/10_personalized_ml.py
python ML/Code/11_analyze_ml_results.py
```

The scripts generate the intermediate and final outputs used for the statistical and ML results.

---

# Research Results at a Glance

The current completed analysis provides the following overall picture:

### Statistical analysis

* 16 participants
* 13 behavioral variables
* 5 well-being indicators
* 4,160 tested relationships across four analysis families
* 3,443 relationships meeting the `N ≥ 10` criterion
* 528 raw significant relationships
* 100 FDR-significant relationships

### Machine learning

* 91 derived candidate features
* 16 participants
* 5 well-being targets
* 80 personalized models
* 14 models with positive \(R^2\)
* 65 models with negative \(R^2\)
* mean \(R^2 = -0.1147\)
* mean MAE = 0.6871
* mean RMSE = 0.9236
* 7,280 feature-importance records

---

# Interpretation

The statistical results demonstrate that behavioral measures can show measurable associations with self-reported well-being-related variables under the implemented personalized analytical framework.

However, the machine-learning results indicate that these associations do not automatically translate into strong predictive performance.

Most participant-specific models produced non-positive \(R^2\) values. This means that the current implementation should not be presented as a validated predictive or clinical system.

Instead, the project demonstrates the feasibility of a **personalized research framework** for investigating behavioral signals of well-being.

The distinction is important:

```text
Association ≠ Prediction
Prediction ≠ Causation
Behavioral deviation ≠ Diagnosis
Feature importance ≠ Causal importance
```

---

# Limitations

The current project has several important limitations.

## Sample size

The analysis includes only:

**16 participants**

This limits statistical power and generalizability.

## Unequal data availability

Participants do not necessarily have identical amounts of usable data across all variables.

## Missing observations

Longitudinal wearable datasets can contain missing days, missing measurements, and incomplete variables.

## Participant-specific ML models

Individual models may have relatively few observations available for training and evaluation.

## Observational data

The data are observational rather than experimental.

Therefore, observed relationships cannot establish causal effects.

## ML performance

The current predictive performance is generally limited, with most models producing non-positive \(R^2\).

## No clinical validation

The system has not undergone clinical validation.

## No external validation cohort

The current analysis does not include an independent external validation dataset.

## Well-being measurement

Well-being is represented by five self-reported indicators rather than a complete multidimensional measurement of well-being.

---

# Ethical Interpretation

The system is intended as an **early-warning and research framework**, not a diagnostic tool.

An unusual behavioral pattern may have many possible explanations.

For example:

* unusually low activity may reflect illness, fatigue, schedule changes, or other circumstances;
* unusually high activity may also have multiple explanations;
* unusual sleep duration may arise from many behavioral or situational factors.

The current analysis cannot determine the underlying cause of a behavioral deviation.

Therefore:

> **A detected behavioral deviation should be interpreted as a potential signal for further investigation, not as evidence of a health condition.**

---

# AI Use and Research Transparency

AI tools were used during the project as a research and coding assistant.

AI-supported activities included:

* code development
* debugging
* restructuring analysis scripts
* statistical concept explanation
* documentation improvement
* English-language editing
* report organization
* discussion of methodological alternatives

AI-generated suggestions were not treated as independent scientific evidence.

Final methodological decisions, interpretation of results, and research conclusions remain the responsibility of the researcher.

---

# Current Project Status

The current repository contains the completed research pipeline covering:

* personalized baseline construction
* behavioral deviation detection
* primary and sensitivity thresholds
* same-day statistical analysis
* one-day lagged analysis
* seven-day historical analysis
* multiple-testing correction
* final statistical summaries
* ML data preparation
* feature-availability analysis
* personalized ML modeling
* model evaluation
* feature-importance analysis
* methodological documentation

The statistical and machine-learning pipelines have been implemented and executed on the PMData participants.

The current results support the **feasibility of studying personalized behavioral signals of well-being**, but they do not establish a clinically validated prediction system.

---

# Future Work

Future development should focus on:

1. larger longitudinal datasets
2. additional participants
3. stronger temporal validation
4. improved predictive modeling
5. more robust handling of missing data
6. external validation
7. additional well-being measures
8. improved feature engineering
9. longitudinal and personalized model comparison
10. evaluation on independent datasets
11. careful assessment before any real-world deployment

---

# Core Research Concept

The central concept of the project can be summarized as:

```text
Learn the individual's normal behavior
                ↓
Detect unusual deviations
                ↓
Examine their relationship with well-being
                ↓
Model subsequent changes
                ↓
Identify potentially useful personalized signals
```

The goal is not to define a universal behavioral threshold for everyone.

Instead, the project investigates whether **changes relative to an individual's own behavioral baseline** can provide useful information about changes in that individual's reported well-being.

---

## Research Position

The current project provides a complete framework combining:

* personalized behavioral baselines
* deviation detection
* temporal statistical analysis
* multiple-testing correction
* personalized machine learning
* model evaluation
* behavioral feature-importance analysis

The statistical analyses identified FDR-significant associations across the completed analysis families.

The machine-learning system was successfully implemented and evaluated at the participant level, although overall predictive performance was limited.

Therefore, the main contribution of the current project is a **personalized methodology for investigating behavioral signals of well-being**, rather than a clinically validated prediction system.
