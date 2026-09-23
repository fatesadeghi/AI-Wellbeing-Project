بله. این هم نسخه کامل با Markdown واقعی، آماده کپی مستقیم در `README.md`:

````markdown
# AI-Based Early Detection of Changes in Well-Being and Quality of Life Among People Living Alone

**Personalized behavioral analysis and machine learning using the PMData dataset**

---

## Overview

This project investigates whether changes in an individual's daily behavioral patterns can be associated with changes in well-being, and whether recent behavioral patterns can support personalized prediction of individual well-being.

The project is based on a personalized framework:

> **Learn an individual's normal behavior → examine deviations from that personal pattern → investigate their relationship with well-being → evaluate personalized prediction.**

Instead of assuming that the same behavioral pattern is normal for everyone, the project uses each participant's own historical behavioral information as a reference point.

The project combines two complementary analytical approaches:

1. **Participant-level statistical analysis**
2. **Personalized machine learning**

The statistical analysis investigates associations between behavioral measures and well-being.

The machine-learning analysis investigates whether recent behavioral patterns can be used to predict an individual's Wellbeing Index.

**Important:** The system is a research framework. Behavioral deviations are treated as potential signals and not as diagnoses or proof of a health condition.

---

## Research Question

> **Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?**

The project addresses this question through personalized behavioral analysis, temporal analysis, and individual-level machine learning.

---

# Dataset

The project uses the **PMData dataset**, which contains longitudinal behavioral, activity, sleep, and self-reported well-being information collected from participants using wearable devices and daily questionnaires.

The final analysis includes:

- **16 participants**
- **13 behavioral variables**
- **5 well-being indicators**

The complete behavioral variable set was retained rather than selecting a smaller subset before analysis.

---

## Behavioral Variables

The 13 behavioral variables used in the analysis are:

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

These variables represent daily physical activity and sleep-related behavior.

---

## Well-Being Indicators

Five self-reported well-being-related indicators are used:

- `fatigue`
- `mood`
- `readiness`
- `sleep_quality`
- `stress`

These indicators represent specific reported aspects of daily well-being rather than the complete multidimensional construct of well-being.

---

# Methodology

## 1. Participant-Level Data Preparation

The PMData information is transformed into participant-level daily datasets.

The longitudinal structure of the data is preserved so that earlier observations can be used to establish participant-specific behavioral reference points and later observations can be examined in relation to them.

---

## 2. Personalized Baseline

A separate behavioral baseline is constructed for each participant.

For each participant, observations are ordered chronologically. The first 50% of chronological observations are used to establish the participant's personal baseline.

```text
Earlier 50% of observations
            ↓
    Personal baseline
            ↓
Later 50% of observations
            ↓
      Analysis period
````

The baseline is calculated separately for each behavioral variable using:

* mean
* standard deviation

This chronological structure prevents later observations from being used to define the earlier personal reference pattern.

---

## 3. Personalized Behavioral Deviation

For each participant and behavioral variable, observations in the analysis period are standardized relative to that participant's own baseline.

The deviation score is:

```text
Z = (X − μ) / σ
```

where:

* `X` = observed behavioral value
* `μ` = participant-specific baseline mean
* `σ` = participant-specific baseline standard deviation

Interpretation:

* positive `Z` → value is above the participant's usual level
* negative `Z` → value is below the participant's usual level
* larger `|Z|` → greater deviation from the participant's baseline variability

The deviation score describes how unusual an observation is relative to the participant's own historical pattern. It does not determine whether the behavior is inherently positive or negative.

---

# Statistical Analysis

The statistical analysis examines participant-level relationships between behavioral measures and well-being-related measures.

A minimum of:

```text
N ≥ 10
```

usable paired observations is required for a participant × behavioral-variable × well-being relationship to be included.

Relationships with fewer than 10 usable observations are excluded from the main correlation analysis.

---

## Pearson Correlation

Pearson correlation is used to evaluate linear associations.

```text
−1 ≤ r ≤ 1
```

where:

* positive `r` indicates a positive linear association
* negative `r` indicates a negative linear association
* values closer to zero indicate weaker linear association

Correlation measures association and does not establish causality.

---

# Seven-Day Historical Analysis

The final statistical analysis examines whether behavioral information from the previous seven days is associated with well-being on the current day.

For a target day `t`, the behavioral history is calculated from:

```text
t−7, t−6, t−5, t−4, t−3, t−2, t−1
```

The current day's behavioral observation is not included in the seven-day historical measure.

The temporal relationship is therefore:

```text
Previous seven days of behavioral data
                  ↓
          Same-day well-being
```

The seven-day window is an exploratory temporal framework. It is not assumed to represent an optimal biological or clinical period.

---

## Final Statistical Results

The final seven-day analysis considered:

* 16 participants
* 13 behavioral variables
* 5 well-being variables

This produced:

```text
16 × 13 × 5 = 1,040
```

potential participant-level relationships.

After applying the minimum observation requirement:

* **845** relationships had `N ≥ 10`
* **794** produced valid correlation results
* **103** were nominally significant at `p < 0.05`

Across valid relationships:

```text
Mean r   = −0.0164
Mean |r| = 0.1688
```

The 103 significant results represent participant-level associations rather than 103 unique participants.

The results are interpreted as associations and not as evidence of causation.

---

# Machine Learning

The project includes a personalized machine-learning component to investigate whether recent behavioral patterns can support prediction of an individual's well-being.

The machine-learning stage uses behavioral information from the previous seven calendar days and predicts the participant's Wellbeing Index.

---

## ML Pipeline

```text
Participant-level daily data
            ↓
Wellbeing Index construction
            ↓
Feature availability analysis
            ↓
Seven-day behavioral feature construction
            ↓
Participant-specific model training
            ↓
Wellbeing Index prediction
            ↓
Prediction evaluation
```

---

## Wellbeing Index

The Wellbeing Index is constructed from the five daily self-reported well-being measures:

* `fatigue`
* `mood`
* `readiness`
* `sleep_quality`
* `stress`

The index provides a single target for the personalized machine-learning stage while combining multiple dimensions of daily reported well-being.

---

## ML Feature Construction

The machine-learning pipeline uses the same 13 behavioral variables as the statistical analysis.

For each behavioral variable, two temporal features are calculated from the previous seven calendar days:

1. seven-day slope
2. seven-day change

Therefore:

```text
13 behavioral variables × 2 temporal features
= 26 behavioral features
```

The seven-day slope represents the direction of change across the available observations in the seven-day window.

The seven-day change represents the difference between the first and last available observations in that window.

The behavioral information from the target day is not used as a prediction feature.

---

## Personalized ML Models

A separate Random Forest regression model is trained for each participant.

The models use earlier observations from the same participant for training.

A minimum of 10 available training observations is required before generating a prediction.

Missing feature values are handled using median imputation within the machine-learning pipeline.

The prediction target is the participant's Wellbeing Index.

---

# ML Prediction Evaluation

The resulting predictions are compared with the observed Wellbeing Index values for the same participant and target date.

The evaluation uses:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R²

The final prediction pipeline generated:

* **1,863 prediction records**
* **1,454 evaluated predictions with observed Wellbeing Index values**

Overall performance:

| Metric |  Result |
| ------ | ------: |
| MAE    |  0.2763 |
| RMSE   |  0.3578 |
| R²     | −0.1451 |

At the participant level:

| Metric |    Mean |
| ------ | ------: |
| MAE    |  0.2713 |
| RMSE   |  0.3527 |
| R²     | −0.1513 |

All 16 participants had negative individual R² values.

The negative R² values indicate that, under the evaluation procedure used in this study, the predictions did not explain the observed variation in the Wellbeing Index better than the corresponding mean-based reference.

Therefore, the current machine-learning results should be interpreted as an exploratory evaluation of the personalized prediction framework rather than as evidence of a reliable real-world prediction system.

---

# Statistical Analysis vs. Machine Learning

The two analytical components address different questions.

### Statistical analysis

> **Are behavioral measures associated with well-being-related measures?**

This component focuses on:

* participant-level relationships
* personalized behavioral patterns
* seven-day behavioral history
* Pearson correlation
* nominal statistical significance

### Machine learning

> **Can recent participant-specific behavioral information be used to predict an individual's Wellbeing Index?**

This component focuses on:

* temporal behavioral features
* personalized models
* prediction
* model evaluation
* MAE
* RMSE
* R²

The two approaches are complementary rather than interchangeable.

A statistical association does not necessarily produce strong predictive performance, and predictive modelling does not by itself establish causality.

---

# Missing Data and Feature Availability

Longitudinal wearable data contain incomplete observations and participant-specific differences in data availability.

The project therefore evaluates feature availability explicitly rather than silently ignoring missing observations.

The machine-learning pipeline tracks:

* `Available`
* `Available_With_Missing`
* `Available_With_Long_Missing_Streak`
* `Available_Sleep_Exempt`
* `Missing_Column`
* `Excluded_First_10_Days_Missing`

The feature-availability analysis identified:

| Availability Category                | Count |
| ------------------------------------ | ----: |
| `Available_With_Missing`             |   115 |
| `Available`                          |    69 |
| `Available_Sleep_Exempt`             |    14 |
| `Available_With_Long_Missing_Streak` |     9 |
| `Missing_Column`                     |     1 |
| `Excluded_First_10_Days_Missing`     |     0 |

Missing observations are not interpreted as evidence that a behavior did not occur.

---

# Complete Research Pipeline

The final research workflow can be summarized as:

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
                           ▼
              Previous seven days of behaviour
                           │
                           ▼
                  Same-day well-being
                           │
                           ▼
                  Pearson correlation
                           │
                           ▼
              Statistical relationship results
                           │
                           ▼
                  Wellbeing Index
                           │
                           ▼
               Seven-day temporal features
                           │
                           ▼
                 26 behavioral features
                           │
                           ▼
              Participant-specific model
                           │
                           ▼
                 Random Forest regression
                           │
                           ▼
                  Wellbeing prediction
                           │
                           ▼
                  MAE / RMSE / R²
```

---

# Repository Structure

The final repository is organized into separate statistical and machine-learning components.

```text
AI-Wellbeing-Project/
│
├── Code/
│   ├── 01_build_baseline.py
│   ├── 02_seven_day_analysis.py
│   └── Report
│
├── ML/
│   ├── 01_build_wellbeing_index.py
│   ├── 02_build_7day_features.py
│   ├── 03_train_personal_model.py
│   ├── 04_predict_and_report.py
│   ├── 05_test_predictions.py
│   ├── run_pipeline.py
│   └── Results/
│
├── data/
│   └── pmdata/
│
├── results/
│   ├── baseline/
│   ├── seven_day/
│   └── final/
│
├── DECISION_LOG.md
├── README.md
└── .gitignore
```

---

# Analysis Code

The final statistical pipeline is organized sequentially:

| Script                     | Purpose                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `01_build_baseline.py`     | Construct participant-specific chronological baselines and daily personalized deviations |
| `02_seven_day_analysis.py` | Analyse previous seven days of behavioral history in relation to same-day well-being     |
| `Report`                   | Generate final statistical summaries and visual outputs                                  |

---

# Machine-Learning Code

The final ML pipeline is organized as:

| Script                        | Purpose                                          |
| ----------------------------- | ------------------------------------------------ |
| `01_build_wellbeing_index.py` | Construct the Wellbeing Index                    |
| `02_build_7day_features.py`   | Construct seven-day temporal behavioral features |
| `03_train_personal_model.py`  | Train participant-specific Random Forest models  |
| `04_predict_and_report.py`    | Generate predictions and prediction reports      |
| `05_test_predictions.py`      | Evaluate prediction performance                  |
| `run_pipeline.py`             | Run the ML workflow                              |

Generated ML results are stored in:

```text
ML/Results/
```

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

The main methodological decisions and their rationale are documented in:

`DECISION_LOG.md`

The repository also preserves project development history through Git commits.

---

# How to Run the Project

## Step 1 — Prepare the Dataset

Place the PMData data in:

```text
data/
└── pmdata/
```

---

## Step 2 — Run the Statistical Pipeline

Run:

```bash
python Code/01_build_baseline.py
python Code/02_seven_day_analysis.py
python Code/Report
```

The scripts generate the baseline, seven-day statistical results, summary tables, and visual outputs.

---

## Step 3 — Run the ML Pipeline

Run:

```bash
python ML/01_build_wellbeing_index.py
python ML/02_build_7day_features.py
python ML/03_train_personal_model.py
python ML/04_predict_and_report.py
python ML/05_test_predictions.py
```

The generated results are stored in:

```text
ML/Results/
```

---

# Research Results at a Glance

## Statistical Analysis

The final statistical analysis includes:

* **16 participants**
* **13 behavioral variables**
* **5 well-being variables**
* **1,040 potential participant-level relationships**
* **845 relationships with N ≥ 10**
* **794 valid correlations**
* **103 nominally significant associations**
* **Mean r = −0.0164**
* **Mean |r| = 0.1688**

---

## Machine Learning

The final machine-learning analysis includes:

* **13 behavioral variables**
* **26 temporal behavioral features**
* **16 participants**
* **1,863 prediction records**
* **1,454 evaluated predictions**
* **MAE = 0.2763**
* **RMSE = 0.3578**
* **R² = −0.1451**

All 16 participants had negative individual R² values.

---

# Interpretation

The statistical analysis provides evidence of participant-level associations between behavioral patterns and self-reported well-being measures under the implemented seven-day analytical framework.

However, statistical association does not automatically translate into strong predictive performance.

The machine-learning evaluation produced negative R² values for all participants under the implemented prediction procedure.

Therefore, the current implementation should be understood as a personalized research framework rather than a validated predictive or clinical system.

The distinction is important:

```text
Association ≠ Prediction

Prediction ≠ Causation

Behavioral deviation ≠ Diagnosis
```

---

# Limitations

The current project has several important limitations.

## Sample Size

The analysis includes 16 participants.

This limits the extent to which the findings can be generalized beyond the participants represented in the dataset.

## Unequal Data Availability

Participants do not necessarily have identical amounts of usable data across all behavioural variables.

## Missing Observations

Longitudinal wearable datasets can contain missing days, incomplete measurements, and differences in data availability.

## Participant-Specific Modelling

Personalized models depend on the amount of historical data available for each participant.

Participants with fewer usable observations provide less information for model training and evaluation.

## Observational Data

The data are observational rather than experimental.

Therefore, observed relationships cannot establish causal effects.

## Machine-Learning Performance

The current predictive performance is limited, with negative R² values across all participants.

## No Clinical Validation

The system has not undergone clinical validation.

## No External Validation Cohort

The current analysis does not include an independent external validation dataset.

## Well-Being Measurement

Well-being is represented using five self-reported indicators rather than a complete multidimensional measurement of well-being.

---

# Ethical Interpretation

The system is intended as a research framework for personalized well-being analysis, not as a diagnostic tool.

An unusual behavioral pattern may have many possible explanations.

For example:

* unusually low activity may reflect illness, fatigue, schedule changes, or other circumstances;
* unusually high activity may also have different personal or situational explanations;
* unusual sleep duration may have multiple possible causes.

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
* behavioral deviation analysis
* seven-day historical analysis
* participant-level correlation analysis
* Wellbeing Index construction
* seven-day temporal feature construction
* personalized machine-learning modelling
* prediction evaluation
* methodological documentation
* reproducibility documentation

The statistical and machine-learning pipelines have been implemented and executed on the PMData participants.

The current results support the feasibility of studying personalized behavioral signals of well-being, but they do not establish a clinically validated prediction system.

---

# Future Work

Future development should focus on:

1. larger longitudinal datasets
2. additional participants
3. longer observation periods
4. stronger temporal validation
5. improved predictive modelling
6. more robust missing-data strategies
7. additional well-being measures
8. richer behavioral feature engineering
9. comparison of personalized and population-level models
10. independent validation datasets
11. linguistic and communication-related features
12. health-related and psychologically relevant measures where appropriate and ethically justified

Future work could also investigate whether speech, text, vocabulary, word choice, and other linguistic patterns provide additional information about changes in individual well-being.

The same personalized framework could potentially be extended to companion animals using activity, movement, sleep-related measurements, and changes in daily routines.

Such extensions would require appropriate privacy, consent, ethical safeguards, and professional oversight.

---

# Core Research Concept

The central concept of the project can be summarized as:

```text
Learn the individual's normal behavior
                ↓
Identify deviations from the personal pattern
                ↓
Examine their relationship with well-being
                ↓
Use recent behavioral history for prediction
                ↓
Evaluate personalized prediction
```

The goal is not to define a universal behavioral threshold for everyone.

Instead, the project investigates whether changes relative to an individual's own behavioral history can provide useful information about changes in that individual's reported well-being.

---

# Research Position

The current project provides a reproducible framework combining:

* personalized behavioral baselines
* behavioral deviation analysis
* seven-day temporal analysis
* participant-level statistical relationships
* personalized machine learning
* prediction evaluation
* methodological documentation

The statistical analysis identified 103 nominally significant participant-level associations in the final seven-day analysis.

The machine-learning system was implemented and evaluated at the participant level, although the current predictive performance was limited.

The main contribution of the project is therefore a **personalized methodology for investigating behavioral signals of well-being**, rather than a clinically validated prediction system.

```
```
