# AI-Wellbeing-Project
AI-based early detection of changes in well-being using behavioral data and personalized baselines.
# AI-Based Early Detection of Changes in Well-Being Among People Living Alone

## Project Overview

This project investigates whether changes in an individual's daily behavior and sleep patterns are associated with changes in self-reported well-being.

The main idea is to first learn each person's usual behavioral pattern and then identify deviations from their personal baseline.

## Dataset

The project uses the **PMData** dataset.

PMData contains daily behavioral and sleep measures collected from wearable devices, together with self-reported well-being measures.

### Behavioral and Sleep Variables

* Steps
* Exercise_Count
* Exercise_Duration
* Exercise_Distance
* Exercise_Calories
* Exercise_Avg_HR
* Sleep_Hours
* Sleep_Duration_Score
* Deep_Sleep_Minutes
* Sleep_Restlessness
* Sleep_Composition
* Sleep_Revitalization
* Sleep_Score

### Well-Being Variables

* fatigue
* mood
* readiness
* sleep_quality
* stress

## Analysis Approach

The current analysis follows these steps:

1. Create a personalized baseline for each participant.
2. Calculate daily behavioral deviations using personalized Z-scores.
3. Detect meaningful deviations using `|Z| ≥ 2`.
4. Perform a sensitivity analysis using `|Z| ≥ 1`.
5. Examine relationships between behavioral deviations and well-being.
6. Apply a minimum sample size of `N ≥ 10`.
7. Use Pearson correlation for participant-level relationships.
8. Apply Benjamini–Hochberg FDR correction.
9. Examine same-day relationships.
10. Examine one-day lagged relationships.
11. Examine the previous seven days of behavioral and sleep history.

## Temporal Analyses

Three temporal approaches are currently examined:

| Analysis               | Description                                                       | FDR-significant |
| ---------------------- | ----------------------------------------------------------------- | --------------: |
| Same-day               | Behavior and well-being on the same day                           |              44 |
| One-day lagged         | Behavior on day t → well-being on day t+1                         |               4 |
| Previous 7-day history | Mean behavior/sleep from the previous 7 days → current well-being |              42 |

The seven-day analysis is an exploratory extension and does not assume that seven days is the optimal time window.

## Main Results

The personalized approach identified meaningful behavioral deviations across all 16 participants.

The final same-day analysis found **44 FDR-significant relationships**.

The one-day lagged analysis found **4 FDR-significant relationships**.

The previous seven-day history analysis found **42 FDR-significant relationships**.

The results also showed differences between participants, supporting the use of personalized rather than universal behavioral baselines.

## Important Limitations

* The dataset contains 16 participants.
* Some observations are incomplete.
* The number of usable observations differs between relationships.
* The analyses are observational.
* Correlation does not establish causality.
* The one-day lag is not the same as validated prediction.
* The seven-day analysis is exploratory.
* The current project does not provide clinical diagnosis or validated clinical prediction.

## Project Structure

```text
AI-Wellbeing-Project/
│
├── data/
│   └── pmdata/
│
├── code/
│   ├── 02_correlations.py
│   ├── 03_personalized_baseline.py
│   ├── 04_change_detection.py
│   ├── 04b_change_detection_sensitivity.py
│   ├── 05_deviation_wellbeing.py
│   ├── 06_deviation_wellbeing_stats.py
│   ├── 07_full_period_wellbeing_analysis.py
│   ├── 08_significant_relationship_summary.py
│   ├── 09_daily_personalized_deviations.py
│   ├── 10_final_fdr_analysis.py
│   ├── 11_lagged_deviation_wellbeing.py
│   ├── 12_final_lagged_fdr.py
│   ├── 13_final_results_table.py
│   ├── 14_seven_day_history_wellbeing.py
│   ├── 15_seven_day_significant_summary.py
│   └── 16_compare_analysis_results.py
│
├── results/
│   ├── baseline/
│   ├── change_detection/
│   └── wellbeing_analysis/
│
├── DECISION_LOG.md
└── README.md
```

## Current Project Status

### Completed

* Dataset selection and preparation
* Personalized baseline construction
* Behavioral deviation calculation
* Change detection
* Sensitivity analysis
* Same-day analysis
* One-day lagged analysis
* Previous seven-day history analysis
* FDR correction
* Comparison of temporal analyses
* Participant-level interpretation

### Next Step

The next stage of the project will investigate machine-learning approaches for early detection using behavioral and sleep history from previous days.
