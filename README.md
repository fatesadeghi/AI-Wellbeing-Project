# AI-Based Early Detection of Changes in Well-Being Among People Living Alone

## Project Overview

This project investigates whether AI can detect changes in a person's well-being by learning their daily habits and identifying unusual behavioral patterns.

The central research question is:

> **Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?**

The project uses a personalized approach. Instead of assuming that the same behavioral threshold applies to everyone, the analysis first learns each participant's usual behavioral pattern and then examines deviations from that individual baseline.

Behavioral deviations are treated as **potential signals of changes in well-being**, not as diagnoses or proof of a health condition.

---

## Dataset

The project uses the **PMData** dataset.

The dataset contains daily behavioral and sleep measures collected from wearable devices together with self-reported well-being indicators.

The analysis includes **16 participants**.

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

- fatigue
- mood
- readiness
- sleep_quality
- stress

These indicators are used as measures of reported well-being-related states and do not represent the full multidimensional construct of well-being.

---

## Research Pipeline

The project follows the following pipeline:

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
Personalized well-being monitoring
