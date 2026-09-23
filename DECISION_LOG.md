# Research Decision Log

## Project

**AI-Based Early Detection of Changes in Well-Being and Quality of Life Among People Living Alone**

## Central Research Question

> Can AI detect changes in a person's well-being by learning their daily habits and spotting unusual behavior?

This decision log documents the main methodological decisions made throughout the project, the reasons for those decisions, and how they were implemented.

The purpose of this document is to support:

- methodological transparency;
- reproducibility;
- traceability of research decisions;
- interpretation of the final results;
- and clear separation between statistical association and machine-learning prediction.

---

# 1. Dataset Selection

## Decision

Use the PMData dataset as the main dataset for the final analysis.

## Reason

The project requires both objective behavioural measurements and well-being-related information.

The initial exploration considered the CASAS smart-home dataset. However, it did not provide sufficient direct self-reported well-being information for the intended analysis.

PMData provided a more suitable combination of:

- behavioural and activity measurements;
- sleep-related measurements;
- self-reported well-being variables.

## Outcome

PMData became the final dataset used for the statistical and machine-learning analyses.

---

# 2. Behavioural Variable Selection

## Decision

Retain the full set of 13 behavioural variables rather than reducing the analysis to a smaller subset.

## Behavioural Variables

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

## Reason

Well-being is multidimensional, and different aspects of daily behaviour may provide different information.

Reducing the behavioural set before analysis could remove potentially informative patterns.

The final analysis therefore retained all 13 behavioural variables as candidate behavioural measures.

## Outcome

All 13 behavioural variables were retained for the statistical analysis and as the underlying behavioural variables for the machine-learning feature construction.

---

# 3. Well-Being Variables

## Decision

Use five self-reported well-being-related variables as outcome measures.

## Variables

1. `fatigue`
2. `mood`
3. `readiness`
4. `sleep_quality`
5. `stress`

## Reason

These variables represent different reported dimensions related to daily well-being.

Using multiple indicators allows the analysis to examine whether behavioural patterns are associated with different aspects of self-reported well-being rather than relying on a single measure.

## Outcome

The five variables were retained for the statistical analysis.

For the machine-learning stage, these daily well-being measures were combined into a Wellbeing Index used as the prediction target.

---

# 4. Participant-Level Analysis

## Decision

Perform the main statistical analysis separately for each participant.

## Reason

The central research question focuses on whether AI can learn an individual's own behavioural patterns and identify meaningful changes in relation to that individual's well-being.

Participants can have different:

- activity levels;
- sleep patterns;
- daily routines;
- behavioural variability;
- relationships between behaviour and well-being.

Combining all participants into a single homogeneous pattern could therefore hide important individual differences.

## Outcome

The statistical analysis was performed at the participant level rather than treating all participants as one homogeneous population.

---

# 5. Personalized Behavioural Baseline

## Decision

Construct a separate behavioural baseline for each participant.

## Reason

The same behavioural value may represent normal behaviour for one person but an unusual change for another.

For example, a particular number of daily steps may be normal for one participant but unusual for another.

Therefore, a population-wide behavioural threshold was not considered sufficient for the personalized framework.

## Implementation

For each participant, available observations were ordered chronologically.

The first 50% of the participant's chronological observations were used to establish the personal baseline.

For each behavioural variable, the baseline included:

- mean;
- standard deviation.

These statistics were calculated separately for each participant and behavioural variable.

## Outcome

Each participant received an individual behavioural baseline that was used as the reference point for subsequent behavioural deviation analysis.

---

# 6. Chronological Baseline Construction

## Decision

Use a chronological 50/50 split for baseline construction.

## Reason

The baseline should represent the individual's earlier behavioural pattern rather than incorporating information from later observations.

A chronological split preserves the temporal structure of the longitudinal data and avoids using future observations to define the earlier personal reference pattern.

## Implementation

For each participant:

```text
Earlier 50% of observations
            ↓
    Personal baseline
            ↓
Later 50% of observations
            ↓
      Analysis period
