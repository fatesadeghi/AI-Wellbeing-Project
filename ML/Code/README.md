# Machine Learning for Personalized Well-Being Monitoring

This directory contains the machine-learning stage of the project.

The objective is to investigate whether individual behavioral patterns can be used to detect potential changes in well-being by learning each person's normal behavioral patterns and identifying unusual deviations.

## ML Pipeline

1. Prepare machine-learning features from the behavioral and well-being data.

2. Use all 13 behavioral variables as initial candidate features.

3. Incorporate personalized baseline information and recent behavioral history.

4. Apply participant-specific machine-learning models to predict next-day changes in well-being.

5. Identify which behavioral features are most informative for each participant and well-being target.

6. Aggregate model-based feature importance from derived features back to the 13 original behavioral variables.

7. Use the learned individual patterns as a basis for personalized well-being monitoring.

## Behavioral Variables

The initial feature set contains:

- Steps
- Exercise_Count
- Exercise_Duration
- Exercise_Distance
- Exercise_Calories
- Exercise_Avg_HR
- Sleep_Hours
- Sleep_Duration_Score
- Deep_Sleep_Minutes
- Sleep_Restlessness
- Sleep_Composition
- Sleep_Revitalization
- Sleep_Score

## Well-Being Indicators

The available well-being-related indicators are:

- fatigue
- mood
- readiness
- sleep_quality
- stress

## Personalized Modeling

The ML stage uses participant-specific models rather than treating all participants as one homogeneous population.

All 13 behavioral variables are initially considered as candidate features. Feature availability is evaluated at the participant level. If an activity/exercise variable is completely unavailable during the first 10 consecutive days, it is excluded for that participant. Sleep variables are not subject to this exclusion rule.

The current implementation generates derived behavioral features from the available data and trains separate models for each participant and well-being target.

## Current Results

The current implementation trained:

- 16 participants
- 5 well-being targets
- 80 participant-specific models
- 91 candidate ML feature columns

The models were evaluated using time-aware cross-validation.

The current results indicate that the dataset supports investigation of personalized behavioral patterns, but does not yet demonstrate strong general predictive performance for next-day well-being changes.

Feature importance is interpreted as model-based association with prediction and is not treated as causal evidence.

## Limitations

The ML results should be interpreted cautiously because:

- the dataset contains only 16 participants;
- observation lengths differ between participants;
- missing behavioral data are present;
- the current predictive performance is limited;
- feature importance does not establish causality;
- the results are intended for personalized monitoring research rather than diagnosis.

The machine-learning stage therefore complements the statistical analyses rather than replacing them.
