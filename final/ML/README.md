\# Machine Learning for Personalized Well-Being Monitoring



This directory contains the machine-learning stage of the project.



The objective is to investigate whether individual behavioral patterns can be used to detect potential changes in well-being by learning each person's normal behavioral patterns and identifying unusual deviations.



\## Planned Pipeline



1\. Prepare machine-learning features from the behavioral and well-being data.

2\. Use all 13 behavioral variables as initial candidate features.

3\. Incorporate personalized baseline information and recent behavioral history.

4\. Train and evaluate participant-aware machine-learning models.

5\. Identify which behavioral features are most informative for each participant.

6\. Use the learned individual patterns for personalized well-being monitoring.

7\. Develop a supportive recommendation layer based on detected changes.



\## Behavioral Variables



The initial feature set contains:



\- Steps

\- Exercise\_Count

\- Exercise\_Duration

\- Exercise\_Distance

\- Exercise\_Calories

\- Exercise\_Avg\_HR

\- Sleep\_Hours

\- Sleep\_Duration\_Score

\- Deep\_Sleep\_Minutes

\- Sleep\_Restlessness

\- Sleep\_Composition

\- Sleep\_Revitalization

\- Sleep\_Score



\## Well-Being Indicators



The available well-being-related indicators are:



\- fatigue

\- mood

\- readiness

\- sleep\_quality

\- stress



The machine-learning stage will build on the personalized-baseline and statistical analyses developed in the previous stages of the project.

