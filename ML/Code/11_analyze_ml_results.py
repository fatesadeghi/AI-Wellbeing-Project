import os
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

# Current file:
# AI-Wellbeing-Project/ML/Code/11_analyze_ml_results.py

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

ML_DIR = os.path.join(
    RESULTS_DIR,
    "ml"
)

OUTPUT_DIR = ML_DIR

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

PERFORMANCE_FILE = os.path.join(
    ML_DIR,
    "ml_model_performance.csv"
)

IMPORTANCE_FILE = os.path.join(
    ML_DIR,
    "ml_feature_importance.csv"
)

FEATURE_STATUS_FILE = os.path.join(
    ML_DIR,
    "ml_final_feature_status.csv"
)


# ============================================================
# 3. BEHAVIORAL VARIABLES
# ============================================================

BEHAVIOR_VARS = [
    "Steps",
    "Exercise_Count",
    "Exercise_Duration",
    "Exercise_Distance",
    "Exercise_Calories",
    "Exercise_Avg_HR",
    "Sleep_Hours",
    "Sleep_Duration_Score",
    "Deep_Sleep_Minutes",
    "Sleep_Restlessness",
    "Sleep_Composition",
    "Sleep_Revitalization",
    "Sleep_Score"
]

WELLBEING_VARS = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]


# ============================================================
# 4. LOAD INPUT FILES
# ============================================================

for file_path in [
    PERFORMANCE_FILE,
    IMPORTANCE_FILE,
    FEATURE_STATUS_FILE
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


performance_df = pd.read_csv(
    PERFORMANCE_FILE
)

importance_df = pd.read_csv(
    IMPORTANCE_FILE
)

feature_status_df = pd.read_csv(
    FEATURE_STATUS_FILE
)


print("=" * 70)
print("ML RESULTS ANALYSIS")
print("=" * 70)

print(
    f"Performance rows: {len(performance_df)}"
)

print(
    f"Feature importance rows: {len(importance_df)}"
)

print(
    f"Feature status rows: {len(feature_status_df)}"
)


# ============================================================
# 5. MAP ML FEATURES BACK TO BEHAVIORAL VARIABLES
# ============================================================

def map_feature_to_behavior(feature_name):
    """
    Map a derived ML feature back to its original
    behavioral variable.

    Example:
        Steps__rolling_7
        -> Steps
    """

    if not isinstance(
        feature_name,
        str
    ):
        return np.nan

    for variable in sorted(
        BEHAVIOR_VARS,
        key=len,
        reverse=True
    ):

        prefix = (
            f"{variable}__"
        )

        if feature_name.startswith(
            prefix
        ):
            return variable

    return np.nan


importance_df[
    "Behavioral_Variable"
] = importance_df[
    "Feature"
].apply(
    map_feature_to_behavior
)


unmapped = importance_df[
    importance_df[
        "Behavioral_Variable"
    ].isna()
]


print(
    f"Unmapped feature rows: "
    f"{len(unmapped)}"
)


# ============================================================
# 6. NORMALIZE FEATURE IMPORTANCE WITHIN
#    PARTICIPANT × WELLBEING
# ============================================================

group_columns = [
    "Participant",
    "Wellbeing"
]


group_total_importance = (
    importance_df
    .groupby(
        group_columns
    )["Importance"]
    .transform("sum")
)


importance_df[
    "Relative_Importance"
] = np.where(
    group_total_importance > 0,
    importance_df["Importance"]
    / group_total_importance,
    np.nan
)


# ============================================================
# 7. AGGREGATE DERIVED FEATURES TO
#    ORIGINAL BEHAVIORAL VARIABLES
# ============================================================

behavior_importance = (
    importance_df
    .dropna(
        subset=["Behavioral_Variable"]
    )
    .groupby(
        [
            "Participant",
            "Wellbeing",
            "Behavioral_Variable"
        ],
        as_index=False
    )
    .agg(
        Importance=(
            "Importance",
            "sum"
        ),
        Relative_Importance=(
            "Relative_Importance",
            "sum"
        ),
        Feature_Representations=(
            "Feature",
            "count"
        )
    )
)


# ============================================================
# 8. TOP 3 BEHAVIORS PER
#    PARTICIPANT × WELLBEING
# ============================================================

behavior_importance[
    "Rank"
] = (
    behavior_importance
    .groupby(
        [
            "Participant",
            "Wellbeing"
        ]
    )["Relative_Importance"]
    .rank(
        method="first",
        ascending=False
    )
)


top3 = behavior_importance[
    behavior_importance["Rank"] <= 3
].copy()

top3 = top3.sort_values(
    [
        "Participant",
        "Wellbeing",
        "Rank"
    ]
)


# ============================================================
# 9. TOP-3 BEHAVIOR FREQUENCY
# ============================================================

top3_frequency = (
    top3
    .groupby(
        "Behavioral_Variable",
        as_index=False
    )
    .agg(
        Top3_Appearances=(
            "Behavioral_Variable",
            "size"
        ),
        Participants=(
            "Participant",
            "nunique"
        ),
        Wellbeing_Targets=(
            "Wellbeing",
            "nunique"
        ),
        Mean_Relative_Importance=(
            "Relative_Importance",
            "mean"
        )
    )
    .sort_values(
        [
            "Top3_Appearances",
            "Mean_Relative_Importance"
        ],
        ascending=[
            False,
            False
        ]
    )
)


# ============================================================
# 10. OVERALL BEHAVIOR IMPORTANCE
# ============================================================

overall_behavior_importance = (
    behavior_importance
    .groupby(
        "Behavioral_Variable",
        as_index=False
    )
    .agg(
        Mean_Importance=(
            "Importance",
            "mean"
        ),
        Mean_Relative_Importance=(
            "Relative_Importance",
            "mean"
        ),
        Participant_Target_Combinations=(
            "Behavioral_Variable",
            "size"
        ),
        Participants=(
            "Participant",
            "nunique"
        ),
        Wellbeing_Targets=(
            "Wellbeing",
            "nunique"
        )
    )
    .sort_values(
        "Mean_Relative_Importance",
        ascending=False
    )
)


# ============================================================
# 11. BEHAVIOR IMPORTANCE BY PARTICIPANT
# ============================================================

behavior_importance_by_participant = (
    behavior_importance
    .groupby(
        [
            "Participant",
            "Behavioral_Variable"
        ],
        as_index=False
    )
    .agg(
        Mean_Importance=(
            "Importance",
            "mean"
        ),
        Mean_Relative_Importance=(
            "Relative_Importance",
            "mean"
        ),
        Wellbeing_Targets=(
            "Wellbeing",
            "nunique"
        )
    )
)


# ============================================================
# 12. BEHAVIOR IMPORTANCE BY TARGET
# ============================================================

behavior_importance_by_target = (
    behavior_importance
    .groupby(
        [
            "Wellbeing",
            "Behavioral_Variable"
        ],
        as_index=False
    )
    .agg(
        Mean_Importance=(
            "Importance",
            "mean"
        ),
        Mean_Relative_Importance=(
            "Relative_Importance",
            "mean"
        ),
        Participants=(
            "Participant",
            "nunique"
        )
    )
)


# ============================================================
# 13. DETAILED FEATURE IMPORTANCE
# ============================================================

detailed_feature_importance = (
    importance_df
    .sort_values(
        [
            "Participant",
            "Wellbeing",
            "Relative_Importance"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )
    .copy()
)


# ============================================================
# 14. PERFORMANCE SUMMARY
# ============================================================

performance_summary = pd.DataFrame([{
    "Total_Models": len(
        performance_df
    ),
    "Positive_R2_Models": int(
        (
            performance_df["R2"] > 0
        ).sum()
    ),
    "Zero_R2_Models": int(
        (
            performance_df["R2"] == 0
        ).sum()
    ),
    "Negative_R2_Models": int(
        (
            performance_df["R2"] < 0
        ).sum()
    ),
    "Mean_R2": performance_df[
        "R2"
    ].mean(),
    "Median_R2": performance_df[
        "R2"
    ].median(),
    "Mean_MAE": performance_df[
        "MAE"
    ].mean(),
    "Mean_RMSE": performance_df[
        "RMSE"
    ].mean()
}])


# ============================================================
# 15. PERFORMANCE BY WELLBEING TARGET
# ============================================================

performance_by_target = (
    performance_df
    .groupby(
        "Wellbeing",
        as_index=False
    )
    .agg(
        Models=(
            "Wellbeing",
            "size"
        ),
        Mean_R2=(
            "R2",
            "mean"
        ),
        Median_R2=(
            "R2",
            "median"
        ),
        Mean_MAE=(
            "MAE",
            "mean"
        ),
        Mean_RMSE=(
            "RMSE",
            "mean"
        )
    )
)


# ============================================================
# 16. PERFORMANCE BY PARTICIPANT
# ============================================================

performance_by_participant = (
    performance_df
    .groupby(
        "Participant",
        as_index=False
    )
    .agg(
        Models=(
            "Participant",
            "size"
        ),
        Mean_R2=(
            "R2",
            "mean"
        ),
        Median_R2=(
            "R2",
            "median"
        ),
        Mean_MAE=(
            "MAE",
            "mean"
        ),
        Mean_RMSE=(
            "RMSE",
            "mean"
        )
    )
)


# ============================================================
# 17. SAVE OUTPUT FILES
# ============================================================

OUTPUT_FILES = {
    "ml_performance_summary.csv":
        performance_summary,

    "ml_performance_by_target.csv":
        performance_by_target,

    "ml_performance_by_participant.csv":
        performance_by_participant,

    "ml_behavior_importance_by_participant_target.csv":
        behavior_importance,

    "ml_overall_behavior_importance.csv":
        overall_behavior_importance,

    "ml_top3_behaviors_by_participant_target.csv":
        top3,

    "ml_top3_behavior_frequency.csv":
        top3_frequency,

    "ml_detailed_feature_importance.csv":
        detailed_feature_importance,

    "ml_behavior_importance_by_participant.csv":
        behavior_importance_by_participant,

    "ml_behavior_importance_by_target.csv":
        behavior_importance_by_target
}


for filename, output_df in OUTPUT_FILES.items():

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    output_df.to_csv(
        output_path,
        index=False
    )


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ML PERFORMANCE SUMMARY")
print("=" * 70)

print(
    performance_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("TOP-3 BEHAVIOR FREQUENCY")
print("=" * 70)

print(
    top3_frequency.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

for filename in OUTPUT_FILES:
    print(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )


print("\n" + "=" * 70)
print("ML RESULTS ANALYSIS COMPLETE")
print("=" * 70)
