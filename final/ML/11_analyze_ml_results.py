import os
import re
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

ML_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ml"
)

OUTPUT_DIR = ML_RESULTS_DIR

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


PERFORMANCE_FILE = os.path.join(
    ML_RESULTS_DIR,
    "ml_model_performance.csv"
)

IMPORTANCE_FILE = os.path.join(
    ML_RESULTS_DIR,
    "ml_feature_importance.csv"
)

STATUS_FILE = os.path.join(
    ML_RESULTS_DIR,
    "ml_final_feature_status.csv"
)


# ============================================================
# 2. ORIGINAL 13 BEHAVIORAL VARIABLES
# ============================================================

BEHAVIOR_VARIABLES = [
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


WELLBEING_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def extract_behavior(feature_name):
    """
    Map a detailed ML feature back to one of the
    original 13 behavioral variables.
    """

    if pd.isna(feature_name):
        return np.nan

    feature = str(feature_name)

    # Exact variable name first
    for behavior in sorted(
        BEHAVIOR_VARIABLES,
        key=len,
        reverse=True
    ):
        if feature == behavior:
            return behavior

    # Features are expected to have the behavioral variable
    # at the beginning, followed by representations such as:
    # _raw, _baseline_deviation, _z, _abs_z, _7d_mean, etc.
    for behavior in sorted(
        BEHAVIOR_VARIABLES,
        key=len,
        reverse=True
    ):
        if feature.startswith(behavior + "_"):
            return behavior

    return np.nan


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ============================================================
# 4. HEADER
# ============================================================

print("=" * 70)
print("ANALYSIS OF PERSONALIZED MACHINE LEARNING RESULTS")
print("=" * 70)

print("\nLoading ML results...")


# ============================================================
# 5. LOAD FILES
# ============================================================

performance = pd.read_csv(
    PERFORMANCE_FILE
)

importance = pd.read_csv(
    IMPORTANCE_FILE
)

feature_status = pd.read_csv(
    STATUS_FILE
)

print(
    f"Performance rows: {len(performance)}"
)

print(
    f"Feature importance rows: {len(importance)}"
)

print(
    f"Feature status rows: {len(feature_status)}"
)


# ============================================================
# 6. MODEL PERFORMANCE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)


performance["R2"] = safe_numeric(
    performance["R2"]
)

performance["MAE"] = safe_numeric(
    performance["MAE"]
)

performance["RMSE"] = safe_numeric(
    performance["RMSE"]
)


total_models = len(performance)

positive_r2 = int(
    (performance["R2"] > 0).sum()
)

zero_r2 = int(
    (performance["R2"] == 0).sum()
)

negative_r2 = int(
    (performance["R2"] < 0).sum()
)

mean_r2 = performance["R2"].mean()

median_r2 = performance["R2"].median()

mean_mae = performance["MAE"].mean()

mean_rmse = performance["RMSE"].mean()


print(
    f"Total models: {total_models}"
)

print(
    f"Positive R2: {positive_r2}"
)

print(
    f"Zero R2: {zero_r2}"
)

print(
    f"Negative R2: {negative_r2}"
)

print(
    f"Mean R2: {mean_r2:.4f}"
)

print(
    f"Median R2: {median_r2:.4f}"
)

print(
    f"Mean MAE: {mean_mae:.4f}"
)

print(
    f"Mean RMSE: {mean_rmse:.4f}"
)


# Save overall performance summary
performance_summary = pd.DataFrame(
    [{
        "Total_Models": total_models,
        "Positive_R2": positive_r2,
        "Zero_R2": zero_r2,
        "Negative_R2": negative_r2,
        "Mean_R2": mean_r2,
        "Median_R2": median_r2,
        "Mean_MAE": mean_mae,
        "Mean_RMSE": mean_rmse
    }]
)

performance_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_performance_summary.csv"
    ),
    index=False
)


# ============================================================
# 7. PERFORMANCE BY WELL-BEING TARGET
# ============================================================

if "Target" in performance.columns:

    performance_by_target = (
        performance
        .groupby(
            "Target",
            as_index=False
        )
        .agg(
            Models=("Target", "size"),
            Mean_R2=("R2", "mean"),
            Median_R2=("R2", "median"),
            Positive_R2=(
                "R2",
                lambda x: int((x > 0).sum())
            ),
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean")
        )
        .sort_values(
            "Mean_R2",
            ascending=False
        )
    )

else:

    performance_by_target = pd.DataFrame()


if not performance_by_target.empty:

    performance_by_target.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ml_performance_by_target.csv"
        ),
        index=False
    )


# ============================================================
# 8. PERFORMANCE BY PARTICIPANT
# ============================================================

if "Participant" in performance.columns:

    performance_by_participant = (
        performance
        .groupby(
            "Participant",
            as_index=False
        )
        .agg(
            Models=("Participant", "size"),
            Mean_R2=("R2", "mean"),
            Median_R2=("R2", "median"),
            Positive_R2=(
                "R2",
                lambda x: int((x > 0).sum())
            ),
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean")
        )
        .sort_values(
            "Mean_R2",
            ascending=False
        )
    )

else:

    performance_by_participant = pd.DataFrame()


if not performance_by_participant.empty:

    performance_by_participant.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ml_performance_by_participant.csv"
        ),
        index=False
    )


# ============================================================
# 9. FEATURE MAPPING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MAPPING")
print("=" * 70)


importance["Behavior_Variable"] = (
    importance["Feature"]
    .apply(extract_behavior)
)


mapped_rows = int(
    importance["Behavior_Variable"].notna().sum()
)

unmapped_rows = int(
    importance["Behavior_Variable"].isna().sum()
)


print(
    f"Mapped rows: {mapped_rows}"
)

print(
    f"Unmapped rows: {unmapped_rows}"
)


if unmapped_rows > 0:

    unmapped = (
        importance[
            importance["Behavior_Variable"].isna()
        ]
        .copy()
    )

    unmapped.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ml_unmapped_features.csv"
        ),
        index=False
    )


# ============================================================
# 10. CLEAN FEATURE IMPORTANCE
# ============================================================

importance["Importance"] = safe_numeric(
    importance["Importance"]
)


importance_clean = importance[
    importance["Behavior_Variable"].notna()
].copy()


# ============================================================
# 11. AGGREGATE 91 FEATURES -> 13 BEHAVIORS
# ============================================================

behavior_importance = (
    importance_clean
    .groupby(
        [
            "Participant",
            "Target",
            "Behavior_Variable"
        ],
        as_index=False
    )
    .agg(
        Mean_Importance=(
            "Importance",
            "mean"
        ),
        Total_Importance=(
            "Importance",
            "sum"
        ),
        Max_Importance=(
            "Importance",
            "max"
        ),
        Feature_Count=(
            "Feature",
            "nunique"
        )
    )
)


# ============================================================
# 12. NORMALIZED IMPORTANCE WITHIN
#     PARTICIPANT × TARGET
# ============================================================

group_total = (
    behavior_importance
    .groupby(
        [
            "Participant",
            "Target"
        ]
    )["Total_Importance"]
    .transform("sum")
)


behavior_importance["Relative_Importance"] = np.where(
    group_total > 0,
    behavior_importance["Total_Importance"] / group_total,
    np.nan
)


# ============================================================
# 13. BEHAVIOR RANK WITHIN EACH
#     PARTICIPANT × TARGET
# ============================================================

behavior_importance["Behavior_Rank"] = (
    behavior_importance
    .groupby(
        [
            "Participant",
            "Target"
        ]
    )["Total_Importance"]
    .rank(
        ascending=False,
        method="min"
    )
)


behavior_importance = (
    behavior_importance
    .sort_values(
        [
            "Participant",
            "Target",
            "Behavior_Rank",
            "Behavior_Variable"
        ]
    )
)


behavior_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_behavior_importance_by_participant_target.csv"
    ),
    index=False
)


# ============================================================
# 14. OVERALL BEHAVIOR IMPORTANCE
# ============================================================

overall_behavior_importance = (
    behavior_importance
    .groupby(
        "Behavior_Variable",
        as_index=False
    )
    .agg(
        Participant_Target_Models=(
            "Behavior_Variable",
            "size"
        ),
        Mean_Total_Importance=(
            "Total_Importance",
            "mean"
        ),
        Mean_Relative_Importance=(
            "Relative_Importance",
            "mean"
        ),
        Median_Relative_Importance=(
            "Relative_Importance",
            "median"
        ),
        Mean_Absolute_Importance=(
            "Mean_Importance",
            "mean"
        ),
        Participants=(
            "Participant",
            "nunique"
        ),
        Targets=(
            "Target",
            "nunique"
        )
    )
)


overall_behavior_importance = (
    overall_behavior_importance
    .sort_values(
        "Mean_Relative_Importance",
        ascending=False
    )
)


overall_behavior_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_overall_behavior_importance.csv"
    ),
    index=False
)


# ============================================================
# 15. TOP 3 BEHAVIORS PER
#     PARTICIPANT × TARGET
# ============================================================

top3 = behavior_importance[
    behavior_importance["Behavior_Rank"] <= 3
].copy()


top3.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_top3_behaviors_by_participant_target.csv"
    ),
    index=False
)


# ============================================================
# 16. HOW OFTEN EACH BEHAVIOR APPEARS
#     IN TOP 3
# ============================================================

top3_behavior_frequency = (
    top3
    .groupby(
        "Behavior_Variable",
        as_index=False
    )
    .agg(
        Top3_Appearances=(
            "Behavior_Variable",
            "size"
        ),
        Unique_Participants_Top3=(
            "Participant",
            "nunique"
        ),
        Unique_Targets_Top3=(
            "Target",
            "nunique"
        ),
        Mean_Relative_Importance=(
            "Relative_Importance",
            "mean"
        ),
        Mean_Total_Importance=(
            "Total_Importance",
            "mean"
        )
    )
)


top3_behavior_frequency = (
    top3_behavior_frequency
    .sort_values(
        [
            "Top3_Appearances",
            "Mean_Relative_Importance"
        ],
        ascending=False
    )
)


top3_behavior_frequency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_top3_behavior_frequency.csv"
    ),
    index=False
)


# ============================================================
# 17. TOP 3 PARTICIPANT COVERAGE
#
# For each behavior:
# number of unique participants for whom
# it appeared in the top 3.
# ============================================================

top3_participants = (
    behavior_importance[
        behavior_importance["Behavior_Rank"] <= 3
    ]
    .groupby(
        "Behavior_Variable",
        as_index=False
    )
    .agg(
        Unique_Participants_Top3=(
            "Participant",
            "nunique"
        )
    )
)


# Merge with frequency table
top3_behavior_frequency = (
    top3_behavior_frequency
    .drop(
        columns=["Unique_Participants_Top3"],
        errors="ignore"
    )
    .merge(
        top3_participants,
        on="Behavior_Variable",
        how="left"
    )
)


top3_behavior_frequency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_top3_behavior_frequency.csv"
    ),
    index=False
)


# ============================================================
# 18. FEATURE REPRESENTATION IMPORTANCE
#
# This preserves the detailed ML feature representation,
# e.g. raw, z-score, deviation, rolling features.
# ============================================================

feature_representation = (
    importance_clean
    .groupby(
        "Feature",
        as_index=False
    )
    .agg(
        Mean_Importance=(
            "Importance",
            "mean"
        ),
        Median_Importance=(
            "Importance",
            "median"
        ),
        Participants=(
            "Participant",
            "nunique"
        ),
        Targets=(
            "Target",
            "nunique"
        )
    )
    .sort_values(
        "Mean_Importance",
        ascending=False
    )
)


feature_representation.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_detailed_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 19. PARTICIPANT-SPECIFIC BEHAVIOR IMPORTANCE
# ============================================================

participant_behavior_importance = (
    behavior_importance
    .groupby(
        [
            "Participant",
            "Behavior_Variable"
        ],
        as_index=False
    )
    .agg(
        Targets=(
            "Target",
            "nunique"
        ),
        Mean_Total_Importance=(
            "Total_Importance",
            "mean"
        ),
        Mean_Relative_Importance=(
            "Relative_Importance",
            "mean"
        ),
        Max_Relative_Importance=(
            "Relative_Importance",
            "max"
        ),
        Top3_Target_Count=(
            "Behavior_Rank",
            lambda x: int((x <= 3).sum())
        )
    )
)


participant_behavior_importance = (
    participant_behavior_importance
    .sort_values(
        [
            "Participant",
            "Mean_Relative_Importance"
        ],
        ascending=[
            True,
            False
        ]
    )
)


participant_behavior_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_behavior_importance_by_participant.csv"
    ),
    index=False
)


# ============================================================
# 20. TARGET-SPECIFIC BEHAVIOR IMPORTANCE
# ============================================================

target_behavior_importance = (
    behavior_importance
    .groupby(
        [
            "Target",
            "Behavior_Variable"
        ],
        as_index=False
    )
    .agg(
        Participants=(
            "Participant",
            "nunique"
        ),
        Mean_Total_Importance=(
            "Total_Importance",
            "mean"
        ),
        Mean_Relative_Importance=(
            "Relative_Importance",
            "mean"
        ),
        Median_Relative_Importance=(
            "Relative_Importance",
            "median"
        ),
        Top3_Count=(
            "Behavior_Rank",
            lambda x: int((x <= 3).sum())
        )
    )
)


target_behavior_importance = (
    target_behavior_importance
    .sort_values(
        [
            "Target",
            "Mean_Relative_Importance"
        ],
        ascending=[
            True,
            False
        ]
    )
)


target_behavior_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_behavior_importance_by_target.csv"
    ),
    index=False
)


# ============================================================
# 21. ACTIVE FEATURE STATUS SUMMARY
# ============================================================

if (
    "Participant" in feature_status.columns
    and "Behavior_Variable" in feature_status.columns
):

    feature_status_summary = (
        feature_status
        .groupby(
            [
                "Participant",
                "Behavior_Variable"
            ],
            as_index=False
        )
        .first()
    )

else:

    feature_status_summary = pd.DataFrame()


if not feature_status_summary.empty:

    feature_status_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ml_feature_status_by_behavior.csv"
        ),
        index=False
    )


# ============================================================
# 22. CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BEHAVIOR-LEVEL FEATURE IMPORTANCE")
print("=" * 70)

display_columns = [
    "Behavior_Variable",
    "Top3_Appearances",
    "Unique_Participants_Top3",
    "Unique_Targets_Top3",
    "Mean_Relative_Importance"
]


print(
    top3_behavior_frequency[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 23. PARTICIPANT × TARGET TOP 3
# ============================================================

print("\n" + "=" * 70)
print("TOP 3 BEHAVIORS BY PARTICIPANT AND TARGET")
print("=" * 70)

for participant in sorted(
    top3["Participant"].dropna().unique()
):

    participant_data = top3[
        top3["Participant"] == participant
    ].copy()

    print(
        f"\n{participant}"
    )

    for target in sorted(
        participant_data["Target"]
        .dropna()
        .unique()
    ):

        subset = participant_data[
            participant_data["Target"] == target
        ].sort_values(
            "Behavior_Rank"
        )

        behaviors = []

        for _, row in subset.iterrows():

            behaviors.append(
                f"{int(row['Behavior_Rank'])}. "
                f"{row['Behavior_Variable']} "
                f"({row['Relative_Importance']:.3f})"
            )

        print(
            f"  {target}: "
            + "; ".join(behaviors)
        )


# ============================================================
# 24. FINAL FILE LIST
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

output_files = [
    "ml_performance_summary.csv",
    "ml_performance_by_target.csv",
    "ml_performance_by_participant.csv",
    "ml_behavior_importance_by_participant_target.csv",
    "ml_overall_behavior_importance.csv",
    "ml_top3_behaviors_by_participant_target.csv",
    "ml_top3_behavior_frequency.csv",
    "ml_detailed_feature_importance.csv",
    "ml_behavior_importance_by_participant.csv",
    "ml_behavior_importance_by_target.csv",
    "ml_feature_status_by_behavior.csv"
]

for filename in output_files:

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):

        print(
            f"[OK] {filename}"
        )


print("\nDone.")