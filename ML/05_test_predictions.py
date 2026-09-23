import os
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "ML",
    "Results"
)

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "wellbeing_predictions_full.csv"
)

WELLBEING_FILE = os.path.join(
    RESULTS_DIR,
    "wellbeing_index.csv"
)

PARTICIPANT_OUTPUT = os.path.join(
    RESULTS_DIR,
    "prediction_test_by_participant.csv"
)

OVERALL_OUTPUT = os.path.join(
    RESULTS_DIR,
    "prediction_test_overall.csv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "prediction_test_summary.txt"
)


# ============================================================
# 2. METRIC FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = np.nan

    return mae, rmse, r2


# ============================================================
# 3. START
# ============================================================

print("=" * 80)
print("ML PREDICTION TEST")
print("=" * 80)


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

if not os.path.exists(PREDICTIONS_FILE):
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_FILE}"
    )

if not os.path.exists(WELLBEING_FILE):
    raise FileNotFoundError(
        f"Wellbeing index file not found:\n{WELLBEING_FILE}"
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

predictions = pd.read_csv(
    PREDICTIONS_FILE
)

wellbeing = pd.read_csv(
    WELLBEING_FILE
)

print(
    f"Prediction rows loaded: {len(predictions)}"
)

print(
    f"Wellbeing rows loaded: {len(wellbeing)}"
)


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_prediction_columns = [
    "participant_id",
    "target_date",
    "predicted_wellbeing",
    "prediction_status"
]

required_wellbeing_columns = [
    "participant_id",
    "date",
    "Wellbeing_Index"
]

missing_prediction_columns = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]

missing_wellbeing_columns = [
    column
    for column in required_wellbeing_columns
    if column not in wellbeing.columns
]

if missing_prediction_columns:

    raise ValueError(
        "Missing prediction columns:\n"
        + "\n".join(missing_prediction_columns)
    )

if missing_wellbeing_columns:

    raise ValueError(
        "Missing wellbeing columns:\n"
        + "\n".join(missing_wellbeing_columns)
    )


# ============================================================
# 7. PREPARE DATES AND NUMERIC VALUES
# ============================================================

predictions["target_date"] = pd.to_datetime(
    predictions["target_date"],
    errors="coerce"
)

wellbeing["date"] = pd.to_datetime(
    wellbeing["date"],
    errors="coerce"
)

predictions["predicted_wellbeing"] = pd.to_numeric(
    predictions["predicted_wellbeing"],
    errors="coerce"
)

wellbeing["Wellbeing_Index"] = pd.to_numeric(
    wellbeing["Wellbeing_Index"],
    errors="coerce"
)


# ============================================================
# 8. KEEP ONLY ACTUAL PREDICTIONS
# ============================================================

predictions = predictions[
    predictions["prediction_status"] == "Predicted"
].copy()

print(
    f"Predicted rows available for testing: "
    f"{len(predictions)}"
)


# ============================================================
# 9. MATCH ACTUAL WELLBEING VALUES
# ============================================================

evaluation_df = predictions.merge(
    wellbeing[
        [
            "participant_id",
            "date",
            "Wellbeing_Index"
        ]
    ],
    left_on=[
        "participant_id",
        "target_date"
    ],
    right_on=[
        "participant_id",
        "date"
    ],
    how="left"
)

evaluation_df = evaluation_df.rename(
    columns={
        "Wellbeing_Index": "actual_wellbeing"
    }
)


# ============================================================
# 10. KEEP ONLY COMPLETE PAIRS
# ============================================================

evaluation_df = evaluation_df.dropna(
    subset=[
        "predicted_wellbeing",
        "actual_wellbeing"
    ]
).copy()

print(
    f"Rows with both predicted and actual wellbeing: "
    f"{len(evaluation_df)}"
)

if len(evaluation_df) == 0:

    raise ValueError(
        "No prediction rows could be matched with "
        "actual Wellbeing_Index values."
    )


# ============================================================
# 11. OVERALL METRICS
# ============================================================

overall_mae, overall_rmse, overall_r2 = calculate_metrics(
    evaluation_df["actual_wellbeing"],
    evaluation_df["predicted_wellbeing"]
)

overall_table = pd.DataFrame(
    [
        {
            "Level": "Overall",
            "N": len(evaluation_df),
            "MAE": overall_mae,
            "RMSE": overall_rmse,
            "R2": overall_r2
        }
    ]
)


# ============================================================
# 12. PARTICIPANT-LEVEL METRICS
# ============================================================

participant_results = []

for participant_id, group in evaluation_df.groupby(
    "participant_id"
):

    if len(group) < 2:
        continue

    mae, rmse, r2 = calculate_metrics(
        group["actual_wellbeing"],
        group["predicted_wellbeing"]
    )

    participant_results.append(
        {
            "Participant": participant_id,
            "N": len(group),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )


participant_table = pd.DataFrame(
    participant_results
)


# ============================================================
# 13. SAVE RESULT TABLES
# ============================================================

participant_table.to_csv(
    PARTICIPANT_OUTPUT,
    index=False
)

overall_table.to_csv(
    OVERALL_OUTPUT,
    index=False
)


# ============================================================
# 14. PARTICIPANT SUMMARY
# ============================================================

if len(participant_table) > 0:

    mean_participant_mae = (
        participant_table["MAE"].mean()
    )

    mean_participant_rmse = (
        participant_table["RMSE"].mean()
    )

    mean_participant_r2 = (
        participant_table["R2"].mean()
    )

else:

    mean_participant_mae = np.nan
    mean_participant_rmse = np.nan
    mean_participant_r2 = np.nan


# ============================================================
# 15. TEXT SUMMARY
# ============================================================

summary_lines = [
    "ML PREDICTION TEST SUMMARY",
    "==========================",
    "",
    "Prediction framework:",
    "Personalized Random Forest",
    "Training: all available past data for each participant",
    "Prediction features: previous 7 calendar days",
    "Target: same-day Wellbeing_Index",
    "",
    f"Prediction rows loaded: {len(predictions)}",
    f"Rows evaluated: {len(evaluation_df)}",
    f"Participants evaluated: {len(participant_table)}",
    "",
    "OVERALL RESULTS",
    "---------------",
    f"MAE:  {overall_mae:.4f}",
    f"RMSE: {overall_rmse:.4f}",
    f"R2:   {overall_r2:.4f}",
    "",
    "MEAN PARTICIPANT-LEVEL RESULTS",
    "-------------------------------",
    f"Mean MAE:  {mean_participant_mae:.4f}",
    f"Mean RMSE: {mean_participant_rmse:.4f}",
    f"Mean R2:   {mean_participant_r2:.4f}",
    "",
    "Metric interpretation:",
    "MAE = mean absolute prediction error.",
    "RMSE = root mean squared prediction error.",
    "R2 = proportion of variance explained by predictions.",
    "",
    "The test compares predicted Wellbeing_Index values",
    "with the actual Wellbeing_Index values for the same",
    "participant and target date."
]


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary_lines)
    )


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print()
print("-" * 80)
print("OVERALL RESULTS")
print("-" * 80)

print(
    f"MAE :  {overall_mae:.4f}"
)

print(
    f"RMSE:  {overall_rmse:.4f}"
)

print(
    f"R2  :  {overall_r2:.4f}"
)


print()
print("-" * 80)
print("PARTICIPANT-LEVEL RESULTS")
print("-" * 80)

if len(participant_table) > 0:

    print(
        participant_table.to_string(
            index=False
        )
    )

else:

    print(
        "No participants had enough observations "
        "for participant-level evaluation."
    )


print()
print("-" * 80)
print("SAVED")
print("-" * 80)

print(
    f"Participant table:\n{PARTICIPANT_OUTPUT}"
)

print(
    f"Overall table:\n{OVERALL_OUTPUT}"
)

print(
    f"Summary:\n{SUMMARY_FILE}"
)

print()
print("=" * 80)
print("PREDICTION TEST COMPLETE")
print("=" * 80)
