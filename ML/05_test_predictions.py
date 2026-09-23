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

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "prediction_test_report.csv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "prediction_test_summary.txt"
)


# ============================================================
# 2. METRIC FUNCTIONS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    mae = np.mean(np.abs(errors))

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
# 3. LOAD PREDICTIONS
# ============================================================

print("=" * 80)
print("ML PREDICTION TEST")
print("=" * 80)

if not os.path.exists(PREDICTIONS_FILE):
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_FILE}"
    )

df = pd.read_csv(PREDICTIONS_FILE)

print(f"Prediction rows loaded: {len(df)}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "participant",
    "target_date",
    "predicted_wellbeing",
    "actual_wellbeing"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 5. CLEAN DATA
# ============================================================

df["target_date"] = pd.to_datetime(
    df["target_date"],
    errors="coerce"
)

df["predicted_wellbeing"] = pd.to_numeric(
    df["predicted_wellbeing"],
    errors="coerce"
)

df["actual_wellbeing"] = pd.to_numeric(
    df["actual_wellbeing"],
    errors="coerce"
)


# Only rows where both prediction and actual value exist
evaluation_df = df.dropna(
    subset=[
        "predicted_wellbeing",
        "actual_wellbeing"
    ]
).copy()


print(
    f"Rows with both prediction and actual wellbeing: "
    f"{len(evaluation_df)}"
)


# ============================================================
# 6. OVERALL TEST
# ============================================================

if len(evaluation_df) == 0:
    raise ValueError(
        "No rows contain both predicted and actual wellbeing."
    )

overall_mae, overall_rmse, overall_r2 = calculate_metrics(
    evaluation_df["actual_wellbeing"],
    evaluation_df["predicted_wellbeing"]
)


# ============================================================
# 7. PARTICIPANT-LEVEL TEST
# ============================================================

participant_results = []

for participant, group in evaluation_df.groupby(
    "participant"
):

    if len(group) < 2:
        continue

    mae, rmse, r2 = calculate_metrics(
        group["actual_wellbeing"],
        group["predicted_wellbeing"]
    )

    participant_results.append({
        "Participant": participant,
        "N": len(group),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


participant_results_df = pd.DataFrame(
    participant_results
)


# ============================================================
# 8. SAVE PARTICIPANT TABLE
# ============================================================

participant_results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 9. SUMMARY REPORT
# ============================================================

mean_participant_mae = (
    participant_results_df["MAE"].mean()
    if len(participant_results_df) > 0
    else np.nan
)

mean_participant_rmse = (
    participant_results_df["RMSE"].mean()
    if len(participant_results_df) > 0
    else np.nan
)

mean_participant_r2 = (
    participant_results_df["R2"].mean()
    if len(participant_results_df) > 0
    else np.nan
)


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
    f"Prediction rows loaded: {len(df)}",
    f"Rows evaluated: {len(evaluation_df)}",
    f"Participants evaluated: {len(participant_results_df)}",
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
    "Interpretation:",
    "MAE and RMSE measure prediction error.",
    "Lower MAE and RMSE indicate smaller prediction errors.",
    "R2 describes how much variation in the actual wellbeing",
    "values is explained by the predictions.",
]


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary_lines)
    )


# ============================================================
# 10. PRINT RESULTS
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

if len(participant_results_df) > 0:
    print(
        participant_results_df.to_string(
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
    f"Participant report:\n{OUTPUT_FILE}"
)

print(
    f"Summary report:\n{SUMMARY_FILE}"
)

print()
print("=" * 80)
print("PREDICTION TEST COMPLETE")
print("=" * 80)
