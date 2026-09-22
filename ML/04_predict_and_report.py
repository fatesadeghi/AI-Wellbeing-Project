"""
04_predict_and_report.py

Purpose
-------
Create the final personalized wellbeing prediction and report.

Input
-----
- behavior_7day_features.csv
- personal_model_predictions.csv
- wellbeing_index.csv

Output
------
For every participant and target date:

    - Predicted Wellbeing
    - Wellbeing Status
    - Alert Level
    - 7-day behavioral trends
    - Important behavioral changes
    - Interpretation

Wellbeing status
----------------
The predicted Wellbeing Index is standardized against the
participant's historical Wellbeing Index values available
BEFORE the target date.

    Z < -1       -> Low
    -1 <= Z <= 1 -> Moderate
    Z > 1        -> High

Alert level
-----------
The alert level is based on wellbeing status:

    High wellbeing     -> Low alert
    Moderate wellbeing -> Moderate alert
    Low wellbeing      -> High alert

Important
---------
No future wellbeing observations are used to determine the
status of a prediction.

Behavioral features are based only on the previous 7 days.
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DIR = PROJECT_ROOT / "ML"
RESULTS_DIR = ML_DIR / "Results"

FEATURE_FILE = (
    RESULTS_DIR / "behavior_7day_features.csv"
)

PREDICTION_FILE = (
    RESULTS_DIR / "personal_model_predictions.csv"
)

WELLBEING_FILE = (
    RESULTS_DIR / "wellbeing_index.csv"
)

SHORT_OUTPUT_FILE = (
    RESULTS_DIR / "wellbeing_predictions_short.csv"
)

FULL_OUTPUT_FILE = (
    RESULTS_DIR / "wellbeing_predictions_full.csv"
)

TEXT_REPORT_FILE = (
    RESULTS_DIR / "wellbeing_research_report.txt"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

LOW_THRESHOLD = -1.0
HIGH_THRESHOLD = 1.0

MIN_BASELINE_SAMPLES = 2

TOP_BEHAVIOR_CHANGES = 5

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
    "Sleep_Score",
]


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

def validate_input_files() -> None:
    """
    Make sure all required input files exist.
    """

    required_files = [
        FEATURE_FILE,
        PREDICTION_FILE,
        WELLBEING_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )


def validate_columns(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    wellbeing: pd.DataFrame,
) -> None:
    """
    Validate all required columns.
    """

    feature_columns = [
        "participant_id",
        "target_date",
    ]

    for variable in BEHAVIOR_VARIABLES:

        feature_columns.extend(
            [
                f"{variable}_7d_slope",
                f"{variable}_7d_change",
            ]
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in features.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing feature columns:\n"
            + "\n".join(missing_features)
        )

    prediction_columns = [
        "participant_id",
        "target_date",
        "actual_wellbeing",
        "predicted_wellbeing",
        "prediction_status",
    ]

    missing_predictions = [
        column
        for column in prediction_columns
        if column not in predictions.columns
    ]

    if missing_predictions:

        raise ValueError(
            "Missing prediction columns:\n"
            + "\n".join(missing_predictions)
        )

    wellbeing_columns = [
        "participant_id",
        "date",
        "Wellbeing_Index",
    ]

    missing_wellbeing = [
        column
        for column in wellbeing_columns
        if column not in wellbeing.columns
    ]

    if missing_wellbeing:

        raise ValueError(
            "Missing wellbeing columns:\n"
            + "\n".join(missing_wellbeing)
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load all required datasets.
    """

    validate_input_files()

    features = pd.read_csv(
        FEATURE_FILE
    )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    wellbeing = pd.read_csv(
        WELLBEING_FILE
    )

    validate_columns(
        features,
        predictions,
        wellbeing,
    )

    return (
        features,
        predictions,
        wellbeing,
    )


# ============================================================
# 5. PREPARE DATA
# ============================================================

def prepare_data(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    wellbeing: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Convert dates and participant IDs into consistent formats.
    """

    features = features.copy()
    predictions = predictions.copy()
    wellbeing = wellbeing.copy()

    features["target_date"] = pd.to_datetime(
        features["target_date"],
        errors="coerce",
    )

    predictions["target_date"] = pd.to_datetime(
        predictions["target_date"],
        errors="coerce",
    )

    wellbeing["date"] = pd.to_datetime(
        wellbeing["date"],
        errors="coerce",
    )

    features["participant_id"] = (
        features["participant_id"]
        .astype(str)
    )

    predictions["participant_id"] = (
        predictions["participant_id"]
        .astype(str)
    )

    wellbeing["participant_id"] = (
        wellbeing["participant_id"]
        .astype(str)
    )

    predictions["actual_wellbeing"] = pd.to_numeric(
        predictions["actual_wellbeing"],
        errors="coerce",
    )

    predictions["predicted_wellbeing"] = pd.to_numeric(
        predictions["predicted_wellbeing"],
        errors="coerce",
    )

    wellbeing["Wellbeing_Index"] = pd.to_numeric(
        wellbeing["Wellbeing_Index"],
        errors="coerce",
    )

    return (
        features,
        predictions,
        wellbeing,
    )


# ============================================================
# 6. CALCULATE HISTORICAL BASELINE
# ============================================================

def calculate_historical_baseline(
    wellbeing_df: pd.DataFrame,
    participant_id: str,
    target_date: pd.Timestamp,
) -> tuple[float, float, int]:
    """
    Calculate the participant's historical Wellbeing Index
    mean and standard deviation using only observations
    BEFORE the target date.
    """

    historical = wellbeing_df[
        (wellbeing_df["participant_id"] == participant_id)
        & (wellbeing_df["date"] < target_date)
    ].copy()

    historical = historical.dropna(
        subset=["Wellbeing_Index"]
    )

    values = historical[
        "Wellbeing_Index"
    ]

    sample_count = len(values)

    if sample_count < MIN_BASELINE_SAMPLES:

        return (
            np.nan,
            np.nan,
            sample_count,
        )

    mean_value = values.mean()

    std_value = values.std(
        ddof=1
    )

    if pd.isna(std_value) or std_value == 0:

        return (
            float(mean_value),
            np.nan,
            sample_count,
        )

    return (
        float(mean_value),
        float(std_value),
        sample_count,
    )


# ============================================================
# 7. CLASSIFY WELLBEING
# ============================================================

def classify_wellbeing(
    wellbeing_z: float,
) -> str:
    """
    Convert Wellbeing Z-score into Low / Moderate / High.
    """

    if pd.isna(wellbeing_z):

        return "Unavailable"

    if wellbeing_z < LOW_THRESHOLD:

        return "Low"

    if wellbeing_z > HIGH_THRESHOLD:

        return "High"

    return "Moderate"


# ============================================================
# 8. DETERMINE ALERT LEVEL
# ============================================================

def determine_alert_level(
    wellbeing_status: str,
) -> str:
    """
    Convert wellbeing status into alert level.
    """

    if wellbeing_status == "Low":

        return "High"

    if wellbeing_status == "High":

        return "Low"

    if wellbeing_status == "Moderate":

        return "Moderate"

    return "Unavailable"


# ============================================================
# 9. BEHAVIORAL TREND DESCRIPTION
# ============================================================

def describe_slope(
    slope: float,
) -> str:
    """
    Describe the direction of a behavioral trend.
    """

    if pd.isna(slope):

        return "Unavailable"

    if slope > 0:

        return "Increasing"

    if slope < 0:

        return "Decreasing"

    return "Stable"


# ============================================================
# 10. BUILD BEHAVIORAL TREND SUMMARY
# ============================================================

def build_behavior_trend_summary(
    row: pd.Series,
) -> str:
    """
    Create a compact textual summary of 7-day trends.
    """

    trends = []

    for variable in BEHAVIOR_VARIABLES:

        slope_column = (
            f"{variable}_7d_slope"
        )

        slope = row.get(
            slope_column,
            np.nan,
        )

        direction = describe_slope(
            slope
        )

        if direction != "Unavailable":

            trends.append(
                f"{variable}: {direction}"
            )

    if not trends:

        return "No behavioral trend information available."

    return "; ".join(trends)


# ============================================================
# 11. FIND IMPORTANT BEHAVIORAL CHANGES
# ============================================================

def find_important_behavior_changes(
    row: pd.Series,
) -> str:
    """
    Identify the largest absolute 7-day behavioral changes.

    Because behavioral variables have different units, this
    section reports the raw change together with the variable
    name rather than comparing the numerical magnitudes as
    if they were on the same scale.
    """

    changes = []

    for variable in BEHAVIOR_VARIABLES:

        change_column = (
            f"{variable}_7d_change"
        )

        change = row.get(
            change_column,
            np.nan,
        )

        if pd.isna(change):

            continue

        changes.append(
            {
                "variable": variable,
                "change": float(change),
                "absolute_change": abs(
                    float(change)
                ),
            }
        )

    if not changes:

        return "No measurable behavioral changes available."

    changes = sorted(
        changes,
        key=lambda item: item["absolute_change"],
        reverse=True,
    )

    top_changes = changes[
        :TOP_BEHAVIOR_CHANGES
    ]

    descriptions = []

    for item in top_changes:

        variable = item["variable"]
        change = item["change"]

        if change > 0:

            direction = "increased"

        elif change < 0:

            direction = "decreased"

        else:

            direction = "was stable"

        descriptions.append(
            f"{variable} {direction} "
            f"(change={change:.3f})"
        )

    return "; ".join(
        descriptions
    )


# ============================================================
# 12. INTERPRETATION
# ============================================================

def build_interpretation(
    wellbeing_status: str,
    alert_level: str,
) -> str:
    """
    Create a concise interpretation of the predicted status.
    """

    if wellbeing_status == "High":

        return (
            "Predicted wellbeing is above the participant's "
            "historical reference range."
        )

    if wellbeing_status == "Moderate":

        return (
            "Predicted wellbeing is within the participant's "
            "historical reference range."
        )

    if wellbeing_status == "Low":

        return (
            "Predicted wellbeing is below the participant's "
            "historical reference range and may warrant "
            "closer monitoring."
        )

    return (
        "There is not enough historical wellbeing information "
        "to determine the predicted status."
    )


# ============================================================
# 13. BUILD FINAL REPORT
# ============================================================

def build_reports(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    wellbeing: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build short and full prediction reports.
    """

    # --------------------------------------------------------
    # Keep only actual predictions.
    # --------------------------------------------------------

    predictions = predictions[
        predictions[
            "prediction_status"
        ] == "Predicted"
    ].copy()

    predictions = predictions.dropna(
        subset=[
            "predicted_wellbeing",
            "target_date",
        ]
    )

    # --------------------------------------------------------
    # Merge behavioral features.
    # --------------------------------------------------------

    report = predictions.merge(
        features,
        on=[
            "participant_id",
            "target_date",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Create report rows.
    # --------------------------------------------------------

    full_rows = []

    for _, row in report.iterrows():

        participant_id = row[
            "participant_id"
        ]

        target_date = row[
            "target_date"
        ]

        predicted_wellbeing = float(
            row[
                "predicted_wellbeing"
            ]
        )

        # ----------------------------------------------------
        # Historical baseline
        # ----------------------------------------------------

        baseline_mean, baseline_sd, baseline_n = (
            calculate_historical_baseline(
                wellbeing_df=wellbeing,
                participant_id=participant_id,
                target_date=target_date,
            )
        )

        # ----------------------------------------------------
        # Wellbeing Z-score
        # ----------------------------------------------------

        if (
            pd.isna(baseline_sd)
            or baseline_sd == 0
        ):

            wellbeing_z = np.nan

        else:

            wellbeing_z = (
                predicted_wellbeing
                - baseline_mean
            ) / baseline_sd

        # ----------------------------------------------------
        # Status and alert
        # ----------------------------------------------------

        wellbeing_status = (
            classify_wellbeing(
                wellbeing_z
            )
        )

        alert_level = (
            determine_alert_level(
                wellbeing_status
            )
        )

        # ----------------------------------------------------
        # Behavioral information
        # ----------------------------------------------------

        trend_summary = (
            build_behavior_trend_summary(
                row
            )
        )

        important_changes = (
            find_important_behavior_changes(
                row
            )
        )

        interpretation = (
            build_interpretation(
                wellbeing_status,
                alert_level,
            )
        )

        # ----------------------------------------------------
        # Full report row
        # ----------------------------------------------------

        full_rows.append(
            {
                "participant_id": participant_id,
                "target_date": target_date,
                "predicted_wellbeing": (
                    predicted_wellbeing
                ),
                "historical_baseline_mean": (
                    baseline_mean
                ),
                "historical_baseline_sd": (
                    baseline_sd
                ),
                "historical_baseline_n": (
                    baseline_n
                ),
                "predicted_wellbeing_z": (
                    wellbeing_z
                ),
                "wellbeing_status": (
                    wellbeing_status
                ),
                "alert_level": (
                    alert_level
                ),
                "behavioral_7day_trends": (
                    trend_summary
                ),
                "important_behavioral_changes": (
                    important_changes
                ),
                "interpretation": (
                    interpretation
                ),
            }
        )

    full_df = pd.DataFrame(
        full_rows
    )

    # --------------------------------------------------------
    # Short output
    # --------------------------------------------------------

    short_df = full_df[
        [
            "participant_id",
            "target_date",
            "predicted_wellbeing",
            "wellbeing_status",
            "alert_level",
        ]
    ].copy()

    return (
        short_df,
        full_df,
    )


# ============================================================
# 14. CREATE TEXT RESEARCH REPORT
# ============================================================

def create_text_report(
    full_df: pd.DataFrame,
) -> None:
    """
    Create a human-readable research report.
    """

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "PERSONALIZED WELLBEING PREDICTION REPORT"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Status thresholds:"
    )

    lines.append(
        "  Z < -1       -> Low"
    )

    lines.append(
        "  -1 <= Z <= 1 -> Moderate"
    )

    lines.append(
        "  Z > 1        -> High"
    )

    lines.append("")

    lines.append(
        "Alert levels:"
    )

    lines.append(
        "  High wellbeing     -> Low alert"
    )

    lines.append(
        "  Moderate wellbeing -> Moderate alert"
    )

    lines.append(
        "  Low wellbeing      -> High alert"
    )

    lines.append("")

    lines.append(
        "Note: Baseline statistics use only wellbeing "
        "observations available before each target date."
    )

    lines.append("")

    for _, row in full_df.iterrows():

        lines.append(
            "-" * 80
        )

        lines.append(
            f"Participant: "
            f"{row['participant_id']}"
        )

        lines.append(
            f"Target Date: "
            f"{row['target_date'].date()}"
        )

        lines.append(
            f"Predicted Wellbeing: "
            f"{row['predicted_wellbeing']:.3f}"
        )

        lines.append(
            f"Wellbeing Status: "
            f"{row['wellbeing_status']}"
        )

        lines.append(
            f"Alert Level: "
            f"{row['alert_level']}"
        )

        if not pd.isna(
            row["predicted_wellbeing_z"]
        ):

            lines.append(
                f"Predicted Wellbeing Z-score: "
                f"{row['predicted_wellbeing_z']:.3f}"
            )

        lines.append("")

        lines.append(
            "7-Day Behavioral Trends:"
        )

        lines.append(
            str(
                row[
                    "behavioral_7day_trends"
                ]
            )
        )

        lines.append("")

        lines.append(
            "Important Behavioral Changes:"
        )

        lines.append(
            str(
                row[
                    "important_behavioral_changes"
                ]
            )
        )

        lines.append("")

        lines.append(
            "Interpretation:"
        )

        lines.append(
            str(
                row[
                    "interpretation"
                ]
            )
        )

        lines.append("")

    TEXT_REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# 15. MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("PREDICT AND REPORT PERSONALIZED WELLBEING")
    print("=" * 80)

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    features, predictions, wellbeing = (
        load_data()
    )

    print(
        f"Feature rows: {len(features)}"
    )

    print(
        f"Prediction rows: {len(predictions)}"
    )

    print(
        f"Wellbeing rows: {len(wellbeing)}"
    )

    print()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    features, predictions, wellbeing = (
        prepare_data(
            features,
            predictions,
            wellbeing,
        )
    )

    # --------------------------------------------------------
    # Build reports
    # --------------------------------------------------------

    short_df, full_df = (
        build_reports(
            features,
            predictions,
            wellbeing,
        )
    )

    # --------------------------------------------------------
    # Save CSV outputs
    # --------------------------------------------------------

    short_df.to_csv(
        SHORT_OUTPUT_FILE,
        index=False,
    )

    full_df.to_csv(
        FULL_OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    create_text_report(
        full_df
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)

    print(
        f"Final predictions: "
        f"{len(full_df)}"
    )

    if not full_df.empty:

        print()

        print(
            "Wellbeing status counts:"
        )

        print(
            full_df[
                "wellbeing_status"
            ]
            .value_counts()
            .to_string()
        )

        print()

        print(
            "Alert level counts:"
        )

        print(
            full_df[
                "alert_level"
            ]
            .value_counts()
            .to_string()
        )

    print()
    print(
        f"Short output:\n"
        f"  {SHORT_OUTPUT_FILE}"
    )

    print()
    print(
        f"Full output:\n"
        f"  {FULL_OUTPUT_FILE}"
    )

    print()
    print(
        f"Research report:\n"
        f"  {TEXT_REPORT_FILE}"
    )


# ============================================================
# 16. RUN
# ============================================================

if __name__ == "__main__":
    main()
