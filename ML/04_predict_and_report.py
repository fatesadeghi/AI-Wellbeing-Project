from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "ML" / "Results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FEATURES_FILE = (
    RESULTS_DIR
    / "behavior_7day_features.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "personal_model_predictions.csv"
)

WELLBEING_FILE = (
    RESULTS_DIR
    / "wellbeing_index.csv"
)

SHORT_OUTPUT = (
    RESULTS_DIR
    / "wellbeing_predictions_short.csv"
)

FULL_OUTPUT = (
    RESULTS_DIR
    / "wellbeing_predictions_full.csv"
)

REPORT_OUTPUT = (
    RESULTS_DIR
    / "wellbeing_research_report.txt"
)


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


def classify_wellbeing(z_score):

    if pd.isna(z_score):
        return "Insufficient_History"

    if z_score < LOW_THRESHOLD:
        return "Low"

    if z_score > HIGH_THRESHOLD:
        return "High"

    return "Moderate"


def classify_alert(status):

    if status == "Low":
        return "High"

    if status == "Moderate":
        return "Moderate"

    if status == "High":
        return "Low"

    return "Unavailable"


def get_behavior_summary(row):

    increasing = []
    decreasing = []
    stable = []

    for variable in BEHAVIOR_VARIABLES:

        column = f"{variable}_7d_slope"

        value = row.get(column, np.nan)

        if pd.isna(value):
            continue

        if value > 0:
            increasing.append(variable)

        elif value < 0:
            decreasing.append(variable)

        else:
            stable.append(variable)

    return (
        increasing,
        decreasing,
        stable,
    )


def get_top_behavior_changes(row):

    changes = []

    for variable in BEHAVIOR_VARIABLES:

        column = f"{variable}_7d_change"

        value = row.get(column, np.nan)

        if pd.isna(value):
            continue

        changes.append(
            (
                variable,
                value,
                abs(value),
            )
        )

    changes.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    return changes[:TOP_BEHAVIOR_CHANGES]


def main():

    print()
    print("=" * 80)
    print("PREDICT AND GENERATE WELLBEING REPORT")
    print("=" * 80)
    print()

    required_files = [
        FEATURES_FILE,
        PREDICTIONS_FILE,
        WELLBEING_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    features_df = pd.read_csv(
        FEATURES_FILE
    )

    predictions_df = pd.read_csv(
        PREDICTIONS_FILE
    )

    wellbeing_df = pd.read_csv(
        WELLBEING_FILE
    )

    features_df["target_date"] = pd.to_datetime(
        features_df["target_date"],
        errors="coerce",
    )

    predictions_df["target_date"] = pd.to_datetime(
        predictions_df["target_date"],
        errors="coerce",
    )

    wellbeing_df["date"] = pd.to_datetime(
        wellbeing_df["date"],
        errors="coerce",
    )

    predictions_df = predictions_df[
        predictions_df["prediction_status"]
        == "Predicted"
    ].copy()

    merged_df = predictions_df.merge(
        features_df,
        on=[
            "participant_id",
            "target_date",
        ],
        how="left",
    )

    merged_df = merged_df.sort_values(
        [
            "participant_id",
            "target_date",
        ]
    ).reset_index(drop=True)

    results = []
    report_sections = []

    for participant_id, participant_data in merged_df.groupby(
        "participant_id"
    ):

        participant_data = (
            participant_data
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        participant_wellbeing = wellbeing_df[
            wellbeing_df["participant_id"]
            == participant_id
        ].copy()

        for _, row in participant_data.iterrows():

            target_date = row["target_date"]

            historical = participant_wellbeing[
                participant_wellbeing["date"]
                < target_date
            ].copy()

            historical = historical[
                historical["Wellbeing_Index"]
                .notna()
            ]

            predicted_value = row[
                "predicted_wellbeing"
            ]

            baseline_mean = np.nan
            baseline_std = np.nan
            predicted_z = np.nan

            if len(historical) >= MIN_BASELINE_SAMPLES:

                baseline_mean = (
                    historical[
                        "Wellbeing_Index"
                    ].mean()
                )

                baseline_std = (
                    historical[
                        "Wellbeing_Index"
                    ].std(
                        ddof=1
                    )
                )

                if (
                    pd.notna(baseline_std)
                    and baseline_std > 0
                ):

                    predicted_z = (
                        predicted_value
                        - baseline_mean
                    ) / baseline_std

            status = classify_wellbeing(
                predicted_z
            )

            alert = classify_alert(
                status
            )

            increasing, decreasing, stable = (
                get_behavior_summary(row)
            )

            top_changes = (
                get_top_behavior_changes(row)
            )

            results.append(
                {
                    "participant_id":
                        participant_id,
                    "target_date":
                        target_date,
                    "predicted_wellbeing":
                        predicted_value,
                    "historical_baseline_mean":
                        baseline_mean,
                    "historical_baseline_sd":
                        baseline_std,
                    "predicted_wellbeing_z":
                        predicted_z,
                    "wellbeing_status":
                        status,
                    "alert_level":
                        alert,
                    "training_samples":
                        row["training_samples"],
                    "prediction_status":
                        row["prediction_status"],
                }
            )

            change_text = []

            for variable, value, _ in top_changes:

                direction = (
                    "increase"
                    if value > 0
                    else "decrease"
                )

                change_text.append(
                    f"{variable}: "
                    f"{direction} "
                    f"({value:.3f})"
                )

            report_sections.append(
                {
                    "participant_id":
                        participant_id,
                    "target_date":
                        target_date,
                    "predicted_wellbeing":
                        predicted_value,
                    "predicted_z":
                        predicted_z,
                    "status":
                        status,
                    "alert":
                        alert,
                    "increasing":
                        increasing,
                    "decreasing":
                        decreasing,
                    "stable":
                        stable,
                    "changes":
                        change_text,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        [
            "participant_id",
            "target_date",
        ]
    ).reset_index(drop=True)

    short_df = results_df[
        [
            "participant_id",
            "target_date",
            "predicted_wellbeing",
            "predicted_wellbeing_z",
            "wellbeing_status",
            "alert_level",
        ]
    ].copy()

    full_df = results_df.copy()

    short_df.to_csv(
        SHORT_OUTPUT,
        index=False,
    )

    full_df.to_csv(
        FULL_OUTPUT,
        index=False,
    )

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as report_file:

        report_file.write(
            "PERSONALIZED WELLBEING PREDICTION REPORT\n"
        )

        report_file.write(
            "=" * 80
            + "\n\n"
        )

        report_file.write(
            "Wellbeing status thresholds:\n"
        )

        report_file.write(
            "Z < -1   = Low\n"
        )

        report_file.write(
            "-1 to +1 = Moderate\n"
        )

        report_file.write(
            "Z > +1   = High\n\n"
        )

        report_file.write(
            "Alert levels:\n"
        )

        report_file.write(
            "Low wellbeing      = High alert\n"
        )

        report_file.write(
            "Moderate wellbeing = Moderate alert\n"
        )

        report_file.write(
            "High wellbeing     = Low alert\n\n"
        )

        report_file.write(
            "=" * 80
            + "\n\n"
        )

        for section in report_sections:

            report_file.write(
                f"Participant: "
                f"{section['participant_id']}\n"
            )

            report_file.write(
                f"Target date: "
                f"{section['target_date'].date()}\n"
            )

            report_file.write(
                f"Predicted wellbeing: "
                f"{section['predicted_wellbeing']:.4f}\n"
            )

            if pd.notna(
                section["predicted_z"]
            ):

                report_file.write(
                    f"Wellbeing Z-score: "
                    f"{section['predicted_z']:.4f}\n"
                )

            else:

                report_file.write(
                    "Wellbeing Z-score: "
                    "Insufficient history\n"
                )

            report_file.write(
                f"Status: "
                f"{section['status']}\n"
            )

            report_file.write(
                f"Alert level: "
                f"{section['alert']}\n"
            )

            if section["increasing"]:

                report_file.write(
                    "Increasing behaviors: "
                    + ", ".join(
                        section["increasing"]
                    )
                    + "\n"
                )

            if section["decreasing"]:

                report_file.write(
                    "Decreasing behaviors: "
                    + ", ".join(
                        section["decreasing"]
                    )
                    + "\n"
                )

            if section["stable"]:

                report_file.write(
                    "Stable behaviors: "
                    + ", ".join(
                        section["stable"]
                    )
                    + "\n"
                )

            if section["changes"]:

                report_file.write(
                    "Top behavioral changes:\n"
                )

                for change in section["changes"]:

                    report_file.write(
                        f"  - {change}\n"
                    )

            report_file.write("\n")
            report_file.write(
                "-" * 80
                + "\n\n"
            )

    print(
        f"Predictions processed: "
        f"{len(results_df)}"
    )

    print(
        f"Saved:\n{SHORT_OUTPUT}"
    )

    print(
        f"Saved:\n{FULL_OUTPUT}"
    )

    print(
        f"Saved:\n{REPORT_OUTPUT}"
    )

    print()


if __name__ == "__main__":
    main()
