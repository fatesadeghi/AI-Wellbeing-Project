from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "pmdata"
ML_DIR = PROJECT_ROOT / "ML"
RESULTS_DIR = ML_DIR / "Results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


def find_participant_files():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found:\n{DATA_DIR}"
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No participant CSV files found in:\n{DATA_DIR}"
        )

    return files


def calculate_slope(dates, values):
    valid = pd.notna(values)

    dates = dates[valid]
    values = values[valid]

    if len(values) < 2:
        return np.nan

    day_numbers = (
        dates - dates.min()
    ).dt.total_seconds() / 86400.0

    if day_numbers.nunique() < 2:
        return np.nan

    return np.polyfit(
        day_numbers.to_numpy(),
        values.to_numpy(),
        1,
    )[0]


def calculate_change(values):
    values = values.dropna()

    if len(values) < 2:
        return np.nan

    return values.iloc[-1] - values.iloc[0]


def build_features_for_target_date(
    participant_df,
    target_date,
):
    history_start = (
        target_date
        - pd.Timedelta(days=7)
    )

    history_end = (
        target_date
        - pd.Timedelta(days=1)
    )

    history = participant_df[
        (participant_df["date"] >= history_start)
        & (participant_df["date"] <= history_end)
    ].copy()

    features = {}

    for variable in BEHAVIOR_VARIABLES:

        values = pd.to_numeric(
            history[variable],
            errors="coerce",
        )

        features[
            f"{variable}_7d_slope"
        ] = calculate_slope(
            history["date"],
            values,
        )

        features[
            f"{variable}_7d_change"
        ] = calculate_change(
            values
        )

    return features


def process_participant(file_path):

    participant_id = file_path.stem

    df = pd.read_csv(file_path)

    date_candidates = [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "timestamp",
        "Timestamp",
    ]

    date_column = None

    for candidate in date_candidates:
        if candidate in df.columns:
            date_column = candidate
            break

    if date_column is None:
        raise ValueError(
            f"No date column found in {file_path.name}"
        )

    missing_columns = [
        variable
        for variable in BEHAVIOR_VARIABLES
        if variable not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing behavioral columns in "
            f"{file_path.name}: {missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    ).copy()

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    for variable in BEHAVIOR_VARIABLES:
        df[variable] = pd.to_numeric(
            df[variable],
            errors="coerce",
        )

    if df.empty:
        return [], {
            "participant_id": participant_id,
            "source_file": file_path.name,
            "target_dates": 0,
            "features_created": 0,
        }

    first_date = df["date"].min()

    target_dates = pd.date_range(
        start=first_date + pd.Timedelta(days=7),
        end=df["date"].max(),
        freq="D",
    )

    participant_results = []

    for target_date in target_dates:

        features = build_features_for_target_date(
            df,
            target_date,
        )

        row = {
            "participant_id": participant_id,
            "target_date": target_date,
        }

        row.update(features)

        participant_results.append(row)

    processing_info = {
        "participant_id": participant_id,
        "source_file": file_path.name,
        "target_dates": len(target_dates),
        "features_created": len(
            participant_results
        ),
    }

    return participant_results, processing_info


def main():

    print()
    print("=" * 80)
    print("BUILD 7-DAY BEHAVIORAL FEATURES")
    print("=" * 80)
    print()

    print(
        f"Data directory:\n{DATA_DIR}"
    )

    print(
        f"Results directory:\n{RESULTS_DIR}"
    )

    print()

    participant_files = find_participant_files()

    print(
        f"Participant files found: "
        f"{len(participant_files)}"
    )

    print()

    all_features = []
    processing_log = []

    for file_path in participant_files:

        participant_id = file_path.stem

        print(
            f"Processing: {participant_id}"
        )

        try:

            features, info = process_participant(
                file_path
            )

            all_features.extend(
                features
            )

            processing_log.append(
                info
            )

            print(
                f"  Target dates: "
                f"{info['target_dates']}"
            )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

            processing_log.append(
                {
                    "participant_id": participant_id,
                    "source_file": file_path.name,
                    "target_dates": 0,
                    "features_created": 0,
                    "error": str(error),
                }
            )

    if not all_features:
        raise RuntimeError(
            "No 7-day behavioral features were created."
        )

    features_df = pd.DataFrame(
        all_features
    )

    features_df = features_df.sort_values(
        [
            "participant_id",
            "target_date",
        ]
    ).reset_index(
        drop=True
    )

    processing_log_df = pd.DataFrame(
        processing_log
    )

    features_output = (
        RESULTS_DIR
        / "behavior_7day_features.csv"
    )

    log_output = (
        RESULTS_DIR
        / "behavior_7day_feature_processing_log.csv"
    )

    features_df.to_csv(
        features_output,
        index=False,
    )

    processing_log_df.to_csv(
        log_output,
        index=False,
    )

    print()
    print("=" * 80)
    print("7-DAY FEATURES COMPLETED")
    print("=" * 80)
    print()

    print(
        f"Participants processed: "
        f"{len(processing_log_df)}"
    )

    print(
        f"Feature rows created: "
        f"{len(features_df)}"
    )

    print(
        f"Feature columns: "
        f"{len(features_df.columns)}"
    )

    print()

    print(
        f"Saved:\n{features_output}"
    )

    print(
        f"Saved:\n{log_output}"
    )

    print()


if __name__ == "__main__":
    main()
