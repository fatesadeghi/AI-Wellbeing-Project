"""
02_build_7day_features.py

Purpose
-------
Build personalized 7-day behavioral features for the ML pipeline.

For each participant and each target date T:

    History window = the previous 7 calendar days
                     T-7, T-6, ..., T-1

For each behavioral variable, two features are calculated:

    1. 7-day slope
       Linear trend across the previous 7 days.

    2. Change magnitude
       Absolute change between the first and last observed
       values in the 7-day history.

The target day's behavioral value is NEVER used as a feature.

Behavioral variables
--------------------
Steps
Exercise_Count
Exercise_Duration
Exercise_Distance
Exercise_Calories
Exercise_Avg_HR
Sleep_Hours
Sleep_Duration_Score
Deep_Sleep_Minutes
Sleep_Restlessness
Sleep_Composition
Sleep_Revitalization
Sleep_Score

Important
---------
- Each participant is processed independently.
- No participant-level pooling is performed here.
- The feature window is strictly historical.
- Calendar gaps are preserved; missing days are not silently
  replaced or forward-filled.
- No wellbeing variable is used as a predictor.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "pmdata"
ML_DIR = PROJECT_ROOT / "ML"
RESULTS_DIR = ML_DIR / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
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
    "Sleep_Score",
]

WELLBEING_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
]

HISTORY_DAYS = 7

MIN_OBSERVATIONS_FOR_FEATURE = 2


# ============================================================
# 3. COLUMN HELPERS
# ============================================================

def find_date_column(df: pd.DataFrame) -> str:
    """
    Find the date column in a participant dataframe.
    """

    candidates = [
        "date",
        "Date",
        "DATE",
        "day",
        "Day",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "No date column found. "
        f"Expected one of: {candidates}"
    )


def find_participant_column(df: pd.DataFrame) -> str | None:
    """
    Find a participant ID column if present.
    """

    candidates = [
        "participant_id",
        "Participant_ID",
        "participant",
        "Participant",
        "id",
        "ID",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================
# 4. DATE PREPARATION
# ============================================================

def prepare_participant_data(
    df: pd.DataFrame,
    participant_id: str,
) -> tuple[pd.DataFrame, str]:
    """
    Prepare one participant's dataframe.

    Returns
    -------
    prepared dataframe
    date column name
    """

    date_column = find_date_column(df)

    df = df.copy()

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[date_column]
    )

    # --------------------------------------------------------
    # Check for duplicate dates
    # --------------------------------------------------------

    duplicate_dates = df[date_column].duplicated(
        keep=False
    )

    if duplicate_dates.any():

        duplicate_count = duplicate_dates.sum()

        raise ValueError(
            f"Participant {participant_id} has "
            f"{duplicate_count} rows with duplicate dates. "
            "Daily features require one row per date."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        date_column
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Convert behavioral variables to numeric
    # --------------------------------------------------------

    for variable in BEHAVIOR_VARIABLES:

        if variable in df.columns:
            df[variable] = pd.to_numeric(
                df[variable],
                errors="coerce",
            )

    return df, date_column


# ============================================================
# 5. 7-DAY FEATURE FUNCTIONS
# ============================================================

def calculate_slope(
    values: pd.Series,
) -> float:
    """
    Calculate the linear slope over the observed values.

    The x-axis is the actual calendar-day offset.

    Missing observations are excluded.

    At least two observations are required.
    """

    valid = values.dropna()

    if len(valid) < MIN_OBSERVATIONS_FOR_FEATURE:
        return np.nan

    # Convert dates to integer day offsets.
    x = (
        valid.index
        .to_series()
        .astype("int64")
        .to_numpy()
    )

    y = valid.to_numpy(dtype=float)

    if len(np.unique(x)) < 2:
        return np.nan

    slope = np.polyfit(
        x,
        y,
        1,
    )[0]

    return float(slope)


def calculate_change_magnitude(
    values: pd.Series,
) -> float:
    """
    Calculate the absolute change between the earliest
    and latest observed values in the 7-day window.

    Missing observations are ignored.

    At least two observations are required.
    """

    valid = values.dropna()

    if len(valid) < MIN_OBSERVATIONS_FOR_FEATURE:
        return np.nan

    first_value = float(valid.iloc[0])
    last_value = float(valid.iloc[-1])

    return abs(last_value - first_value)


# ============================================================
# 6. BUILD FEATURES FOR ONE TARGET DATE
# ============================================================

def build_features_for_target_date(
    participant_df: pd.DataFrame,
    date_column: str,
    target_date: pd.Timestamp,
    participant_id: str,
) -> dict:
    """
    Build all 7-day behavioral features for one target date.

    Only the previous 7 calendar days are considered.
    The target date itself is excluded.
    """

    history_start = (
        target_date
        - pd.Timedelta(days=HISTORY_DAYS)
    )

    history_end = (
        target_date
        - pd.Timedelta(days=1)
    )

    history = participant_df[
        (participant_df[date_column] >= history_start)
        & (participant_df[date_column] <= history_end)
    ].copy()

    # --------------------------------------------------------
    # Ensure chronological order
    # --------------------------------------------------------

    history = history.sort_values(
        date_column
    )

    result = {
        "participant_id": participant_id,
        "target_date": target_date,
        "history_start": history_start,
        "history_end": history_end,
        "history_days_available": len(history),
    }

    # --------------------------------------------------------
    # Calculate behavioral features
    # --------------------------------------------------------

    for variable in BEHAVIOR_VARIABLES:

        slope_column = (
            f"{variable}_7d_slope"
        )

        change_column = (
            f"{variable}_7d_change"
        )

        if variable not in history.columns:

            result[slope_column] = np.nan
            result[change_column] = np.nan

            continue

        series = history[
            variable
        ].dropna()

        if len(series) < MIN_OBSERVATIONS_FOR_FEATURE:

            result[slope_column] = np.nan
            result[change_column] = np.nan

            continue

        # ----------------------------------------------------
        # Slope using actual calendar positions
        # ----------------------------------------------------

        dated_series = (
            history[
                [date_column, variable]
            ]
            .dropna()
            .sort_values(date_column)
        )

        if len(dated_series) >= MIN_OBSERVATIONS_FOR_FEATURE:

            x = (
                (
                    dated_series[date_column]
                    - history_start
                )
                .dt.total_seconds()
                / (24 * 60 * 60)
            ).to_numpy(dtype=float)

            y = dated_series[
                variable
            ].to_numpy(dtype=float)

            if len(np.unique(x)) >= 2:

                result[slope_column] = float(
                    np.polyfit(
                        x,
                        y,
                        1,
                    )[0]
                )

            else:

                result[slope_column] = np.nan

        else:

            result[slope_column] = np.nan

        # ----------------------------------------------------
        # Magnitude of change
        # ----------------------------------------------------

        result[change_column] = (
            float(dated_series[variable].iloc[-1])
            - float(dated_series[variable].iloc[0])
        )

    return result


# ============================================================
# 7. BUILD FEATURES FOR ONE PARTICIPANT
# ============================================================

def build_participant_features(
    df: pd.DataFrame,
    participant_id: str,
) -> pd.DataFrame:
    """
    Build historical 7-day features for one participant.

    Features are created for every observed target date
    starting from the 8th calendar day of the participant's
    observed timeline.
    """

    df, date_column = prepare_participant_data(
        df,
        participant_id,
    )

    if df.empty:
        return pd.DataFrame()

    dates = (
        df[date_column]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    feature_rows = []

    # --------------------------------------------------------
    # Need at least 8 observed calendar dates so that there
    # can be a target date after the initial 7-day history.
    # --------------------------------------------------------

    for target_date in dates:

        earliest_allowed_target = (
            dates[0]
            + pd.Timedelta(days=HISTORY_DAYS)
        )

        if target_date < earliest_allowed_target:
            continue

        row = build_features_for_target_date(
            participant_df=df,
            date_column=date_column,
            target_date=target_date,
            participant_id=participant_id,
        )

        feature_rows.append(row)

    if not feature_rows:
        return pd.DataFrame()

    result = pd.DataFrame(
        feature_rows
    )

    return result


# ============================================================
# 8. FIND PARTICIPANT FILES
# ============================================================

def find_participant_files() -> list[Path]:
    """
    Find participant CSV files.
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found:\n{DATA_DIR}"
        )

    files = sorted(
        DATA_DIR.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No participant CSV files found in:\n{DATA_DIR}"
        )

    return files


# ============================================================
# 9. PROCESS ALL PARTICIPANTS
# ============================================================

def main() -> None:

    print("=" * 80)
    print("BUILD PERSONALIZED 7-DAY BEHAVIORAL FEATURES")
    print("=" * 80)

    participant_files = (
        find_participant_files()
    )

    print(
        f"Participant files found: "
        f"{len(participant_files)}"
    )

    print(
        f"History window: "
        f"{HISTORY_DAYS} days"
    )

    print()

    all_features = []
    processing_errors = []

    for file_path in participant_files:

        participant_id = file_path.stem

        print("-" * 80)
        print(
            f"Participant: "
            f"{participant_id}"
        )

        try:

            df = pd.read_csv(
                file_path
            )

            participant_column = (
                find_participant_column(df)
            )

            if participant_column is not None:

                unique_ids = (
                    df[participant_column]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                if len(unique_ids) == 1:

                    participant_id = (
                        unique_ids[0]
                    )

            features = (
                build_participant_features(
                    df=df,
                    participant_id=participant_id,
                )
            )

            print(
                f"  Feature rows: "
                f"{len(features)}"
            )

            if not features.empty:

                all_features.append(
                    features
                )

        except Exception as error:

            processing_errors.append(
                {
                    "participant_id": participant_id,
                    "file": file_path.name,
                    "error": str(error),
                }
            )

            print(
                f"  ERROR: {error}"
            )

    # ========================================================
    # 10. COMBINE RESULTS
    # ========================================================

    if not all_features:

        raise RuntimeError(
            "No participant features were created."
        )

    final_df = pd.concat(
        all_features,
        ignore_index=True,
    )

    final_df = final_df.sort_values(
        [
            "participant_id",
            "target_date",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 11. SAVE FEATURE DATA
    # ========================================================

    output_file = (
        RESULTS_DIR
        / "behavior_7day_features.csv"
    )

    final_df.to_csv(
        output_file,
        index=False,
    )

    # ========================================================
    # 12. SAVE PROCESSING LOG
    # ========================================================

    log_file = (
        RESULTS_DIR
        / "behavior_7day_feature_processing_log.csv"
    )

    if processing_errors:

        log_df = pd.DataFrame(
            processing_errors
        )

    else:

        log_df = pd.DataFrame(
            columns=[
                "participant_id",
                "file",
                "error",
            ]
        )

    log_df.to_csv(
        log_file,
        index=False,
    )

    # ========================================================
    # 13. FEATURE SUMMARY
    # ========================================================

    feature_columns = [
        column
        for column in final_df.columns
        if column.endswith("_7d_slope")
        or column.endswith("_7d_change")
    ]

    print()
    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)

    print(
        f"Participants processed: "
        f"{len(all_features)}"
    )

    print(
        f"Feature rows: "
        f"{len(final_df)}"
    )

    print(
        f"Behavior variables: "
        f"{len(BEHAVIOR_VARIABLES)}"
    )

    print(
        f"Generated features: "
        f"{len(feature_columns)}"
    )

    print()
    print("Expected behavioral features:")
    print(
        f"  {len(BEHAVIOR_VARIABLES)} variables "
        f"x 2 features = "
        f"{len(BEHAVIOR_VARIABLES) * 2}"
    )

    print()
    print(
        f"Output:\n"
        f"  {output_file}"
    )

    print()
    print(
        f"Processing log:\n"
        f"  {log_file}"
    )


# ============================================================
# 14. RUN
# ============================================================

if __name__ == "__main__":
    main()
