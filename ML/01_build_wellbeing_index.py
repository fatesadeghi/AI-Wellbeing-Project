from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "pmdata"

OUTPUT_DIR = PROJECT_ROOT / "ML" / "Results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

WELLBEING_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
]

# Positive direction means:
# higher value = better wellbeing
DIRECTION = {
    "fatigue": -1,
    "mood": 1,
    "readiness": 1,
    "sleep_quality": 1,
    "stress": -1,
}

EXPECTED_OUTPUT_COLUMNS = [
    "participant_id",
    "date",
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
    "Z_fatigue",
    "Z_mood",
    "Z_readiness",
    "Z_sleep_quality",
    "Z_stress",
    "Wellbeing_Index",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def find_date_column(df: pd.DataFrame) -> str:
    """
    Find the date column in a participant dataframe.
    """

    possible_names = [
        "date",
        "Date",
        "DATE",
        "day",
        "Day",
    ]

    for column in possible_names:
        if column in df.columns:
            return column

    raise ValueError(
        "No date column found. Expected one of: "
        + ", ".join(possible_names)
    )


def find_participant_column(df: pd.DataFrame) -> str | None:
    """
    Find participant ID column if it exists.
    """

    possible_names = [
        "participant_id",
        "Participant_ID",
        "participant",
        "Participant",
        "id",
        "ID",
    ]

    for column in possible_names:
        if column in df.columns:
            return column

    return None


def calculate_participant_zscore(
    series: pd.Series,
) -> pd.Series:
    """
    Calculate a participant-specific Z-score.

    Missing observations remain missing.

    If the participant has no variance for a variable,
    the Z-score cannot be calculated meaningfully, so NaN
    is returned for that variable.
    """

    mean_value = series.mean(skipna=True)
    std_value = series.std(skipna=True, ddof=1)

    if pd.isna(std_value) or std_value == 0:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    return (series - mean_value) / std_value


def build_wellbeing_index(
    participant_df: pd.DataFrame,
    participant_id: str,
) -> pd.DataFrame:
    """
    Build the Wellbeing Index for one participant.
    """

    df = participant_df.copy()

    date_column = find_date_column(df)

    # --------------------------------------------------------
    # Validate required wellbeing columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in WELLBEING_VARIABLES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Participant {participant_id}: missing wellbeing "
            f"columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Prepare date
    # --------------------------------------------------------

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    if df[date_column].isna().all():
        raise ValueError(
            f"Participant {participant_id}: no valid dates found."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(date_column).reset_index(drop=True)

    # --------------------------------------------------------
    # Keep only required information
    # --------------------------------------------------------

    result = pd.DataFrame()

    result["participant_id"] = participant_id
    result["date"] = df[date_column]

    for variable in WELLBEING_VARIABLES:
        result[variable] = pd.to_numeric(
            df[variable],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Participant-specific Z-scores
    # --------------------------------------------------------

    for variable in WELLBEING_VARIABLES:

        z_column = f"Z_{variable}"

        result[z_column] = calculate_participant_zscore(
            result[variable]
        )

        # Reverse variables where higher raw values mean
        # worse wellbeing.
        if DIRECTION[variable] == -1:
            result[z_column] = -result[z_column]

    # --------------------------------------------------------
    # Calculate the Wellbeing Index
    # --------------------------------------------------------
    #
    # We use equal weights.
    #
    # Important:
    # FDR significance is NOT used as a weighting mechanism.
    # The statistical analysis and the wellbeing index serve
    # different purposes.
    #

    z_columns = [
        "Z_fatigue",
        "Z_mood",
        "Z_readiness",
        "Z_sleep_quality",
        "Z_stress",
    ]

    result["Wellbeing_Index"] = result[z_columns].mean(
        axis=1,
        skipna=False,
    )

    return result


# ============================================================
# 4. LOAD PARTICIPANT FILES
# ============================================================

def find_participant_files() -> list[Path]:
    """
    Find participant CSV files inside data/pmdata.
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found:\n{DATA_DIR}"
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{DATA_DIR}"
        )

    return files


# ============================================================
# 5. PROCESS ALL PARTICIPANTS
# ============================================================

def main() -> None:

    print("=" * 80)
    print("BUILD PARTICIPANT-SPECIFIC WELLBEING INDEX")
    print("=" * 80)

    participant_files = find_participant_files()

    print(f"Participant files found: {len(participant_files)}")
    print()

    all_results = []
    processing_errors = []

    for file_path in participant_files:

        participant_id = file_path.stem

        print("-" * 80)
        print(f"Participant: {participant_id}")

        try:

            df = pd.read_csv(file_path)

            # If participant ID is already stored in the file,
            # use it. Otherwise use the filename.
            participant_column = find_participant_column(df)

            if participant_column is not None:

                non_missing_ids = (
                    df[participant_column]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                if len(non_missing_ids) == 1:
                    participant_id = non_missing_ids[0]

            result = build_wellbeing_index(
                participant_df=df,
                participant_id=participant_id,
            )

            valid_index = result["Wellbeing_Index"].notna().sum()

            print(f"  Rows: {len(result)}")
            print(f"  Valid Wellbeing Index: {valid_index}")

            if valid_index == 0:
                print(
                    "  WARNING: no valid Wellbeing Index "
                    "values were produced."
                )

            all_results.append(result)

        except Exception as error:

            processing_errors.append(
                {
                    "participant_id": participant_id,
                    "file": file_path.name,
                    "error": str(error),
                }
            )

            print(f"  ERROR: {error}")

    # ========================================================
    # 6. COMBINE RESULTS
    # ========================================================

    if not all_results:
        raise RuntimeError(
            "No participant could be processed successfully."
        )

    final_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    final_df = final_df.sort_values(
        ["participant_id", "date"]
    ).reset_index(drop=True)

    # ========================================================
    # 7. SAVE MAIN OUTPUT
    # ========================================================

    output_file = (
        OUTPUT_DIR / "wellbeing_index.csv"
    )

    final_df.to_csv(
        output_file,
        index=False,
    )

    # ========================================================
    # 8. SAVE PROCESSING LOG
    # ========================================================

    log_file = (
        OUTPUT_DIR / "wellbeing_index_processing_log.csv"
    )

    if processing_errors:

        log_df = pd.DataFrame(processing_errors)

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
    # 9. SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)

    print(f"Participants processed: {len(all_results)}")
    print(
        f"Participants with errors: "
        f"{len(processing_errors)}"
    )

    print(
        f"Total rows: {len(final_df)}"
    )

    print(
        f"Valid Wellbeing Index values: "
        f"{final_df['Wellbeing_Index'].notna().sum()}"
    )

    print()
    print(f"Output:")
    print(f"  {output_file}")

    print()
    print(f"Processing log:")
    print(f"  {log_file}")

    print()
    print("Wellbeing Index formula:")
    print(
        "(-Z_fatigue + Z_mood + Z_readiness "
        "+ Z_sleep_quality - Z_stress) / 5"
    )


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":
    main()
