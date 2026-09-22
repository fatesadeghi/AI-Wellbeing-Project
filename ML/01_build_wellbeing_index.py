from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "pmdata"

RESULTS_DIR = PROJECT_ROOT / "ML" / "Results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. VARIABLES
# ============================================================

WELLBEING_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
]


# Variables where higher values indicate worse wellbeing.
REVERSE_VARIABLES = [
    "fatigue",
    "stress",
]


# Equal weighting across the five wellbeing components.
N_COMPONENTS = len(WELLBEING_VARIABLES)


# ============================================================
# 3. FILE DISCOVERY
# ============================================================

def find_participant_files():
    """
    Find participant CSV files inside data/pmdata.
    """

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Data directory not found:\n{DATA_DIR}"
        )

    participant_files = sorted(
        DATA_DIR.glob("*.csv")
    )

    if not participant_files:

        raise FileNotFoundError(
            f"No participant CSV files found in:\n{DATA_DIR}"
        )

    return participant_files


# ============================================================
# 4. PARTICIPANT ID
# ============================================================

def get_participant_id(file_path):
    """
    Extract participant ID from the CSV filename.

    Example:
        p01.csv -> p01
    """

    return file_path.stem


# ============================================================
# 5. Z-SCORE
# ============================================================

def calculate_z_score(series):
    """
    Calculate a participant-specific z-score.

    Missing values remain missing.

    If the participant has zero variance, the standardized
    values are set to 0 for the available observations.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    mean_value = numeric_series.mean()

    std_value = numeric_series.std(
        ddof=0
    )

    if pd.isna(std_value) or std_value == 0:

        result = pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

        result.loc[
            numeric_series.notna()
        ] = 0.0

        return result

    return (
        numeric_series - mean_value
    ) / std_value


# ============================================================
# 6. PROCESS ONE PARTICIPANT
# ============================================================

def process_participant(file_path):
    """
    Process one participant file and return:

        wellbeing dataframe
        processing information
    """

    participant_id = get_participant_id(
        file_path
    )

    df = pd.read_csv(
        file_path
    )

    # --------------------------------------------------------
    # Find date column
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Check wellbeing columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in WELLBEING_VARIABLES
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing wellbeing columns in "
            f"{file_path.name}: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Prepare dataframe
    # --------------------------------------------------------

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[date_column]
    ).copy()

    df = df.sort_values(
        date_column
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Convert wellbeing variables to numeric
    # --------------------------------------------------------

    for column in WELLBEING_VARIABLES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Create output dataframe
    # --------------------------------------------------------

    output = pd.DataFrame(
        {
            "participant_id": participant_id,
            "date": df[date_column],
        }
    )

    # --------------------------------------------------------
    # Calculate participant-specific z-scores
    # --------------------------------------------------------

    component_z_scores = []

    for variable in WELLBEING_VARIABLES:

        z_score = calculate_z_score(
            df[variable]
        )

        # Reverse variables where higher values
        # represent worse wellbeing.
        if variable in REVERSE_VARIABLES:

            z_score = -z_score

        output[
            f"{variable}_z"
        ] = z_score

        component_z_scores.append(
            z_score
        )

    # --------------------------------------------------------
    # Calculate Wellbeing Index
    # --------------------------------------------------------

    component_matrix = pd.concat(
        component_z_scores,
        axis=1,
    )

    output[
        "Wellbeing_Index"
    ] = component_matrix.mean(
        axis=1,
        skipna=False,
    )

    # --------------------------------------------------------
    # Processing information
    # --------------------------------------------------------

    total_days = len(output)

    valid_index_days = (
        output["Wellbeing_Index"]
        .notna()
        .sum()
    )

    processing_info = {
        "participant_id": participant_id,
        "source_file": file_path.name,
        "total_rows": total_days,
        "valid_wellbeing_index_rows": int(
            valid_index_days
        ),
        "missing_wellbeing_index_rows": int(
            total_days - valid_index_days
        ),
    }

    return output, processing_info


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("BUILD PARTICIPANT-SPECIFIC WELLBEING INDEX")
    print("=" * 80)
    print()

    print(
        f"Project root:\n{PROJECT_ROOT}"
    )

    print()

    print(
        f"Data directory:\n{DATA_DIR}"
    )

    print()

    print(
        f"Results directory:\n{RESULTS_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Find participant files
    # --------------------------------------------------------

    participant_files = find_participant_files()

    print(
        f"Participant files found: "
        f"{len(participant_files)}"
    )

    print()

    # --------------------------------------------------------
    # Process participants
    # --------------------------------------------------------

    all_results = []
    processing_log = []

    for file_path in participant_files:

        participant_id = get_participant_id(
            file_path
        )

        print(
            f"Processing: {participant_id}"
        )

        try:

            participant_result, info = (
                process_participant(
                    file_path
                )
            )

            all_results.append(
                participant_result
            )

            processing_log.append(
                info
            )

            print(
                f"  Rows: "
                f"{info['total_rows']}"
            )

            print(
                f"  Valid index rows: "
                f"{info['valid_wellbeing_index_rows']}"
            )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

            processing_log.append(
                {
                    "participant_id": participant_id,
                    "source_file": file_path.name,
                    "total_rows": 0,
                    "valid_wellbeing_index_rows": 0,
                    "missing_wellbeing_index_rows": 0,
                    "error": str(error),
                }
            )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not all_results:

        raise RuntimeError(
            "No participant data were successfully processed."
        )

    # --------------------------------------------------------
    # Combine all participants
    # --------------------------------------------------------

    wellbeing_index = pd.concat(
        all_results,
        ignore_index=True,
    )

    wellbeing_index = wellbeing_index.sort_values(
        [
            "participant_id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    processing_log_df = pd.DataFrame(
        processing_log
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    wellbeing_output = (
        RESULTS_DIR
        / "wellbeing_index.csv"
    )

    processing_output = (
        RESULTS_DIR
        / "wellbeing_index_processing_log.csv"
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    wellbeing_index.to_csv(
        wellbeing_output,
        index=False,
    )

    processing_log_df.to_csv(
        processing_output,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("WELLBEING INDEX COMPLETED")
    print("=" * 80)
    print()

    print(
        f"Participants processed: "
        f"{len(all_results)}"
    )

    print(
        f"Total output rows: "
        f"{len(wellbeing_index)}"
    )

    print(
        f"Valid Wellbeing Index rows: "
        f"{wellbeing_index['Wellbeing_Index'].notna().sum()}"
    )

    print()

    print(
        f"Saved:\n{wellbeing_output}"
    )

    print()

    print(
        f"Saved:\n{processing_output}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
