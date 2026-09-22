from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "pmdata"
RESULTS_DIR = PROJECT_ROOT / "ML" / "Results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


WELLBEING_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
]

REVERSE_VARIABLES = [
    "fatigue",
    "stress",
]


def find_participant_files():

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


def get_participant_id(file_path):

    return file_path.stem


def calculate_z_score(series):

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


def process_participant(file_path):

    participant_id = get_participant_id(
        file_path
    )

    df = pd.read_csv(
        file_path
    )

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
            f"No date column found in "
            f"{file_path.name}"
        )

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

    for column in WELLBEING_VARIABLES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    output = pd.DataFrame(
        {
            "participant_id": participant_id,
            "date": df["date"],
        }
    )

    component_z_scores = []

    for variable in WELLBEING_VARIABLES:

        z_score = calculate_z_score(
            df[variable]
        )

        if variable in REVERSE_VARIABLES:
            z_score = -z_score

        output[
            f"{variable}_z"
        ] = z_score

        component_z_scores.append(
            z_score
        )

    component_matrix = pd.concat(
        component_z_scores,
        axis=1,
    )

    output["Wellbeing_Index"] = (
        component_matrix.mean(
            axis=1,
            skipna=False,
        )
    )

    total_days = len(output)

    valid_index_days = (
        output[
            "Wellbeing_Index"
        ].notna().sum()
    )

    processing_info = {
        "participant_id": participant_id,
        "source_file": file_path.name,
        "total_rows": total_days,
        "valid_wellbeing_index_rows":
            int(valid_index_days),
        "missing_wellbeing_index_rows":
            int(
                total_days
                - valid_index_days
            ),
    }

    return (
        output,
        processing_info,
    )


def main():

    print()
    print("=" * 80)
    print("BUILD PARTICIPANT-SPECIFIC WELLBEING INDEX")
    print("=" * 80)
    print()

    print(
        f"Data directory:\n{DATA_DIR}"
    )

    print(
        f"Results directory:\n{RESULTS_DIR}"
    )

    print()

    participant_files = (
        find_participant_files()
    )

    print(
        f"Participant files found: "
        f"{len(participant_files)}"
    )

    print()

    all_results = []
    processing_log = []

    for file_path in participant_files:

        participant_id = (
            get_participant_id(
                file_path
            )
        )

        print(
            f"Processing: "
            f"{participant_id}"
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
                    "participant_id":
                        participant_id,
                    "source_file":
                        file_path.name,
                    "total_rows": 0,
                    "valid_wellbeing_index_rows":
                        0,
                    "missing_wellbeing_index_rows":
                        0,
                    "error": str(error),
                }
            )

    if not all_results:
        raise RuntimeError(
            "No participant data were "
            "successfully processed."
        )

    wellbeing_index = pd.concat(
        all_results,
        ignore_index=True,
    )

    wellbeing_index = (
        wellbeing_index.sort_values(
            [
                "participant_id",
                "date",
            ]
        ).reset_index(
            drop=True
        )
    )

    processing_log_df = (
        pd.DataFrame(
            processing_log
        )
    )

    wellbeing_output = (
        RESULTS_DIR
        / "wellbeing_index.csv"
    )

    processing_output = (
        RESULTS_DIR
        / "wellbeing_index_processing_log.csv"
    )

    wellbeing_index.to_csv(
        wellbeing_output,
        index=False,
    )

    processing_log_df.to_csv(
        processing_output,
        index=False,
    )

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

    print(
        f"Saved:\n{processing_output}"
    )

    print()


if __name__ == "__main__":
    main()
