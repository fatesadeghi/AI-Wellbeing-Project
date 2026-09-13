python
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

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "pmdata"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "baseline"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

# The first 50% of each participant's chronological data
# is used to learn their personal baseline.

BASELINE_FRACTION = 0.50


OBJECTIVE_VARIABLES = [
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


# ============================================================
# 3. LOAD PARTICIPANT DATA
# ============================================================

participant_data = {}

print("=" * 70)
print("LOADING PARTICIPANT DATA")
print("=" * 70)

print(
    f"Data directory: {DATA_DIR}"
)

print()


if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        f"Data directory not found:\n{DATA_DIR}"
    )


for file in sorted(
    os.listdir(DATA_DIR)
):

    if not file.endswith(
        "_daily_merged.csv"
    ):

        continue


    participant = file.replace(
        "_daily_merged.csv",
        ""
    )


    file_path = os.path.join(
        DATA_DIR,
        file
    )


    try:

        df = pd.read_csv(
            file_path
        )

    except Exception as e:

        print(
            f"Could not read {file}: {e}"
        )

        continue


    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = (
            df
            .sort_values("Date")
            .reset_index(drop=True)
        )


    df = df.drop_duplicates()


    participant_data[
        participant
    ] = df


    print(
        f"{participant}: "
        f"{len(df)} rows"
    )


print()

print(
    "Participants loaded:",
    len(participant_data)
)


# ============================================================
# 4. BUILD PERSONALIZED BASELINES
# ============================================================

baseline_records = []

deviation_records = []


print()
print("=" * 70)
print("BUILDING PERSONALIZED BASELINES")
print("=" * 70)


for participant, df in participant_data.items():

    n_rows = len(df)

    baseline_end = int(
        n_rows * BASELINE_FRACTION
    )


    if baseline_end < 2:

        print(
            f"{participant}: "
            f"not enough data for baseline"
        )

        continue


    baseline_df = (
        df
        .iloc[:baseline_end]
        .copy()
    )


    analysis_df = (
        df
        .iloc[baseline_end:]
        .copy()
    )


    print(
        f"{participant}: "
        f"baseline={len(baseline_df)} days, "
        f"analysis={len(analysis_df)} days"
    )


    # ========================================================
    # 5. PERSONAL BASELINE STATISTICS
    # ========================================================

    baseline_record = {

        "Participant": participant,

        "Baseline_N": len(baseline_df)

    }


    baseline_stats = {}


    for variable in OBJECTIVE_VARIABLES:

        if variable not in baseline_df.columns:

            continue


        values = pd.to_numeric(
            baseline_df[variable],
            errors="coerce"
        ).dropna()


        if len(values) < 2:

            continue


        mean_value = values.mean()

        median_value = values.median()

        std_value = values.std(
            ddof=1
        )


        if (
            pd.isna(std_value)
            or
            std_value == 0
        ):

            std_value = np.nan


        baseline_stats[
            variable
        ] = {

            "mean": mean_value,

            "median": median_value,

            "std": std_value

        }


        baseline_record[
            f"{variable}_Mean"
        ] = mean_value


        baseline_record[
            f"{variable}_Median"
        ] = median_value


        baseline_record[
            f"{variable}_SD"
        ] = std_value


    baseline_records.append(
        baseline_record
    )


    # ========================================================
    # 6. DAILY DEVIATIONS
    # ========================================================

    for _, row in analysis_df.iterrows():

        record = {

            "Participant": participant,

            "Date": row.get("Date")

        }


        for variable in OBJECTIVE_VARIABLES:

            if variable not in row.index:

                continue


            if variable not in baseline_stats:

                continue


            value = pd.to_numeric(
                pd.Series(
                    [row[variable]]
                ),
                errors="coerce"
            ).iloc[0]


            mean_value = baseline_stats[
                variable
            ]["mean"]


            median_value = baseline_stats[
                variable
            ]["median"]


            std_value = baseline_stats[
                variable
            ]["std"]


            # ------------------------------------------------
            # Raw deviation from personal median
            # ------------------------------------------------

            if pd.notna(value):

                record[
                    f"{variable}_Deviation"
                ] = (
                    value - median_value
                )

            else:

                record[
                    f"{variable}_Deviation"
                ] = np.nan


            # ------------------------------------------------
            # Standardized deviation
            # ------------------------------------------------

            if (
                pd.notna(value)
                and
                pd.notna(std_value)
                and
                std_value > 0
            ):

                z_score = (
                    value - mean_value
                ) / std_value


                record[
                    f"{variable}_Z"
                ] = z_score


                record[
                    f"{variable}_Abs_Z"
                ] = abs(z_score)


            else:

                record[
                    f"{variable}_Z"
                ] = np.nan


                record[
                    f"{variable}_Abs_Z"
                ] = np.nan


        deviation_records.append(
            record
        )


# ============================================================
# 7. CREATE DATAFRAMES
# ============================================================

baseline_df = pd.DataFrame(
    baseline_records
)


deviation_df = pd.DataFrame(
    deviation_records
)


# ============================================================
# 8. SAVE AS CSV
# ============================================================

baseline_output = os.path.join(
    OUTPUT_DIR,
    "personalized_baselines.csv"
)


deviation_output = os.path.join(
    OUTPUT_DIR,
    "daily_personalized_deviations.csv"
)


baseline_df.to_csv(
    baseline_output,
    index=False
)


deviation_df.to_csv(
    deviation_output,
    index=False
)


# ============================================================
# 9. FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("PERSONALIZED BASELINE ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    f"Participants analyzed: "
    f"{len(baseline_df)}"
)

print(
    f"Baseline fraction: "
    f"{BASELINE_FRACTION}"
)

print(
    f"Baseline records: "
    f"{len(baseline_df)}"
)

print(
    f"Daily deviation records: "
    f"{len(deviation_df)}"
)

print()

print(
    "Baseline output:"
)

print(
    baseline_output
)

print()

print(
    "Deviation output:"
)

print(
    deviation_output
)

print()
print("=" * 70)
```
