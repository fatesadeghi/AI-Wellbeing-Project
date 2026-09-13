```python
import os
import pandas as pd
import numpy as np


# ============================================================
# 0. PATHS
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
# 1. SETTINGS
# ============================================================

# First 50% of each participant's chronological data
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
# 2. LOAD PARTICIPANT DATA
# ============================================================

participant_data = {}

print("=" * 70)
print("LOADING PARTICIPANT DATA")
print("=" * 70)


for participant in sorted(
    os.listdir(DATA_DIR)
):

    participant_path = os.path.join(
        DATA_DIR,
        participant
    )

    if not os.path.isdir(
        participant_path
    ):
        continue

    csv_files = []

    for file in os.listdir(
        participant_path
    ):

        if file.lower().endswith(".csv"):

            csv_files.append(
                os.path.join(
                    participant_path,
                    file
                )
            )

    if not csv_files:
        continue

    dataframes = []

    for file in sorted(csv_files):

        try:

            df = pd.read_csv(file)

            dataframes.append(df)

        except Exception as e:

            print(
                f"Could not read {file}: {e}"
            )

    if dataframes:

        participant_df = pd.concat(
            dataframes,
            ignore_index=True
        )

        participant_df = (
            participant_df
            .drop_duplicates()
        )

        # Convert Date to datetime
        if "Date" in participant_df.columns:

            participant_df["Date"] = pd.to_datetime(
                participant_df["Date"],
                errors="coerce"
            )

            participant_df = (
                participant_df
                .sort_values("Date")
                .reset_index(drop=True)
            )

        participant_data[
            participant
        ] = participant_df

        print(
            f"{participant}: "
            f"{len(participant_df)} rows"
        )


print()
print(
    "Participants loaded:",
    len(participant_data)
)


# ============================================================
# 3. CREATE PERSONALIZED BASELINES
# ============================================================

baseline_records = []

daily_deviation_records = []


print()
print("=" * 70)
print("BUILDING PERSONALIZED BASELINES")
print("=" * 70)


for participant, df in participant_data.items():

    # --------------------------------------------------------
    # Determine baseline period
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Calculate baseline statistics
    # --------------------------------------------------------

    participant_baseline = {
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


        # Avoid division by zero
        if pd.isna(std_value) or std_value == 0:

            std_value = np.nan


        baseline_stats[
            variable
        ] = {

            "mean": mean_value,

            "median": median_value,

            "std": std_value

        }


        participant_baseline[
            f"{variable}_Mean"
        ] = mean_value


        participant_baseline[
            f"{variable}_Median"
        ] = median_value


        participant_baseline[
            f"{variable}_SD"
        ] = std_value


    baseline_records.append(
        participant_baseline
    )


    # ========================================================
    # 4. CALCULATE DAILY DEVIATIONS
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
                pd.Series([row[variable]]),
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


            # Raw deviation from personal median
            if pd.notna(value):

                record[
                    f"{variable}_Deviation"
                ] = value - median_value

            else:

                record[
                    f"{variable}_Deviation"
                ] = np.nan


            # Standardized deviation from personal mean
            if (
                pd.notna(value)
                and pd.notna(std_value)
                and std_value > 0
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


        daily_deviation_records.append(
            record
        )


# ============================================================
# 5. SAVE BASELINE STATISTICS
# ============================================================

baseline_df = pd.DataFrame(
    baseline_records
)


baseline_output = os.path.join(
    OUTPUT_DIR,
    "personalized_baselines.xlsx"
)


baseline_df.to_excel(
    baseline_output,
    index=False
)


# ============================================================
# 6. SAVE DAILY DEVIATIONS
# ============================================================

deviation_df = pd.DataFrame(
    daily_deviation_records
)


deviation_output = os.path.join(
    OUTPUT_DIR,
    "daily_personalized_deviations.xlsx"
)


deviation_df.to_excel(
    deviation_output,
    index=False
)


# ============================================================
# 7. SUMMARY
# ============================================================

print()
print("=" * 70)
print("BASELINE ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    "Participants:",
    len(participant_data)
)

print(
    "Baseline fraction:",
    BASELINE_FRACTION
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
