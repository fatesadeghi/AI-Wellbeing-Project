import os
import pandas as pd
import numpy as np


# ============================================================
# 01 - BUILD PERSONALIZED BASELINE AND DAILY Z-SCORES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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

BASELINE_FRACTION = 0.50
MIN_BASELINE_N = 7


# ============================================================
# VARIABLES
# ============================================================

BEHAVIORAL_VARIABLES = [
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

WELLNESS_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]


# ============================================================
# SETUP
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("PERSONALIZED BASELINE AND DAILY Z-SCORES")
print("=" * 70)

print(f"Data directory: {DATA_DIR}")
print(f"Baseline fraction: {BASELINE_FRACTION}")
print(f"Minimum baseline observations: {MIN_BASELINE_N}")


# ============================================================
# LOAD PARTICIPANT FILES
# ============================================================

participant_files = sorted(
    filename
    for filename in os.listdir(DATA_DIR)
    if filename.endswith("_daily_merged.csv")
)

print(
    f"\nParticipant files found: "
    f"{len(participant_files)}"
)


baseline_records = []
deviation_records = []

participants_processed = 0
participants_skipped = 0


# ============================================================
# PROCESS EACH PARTICIPANT
# ============================================================

for filename in participant_files:

    participant = filename.replace(
        "_daily_merged.csv",
        ""
    )

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(filepath)

    if "Date" not in df.columns:

        print(
            f"{participant}: skipped - no Date column"
        )

        participants_skipped += 1
        continue

    # --------------------------------------------------------
    # Date cleaning
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    df = df.sort_values(
        "Date"
    )

    # Remove duplicate dates
    duplicate_count = (
        df["Date"].duplicated()
        .sum()
    )

    if duplicate_count > 0:

        print(
            f"{participant}: "
            f"removed {duplicate_count} duplicate date rows"
        )

        df = df.drop_duplicates(
            subset=["Date"],
            keep="first"
        )

    df = df.reset_index(
        drop=True
    )

    total_days = len(df)

    if total_days < 2:

        print(
            f"{participant}: skipped - insufficient data"
        )

        participants_skipped += 1
        continue

    # --------------------------------------------------------
    # Define baseline and analysis periods
    # --------------------------------------------------------

    split_index = int(
        total_days * BASELINE_FRACTION
    )

    baseline_df = df.iloc[
        :split_index
    ].copy()

    analysis_df = df.iloc[
        split_index:
    ].copy()

    print(
        f"{participant}: "
        f"total={total_days}, "
        f"baseline={len(baseline_df)}, "
        f"analysis={len(analysis_df)}"
    )

    if len(baseline_df) < MIN_BASELINE_N:

        print(
            f"{participant}: skipped - "
            f"baseline has fewer than "
            f"{MIN_BASELINE_N} observations"
        )

        participants_skipped += 1
        continue

    # --------------------------------------------------------
    # Calculate personalized baseline statistics
    # --------------------------------------------------------

    baseline_record = {
        "Participant": participant,
        "Baseline_N": len(baseline_df),
        "Baseline_Start": baseline_df["Date"].min(),
        "Baseline_End": baseline_df["Date"].max(),
        "Analysis_Start": analysis_df["Date"].min(),
        "Analysis_End": analysis_df["Date"].max()
    }

    baseline_stats = {}

    all_variables = (
        BEHAVIORAL_VARIABLES
        + WELLNESS_VARIABLES
    )

    for variable in all_variables:

        if variable not in df.columns:

            baseline_stats[variable] = {
                "Mean": np.nan,
                "Median": np.nan,
                "SD": np.nan,
                "N": 0
            }

            baseline_record[
                f"{variable}_Mean"
            ] = np.nan

            baseline_record[
                f"{variable}_Median"
            ] = np.nan

            baseline_record[
                f"{variable}_SD"
            ] = np.nan

            baseline_record[
                f"{variable}_N"
            ] = 0

            continue

        values = pd.to_numeric(
            baseline_df[variable],
            errors="coerce"
        ).dropna()

        n = len(values)

        if n == 0:

            mean = np.nan
            median = np.nan
            sd = np.nan

        elif n == 1:

            mean = values.mean()
            median = values.median()
            sd = np.nan

        else:

            mean = values.mean()
            median = values.median()
            sd = values.std(
                ddof=1
            )

        # A zero SD cannot produce a meaningful Z-score
        if pd.notna(sd) and sd <= 0:
            sd = np.nan

        baseline_stats[variable] = {
            "Mean": mean,
            "Median": median,
            "SD": sd,
            "N": n
        }

        baseline_record[
            f"{variable}_Mean"
        ] = mean

        baseline_record[
            f"{variable}_Median"
        ] = median

        baseline_record[
            f"{variable}_SD"
        ] = sd

        baseline_record[
            f"{variable}_N"
        ] = n

    baseline_records.append(
        baseline_record
    )

    # --------------------------------------------------------
    # Calculate daily personalized deviations
    # --------------------------------------------------------

    for _, row in analysis_df.iterrows():

        record = {
            "Participant": participant,
            "Date": row["Date"]
        }

        for variable in all_variables:

            value = np.nan

            if variable in df.columns:

                value = pd.to_numeric(
                    row[variable],
                    errors="coerce"
                )

            mean = baseline_stats[
                variable
            ]["Mean"]

            median = baseline_stats[
                variable
            ]["Median"]

            sd = baseline_stats[
                variable
            ]["SD"]

            # Raw deviation from personal baseline median
            if (
                pd.notna(value)
                and pd.notna(median)
            ):

                deviation = (
                    value - median
                )

            else:

                deviation = np.nan

            # Standardized deviation from personal baseline mean
            if (
                pd.notna(value)
                and pd.notna(mean)
                and pd.notna(sd)
                and sd > 0
            ):

                z = (
                    value - mean
                ) / sd

            else:

                z = np.nan

            record[
                f"{variable}_Value"
            ] = value

            record[
                f"{variable}_Deviation"
            ] = deviation

            record[
                f"{variable}_Z"
            ] = z

            record[
                f"{variable}_Abs_Z"
            ] = (
                abs(z)
                if pd.notna(z)
                else np.nan
            )

        deviation_records.append(
            record
        )

    participants_processed += 1


# ============================================================
# CREATE OUTPUT DATAFRAMES
# ============================================================

baseline_df = pd.DataFrame(
    baseline_records
)

deviation_df = pd.DataFrame(
    deviation_records
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

baseline_file = os.path.join(
    OUTPUT_DIR,
    "personalized_baselines.csv"
)

deviation_file = os.path.join(
    OUTPUT_DIR,
    "daily_personalized_deviations.csv"
)

baseline_df.to_csv(
    baseline_file,
    index=False
)

deviation_df.to_csv(
    deviation_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BASELINE CONSTRUCTION COMPLETE")
print("=" * 70)

print(
    f"Participants processed: "
    f"{participants_processed}"
)

print(
    f"Participants skipped: "
    f"{participants_skipped}"
)

print(
    f"Baseline records: "
    f"{len(baseline_df)}"
)

print(
    f"Analysis-period daily records: "
    f"{len(deviation_df)}"
)

print("\nOutput files:")
print(baseline_file)
print(deviation_file)

print("=" * 70)
