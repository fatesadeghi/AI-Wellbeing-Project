import os
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# This file is located at:
# AI-Wellbeing-Project/final/Code/01_build_baseline.py
#
# Therefore:
# dirname(__file__)              -> final/Code
# dirname(dirname(__file__))     -> final
# dirname(dirname(dirname(...))) -> AI-Wellbeing-Project

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
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


# ============================================================
# 2. SETTINGS
# ============================================================

BASELINE_FRACTION = 0.50
MIN_BASELINE_N = 7

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


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. CHECK DATA DIRECTORY
# ============================================================

print("=" * 70)
print("PERSONALIZED BASELINE AND DAILY Z-SCORES")
print("=" * 70)

print(
    f"Project directory: {BASE_DIR}"
)

print(
    f"Data directory: {DATA_DIR}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print(
    f"Baseline fraction: {BASELINE_FRACTION}"
)

print(
    f"Minimum baseline observations: {MIN_BASELINE_N}"
)

print()


if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        f"\nData directory not found:\n{DATA_DIR}\n\n"
        "Expected structure:\n"
        "AI-Wellbeing-Project/\n"
        "├── data/\n"
        "│   └── pmdata/\n"
        "└── final/\n"
        "    └── Code/\n"
        "        └── 01_build_baseline.py"
    )


# ============================================================
# 5. FIND PARTICIPANT FILES
# ============================================================

participant_files = sorted(
    [
        filename
        for filename in os.listdir(DATA_DIR)
        if filename.endswith("_daily_merged.csv")
    ]
)


if not participant_files:

    raise RuntimeError(
        f"No participant files found in:\n{DATA_DIR}\n\n"
        "Expected files such as:\n"
        "p01_daily_merged.csv"
    )


print(
    f"Participant files found: {len(participant_files)}"
)

print()


# ============================================================
# 6. STORAGE
# ============================================================

baseline_results = []
daily_results = []


# ============================================================
# 7. PROCESS EACH PARTICIPANT
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

    print(
        f"Processing {participant}..."
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        filepath
    )

    # --------------------------------------------------------
    # CHECK DATE
    # --------------------------------------------------------

    if "Date" not in df.columns:

        print(
            f"  Skipped: Date column not found."
        )

        continue

    # --------------------------------------------------------
    # PREPARE DATE
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    ).copy()

    # Sort chronologically.
    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # Remove duplicate dates.
    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # CHECK NUMBER OF OBSERVATIONS
    # --------------------------------------------------------

    n_total = len(df)

    if n_total < 2:

        print(
            f"  Skipped: only {n_total} valid days."
        )

        continue

    # --------------------------------------------------------
    # 50/50 CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    baseline_n = int(
        np.floor(
            n_total * BASELINE_FRACTION
        )
    )

    if baseline_n < MIN_BASELINE_N:

        print(
            f"  Skipped: baseline has only "
            f"{baseline_n} observations "
            f"(minimum = {MIN_BASELINE_N})."
        )

        continue

    baseline_df = df.iloc[
        :baseline_n
    ].copy()

    analysis_df = df.iloc[
        baseline_n:
    ].copy()

    if analysis_df.empty:

        print(
            "  Skipped: no analysis-period data."
        )

        continue

    # --------------------------------------------------------
    # BASELINE DATE INFORMATION
    # --------------------------------------------------------

    baseline_start = (
        baseline_df["Date"].min()
    )

    baseline_end = (
        baseline_df["Date"].max()
    )

    analysis_start = (
        analysis_df["Date"].min()
    )

    analysis_end = (
        analysis_df["Date"].max()
    )

    # --------------------------------------------------------
    # BASELINE STATISTICS
    # --------------------------------------------------------

    baseline_record = {
        "Participant": participant,
        "Total_N": n_total,
        "Baseline_N": baseline_n,
        "Analysis_N": len(analysis_df),
        "Baseline_Start": baseline_start,
        "Baseline_End": baseline_end,
        "Analysis_Start": analysis_start,
        "Analysis_End": analysis_end,
    }

    # --------------------------------------------------------
    # BEHAVIOR + WELLBEING BASELINE
    # --------------------------------------------------------

    all_baseline_variables = (
        BEHAVIOR_VARIABLES
        + WELLBEING_VARIABLES
    )

    for variable in all_baseline_variables:

        if variable not in baseline_df.columns:

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
        )

        valid_values = values.dropna()

        n_valid = len(valid_values)

        if n_valid == 0:

            mean_value = np.nan
            median_value = np.nan
            sd_value = np.nan

        else:

            mean_value = valid_values.mean()

            median_value = valid_values.median()

            if n_valid >= 2:

                sd_value = valid_values.std(
                    ddof=1
                )

            else:

                sd_value = np.nan

        # A zero or negative SD cannot be used
        # for a meaningful Z-score.
        if pd.notna(sd_value) and sd_value <= 0:

            sd_value = np.nan

        baseline_record[
            f"{variable}_Mean"
        ] = mean_value

        baseline_record[
            f"{variable}_Median"
        ] = median_value

        baseline_record[
            f"{variable}_SD"
        ] = sd_value

        baseline_record[
            f"{variable}_N"
        ] = n_valid

    baseline_results.append(
        baseline_record
    )

    # --------------------------------------------------------
    # DAILY ANALYSIS-PERIOD VALUES
    # --------------------------------------------------------

    for _, row in analysis_df.iterrows():

        daily_record = {
            "Participant": participant,
            "Date": row["Date"],
        }

        # ----------------------------------------------------
        # BEHAVIORAL VARIABLES
        # ----------------------------------------------------

        for variable in BEHAVIOR_VARIABLES:

            if variable not in df.columns:

                daily_record[
                    f"{variable}_Value"
                ] = np.nan

                daily_record[
                    f"{variable}_Deviation"
                ] = np.nan

                daily_record[
                    f"{variable}_Z"
                ] = np.nan

                daily_record[
                    f"{variable}_Abs_Z"
                ] = np.nan

                continue

            value = pd.to_numeric(
                pd.Series([row[variable]]),
                errors="coerce"
            ).iloc[0]

            mean_value = baseline_record[
                f"{variable}_Mean"
            ]

            median_value = baseline_record[
                f"{variable}_Median"
            ]

            sd_value = baseline_record[
                f"{variable}_SD"
            ]

            # Raw deviation from personalized baseline median.
            if pd.notna(value) and pd.notna(median_value):

                deviation = (
                    value - median_value
                )

            else:

                deviation = np.nan

            # Z-score based on personalized baseline mean/SD.
            if (
                pd.notna(value)
                and pd.notna(mean_value)
                and pd.notna(sd_value)
                and sd_value > 0
            ):

                z_score = (
                    value - mean_value
                ) / sd_value

            else:

                z_score = np.nan

            daily_record[
                f"{variable}_Value"
            ] = value

            daily_record[
                f"{variable}_Deviation"
            ] = deviation

            daily_record[
                f"{variable}_Z"
            ] = z_score

            daily_record[
                f"{variable}_Abs_Z"
            ] = (
                abs(z_score)
                if pd.notna(z_score)
                else np.nan
            )

        # ----------------------------------------------------
        # WELLBEING VARIABLES
        # ----------------------------------------------------

        for variable in WELLBEING_VARIABLES:

            if variable not in df.columns:

                daily_record[
                    f"{variable}_Value"
                ] = np.nan

                daily_record[
                    f"{variable}_Z"
                ] = np.nan

                continue

            value = pd.to_numeric(
                pd.Series([row[variable]]),
                errors="coerce"
            ).iloc[0]

            mean_value = baseline_record[
                f"{variable}_Mean"
            ]

            sd_value = baseline_record[
                f"{variable}_SD"
            ]

            if (
                pd.notna(value)
                and pd.notna(mean_value)
                and pd.notna(sd_value)
                and sd_value > 0
            ):

                z_score = (
                    value - mean_value
                ) / sd_value

            else:

                z_score = np.nan

            daily_record[
                f"{variable}_Value"
            ] = value

            daily_record[
                f"{variable}_Z"
            ] = z_score

        daily_results.append(
            daily_record
        )


# ============================================================
# 8. CREATE OUTPUT DATAFRAMES
# ============================================================

baseline_df = pd.DataFrame(
    baseline_results
)

daily_deviation_df = pd.DataFrame(
    daily_results
)


# ============================================================
# 9. CHECK RESULTS
# ============================================================

if baseline_df.empty:

    raise RuntimeError(
        "No participants passed the baseline criteria."
    )

if daily_deviation_df.empty:

    raise RuntimeError(
        "No analysis-period observations were created."
    )


# ============================================================
# 10. SORT RESULTS
# ============================================================

baseline_df = baseline_df.sort_values(
    "Participant"
).reset_index(drop=True)

daily_deviation_df = daily_deviation_df.sort_values(
    ["Participant", "Date"]
).reset_index(drop=True)


# ============================================================
# 11. SAVE BASELINE RESULTS
# ============================================================

baseline_output = os.path.join(
    OUTPUT_DIR,
    "personalized_baselines.csv"
)

baseline_df.to_csv(
    baseline_output,
    index=False
)


# ============================================================
# 12. SAVE DAILY DEVIATIONS
# ============================================================

daily_output = os.path.join(
    OUTPUT_DIR,
    "daily_personalized_deviations.csv"
)

daily_deviation_df.to_csv(
    daily_output,
    index=False
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("BASELINE CONSTRUCTION COMPLETED")
print("=" * 70)

print(
    f"Participants included: "
    f"{baseline_df['Participant'].nunique()}"
)

print(
    f"Baseline records: "
    f"{len(baseline_df)}"
)

print(
    f"Analysis-day records: "
    f"{len(daily_deviation_df)}"
)

print()
print(
    "Saved:"
)

print(
    f"  {baseline_output}"
)

print(
    f"  {daily_output}"
)

print("=" * 70)
