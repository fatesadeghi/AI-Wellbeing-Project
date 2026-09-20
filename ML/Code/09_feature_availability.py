import os
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

# Current file:
# AI-Wellbeing-Project/ML/Code/09_feature_availability.py

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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "ml"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

BEHAVIOR_VARS = [
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

FIRST_10_DAYS = 10

SLEEP_VARIABLES = [
    "Sleep_Hours",
    "Sleep_Duration_Score",
    "Deep_Sleep_Minutes",
    "Sleep_Restlessness",
    "Sleep_Composition",
    "Sleep_Revitalization",
    "Sleep_Score"
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def longest_missing_streak(series):
    """
    Calculate the longest consecutive missing streak.
    """

    missing = series.isna().astype(int)

    if missing.empty:
        return 0

    groups = (
        missing.ne(
            missing.shift()
        ).cumsum()
    )

    streaks = (
        missing
        .groupby(groups)
        .sum()
    )

    if len(streaks) == 0:
        return 0

    return int(
        streaks.max()
    )


def first_10_days_missing(df, variable):
    """
    Check whether the variable is completely missing
    during the first 10 consecutive calendar days.

    Missing calendar dates are explicitly represented
    as missing observations.
    """

    check = df[
        ["Date", variable]
    ].copy()

    check = check.sort_values(
        "Date"
    )

    if check.empty:
        return False

    # Start from the participant's first recorded date
    start_date = check["Date"].min()

    # Define the first 10 consecutive calendar days
    first_10_dates = pd.date_range(
        start=start_date,
        periods=FIRST_10_DAYS,
        freq="D"
    )

    check = check.set_index(
        "Date"
    )

    # Reindex so missing calendar dates are explicitly represented
    first_10 = check.reindex(
        first_10_dates
    )

    # Variable must be missing on all first 10 calendar days
    return bool(
        first_10[variable].isna().all()
    )


# ============================================================
# 4. FIND PARTICIPANTS
# ============================================================

participant_files = sorted(
    [
        filename
        for filename in os.listdir(DATA_DIR)
        if filename.endswith(".csv")
        and filename.startswith("p")
    ]
)

if not participant_files:
    raise FileNotFoundError(
        f"No participant CSV files found in:\n{DATA_DIR}"
    )


print("=" * 70)
print("FEATURE AVAILABILITY ANALYSIS")
print("=" * 70)

print(
    f"Participants found: {len(participant_files)}"
)


# ============================================================
# 5. ANALYZE EACH PARTICIPANT
# ============================================================

availability_rows = []


for filename in participant_files:

    participant = os.path.splitext(
        filename
    )[0]

    file_path = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(
        file_path
    )

    print("\n" + "-" * 70)
    print(f"Processing: {participant}")

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if "Date" not in df.columns:

        print(
            f"WARNING: Date column missing for {participant}"
        )

        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = (
        df
        .dropna(
            subset=["Date"]
        )
        .sort_values(
            "Date"
        )
        .drop_duplicates(
            subset="Date",
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )

    total_rows = len(df)

    # --------------------------------------------------------
    # Behavioral variables
    # --------------------------------------------------------

    for variable in BEHAVIOR_VARS:

        # ----------------------------------------------------
        # Missing column
        # ----------------------------------------------------

        if variable not in df.columns:

            availability_rows.append({
                "Participant": participant,
                "Variable": variable,
                "Variable_Type": (
                    "Sleep"
                    if variable in SLEEP_VARIABLES
                    else "Activity"
                ),
                "Status": "Missing_Column",
                "Total_Rows": total_rows,
                "Missing_Count": np.nan,
                "Missing_Percent": np.nan,
                "Longest_Missing_Streak": np.nan,
                "First_10_Days_All_Missing": False
            })

            print(
                f"{variable}: MISSING COLUMN"
            )

            continue

        # ----------------------------------------------------
        # Missingness statistics
        # ----------------------------------------------------

        missing_count = int(
            df[variable].isna().sum()
        )

        missing_percent = (
            missing_count / total_rows * 100
            if total_rows > 0
            else np.nan
        )

        longest_streak = longest_missing_streak(
            df[variable]
        )

        is_sleep = (
            variable in SLEEP_VARIABLES
        )

        # ----------------------------------------------------
        # First 10 calendar days
        # ----------------------------------------------------

        first10_missing = first_10_days_missing(
            df,
            variable
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if is_sleep:

            # Sleep variables are exempt from the
            # first-10-days exclusion rule.

            if missing_count == 0:

                status = "Available_Sleep_Exempt"

            else:

                status = "Available_With_Missing"

        else:

            if first10_missing:

                status = (
                    "Excluded_First_10_Days_Missing"
                )

            elif missing_count == 0:

                status = "Available"

            elif longest_streak >= FIRST_10_DAYS:

                # A long missing streak occurring later in
                # the record does not trigger exclusion.
                # It is reported only.

                status = (
                    "Available_With_Long_Missing_Streak"
                )

            else:

                status = "Available_With_Missing"

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        availability_rows.append({
            "Participant": participant,
            "Variable": variable,
            "Variable_Type": (
                "Sleep"
                if is_sleep
                else "Activity"
            ),
            "Status": status,
            "Total_Rows": total_rows,
            "Missing_Count": missing_count,
            "Missing_Percent": missing_percent,
            "Longest_Missing_Streak": longest_streak,
            "First_10_Days_All_Missing": first10_missing
        })

        print(
            f"{variable}: {status} | "
            f"missing={missing_count}/{total_rows} | "
            f"longest_streak={longest_streak}"
        )


# ============================================================
# 6. CREATE REPORT
# ============================================================

availability_df = pd.DataFrame(
    availability_rows
)


# ============================================================
# 7. SAVE DETAILED REPORT
# ============================================================

DETAILED_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_feature_availability.csv"
)

availability_df.to_csv(
    DETAILED_FILE,
    index=False
)


# ============================================================
# 8. STATUS SUMMARY
# ============================================================

status_summary = (
    availability_df[
        "Status"
    ]
    .value_counts()
    .rename_axis(
        "Status"
    )
    .reset_index(
        name="Count"
    )
)


SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_feature_availability_summary.csv"
)

status_summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 9. EXCLUDED FEATURES
# ============================================================

excluded_df = availability_df[
    availability_df["Status"]
    ==
    "Excluded_First_10_Days_Missing"
].copy()


EXCLUDED_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_excluded_features.csv"
)

excluded_df.to_csv(
    EXCLUDED_FILE,
    index=False
)


# ============================================================
# 10. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE AVAILABILITY SUMMARY")
print("=" * 70)

print(
    status_summary.to_string(
        index=False
    )
)

print("\nExcluded features:")
print(
    f"{len(excluded_df)}"
)

if len(excluded_df) > 0:

    print(
        excluded_df[
            [
                "Participant",
                "Variable",
                "Missing_Count",
                "Total_Rows",
                "Longest_Missing_Streak"
            ]
        ].to_string(
            index=False
        )
    )

print("\nOutput files:")
print(
    f"  {DETAILED_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)

print(
    f"  {EXCLUDED_FILE}"
)

print("\n" + "=" * 70)
print("FEATURE AVAILABILITY ANALYSIS COMPLETE")
print("=" * 70)
