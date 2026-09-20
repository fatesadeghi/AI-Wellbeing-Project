import os
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

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
    "ml"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. VARIABLES
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
    "Sleep_Score",
]

ACTIVITY_VARS = [
    "Steps",
    "Exercise_Count",
    "Exercise_Duration",
    "Exercise_Distance",
    "Exercise_Calories",
    "Exercise_Avg_HR",
]

SLEEP_VARS = [
    "Sleep_Hours",
    "Sleep_Duration_Score",
    "Deep_Sleep_Minutes",
    "Sleep_Restlessness",
    "Sleep_Composition",
    "Sleep_Revitalization",
    "Sleep_Score",
]


# ============================================================
# 3. FIND PARTICIPANTS
# ============================================================

participant_files = sorted(
    [
        f
        for f in os.listdir(DATA_DIR)
        if f.startswith("p")
        and f.endswith("_daily_merged.csv")
    ]
)

print("=" * 70)
print("FEATURE AVAILABILITY ANALYSIS")
print("=" * 70)
print(f"Participants found: {len(participant_files)}")


# ============================================================
# 4. CHECK FIRST 10 CONSECUTIVE DAYS
# ============================================================

rows = []

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
            f"WARNING: Date column missing for {participant}"
        )
        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    for variable in BEHAVIOR_VARS:

        # ----------------------------------------------------
        # Missing column
        # ----------------------------------------------------

        if variable not in df.columns:

            total_days = len(df)

            missing_days = total_days

            first_10_missing = (
                variable in ACTIVITY_VARS
                and total_days >= 10
            )

            longest_streak = total_days

        else:

            missing = df[variable].isna()

            total_days = len(df)

            missing_days = int(
                missing.sum()
            )

            # ------------------------------------------------
            # Longest consecutive missing streak
            # ------------------------------------------------

            groups = (
                missing
                .ne(missing.shift())
                .cumsum()
            )

            streaks = (
                missing
                .groupby(groups)
                .sum()
            )

            longest_streak = int(
                streaks.max()
            ) if len(streaks) else 0

            # ------------------------------------------------
            # FIRST 10 DAYS
            # ------------------------------------------------

            first_10_missing = (
                variable in ACTIVITY_VARS
                and total_days >= 10
                and missing.iloc[:10].all()
            )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if variable in SLEEP_VARS:

            status = "Available_Sleep_Exempt"

        elif first_10_missing:

            status = "Excluded_First_10_Days_Missing"

        elif missing_days == 0:

            status = "Available"

        elif longest_streak >= 10:

            status = "Available_With_Long_Missing_Streak"

        else:

            status = "Available_With_Missing"

        rows.append(
            {
                "Participant": participant,
                "Variable": variable,
                "Variable_Type": (
                    "Activity_Exercise"
                    if variable in ACTIVITY_VARS
                    else "Sleep"
                ),
                "Total_Days": total_days,
                "Missing_Days": missing_days,
                "Missing_Percentage": (
                    round(
                        100 * missing_days / total_days,
                        2
                    )
                    if total_days > 0
                    else 100.0
                ),
                "Longest_Missing_Streak": longest_streak,
                "First_10_Days_All_Missing": (
                    bool(first_10_missing)
                ),
                "Status": status,
            }
        )


# ============================================================
# 5. SAVE FULL AVAILABILITY TABLE
# ============================================================

availability = pd.DataFrame(rows)

output_file = os.path.join(
    OUTPUT_DIR,
    "ml_feature_availability.csv"
)

availability.to_csv(
    output_file,
    index=False
)


# ============================================================
# 6. SUMMARY
# ============================================================

print()
print("=" * 70)
print("AVAILABILITY SUMMARY")
print("=" * 70)

print(
    availability[
        "Status"
    ].value_counts()
)


# ============================================================
# 7. SHOW AUTOMATIC EXCLUSIONS
# ============================================================

excluded = availability[
    availability["Status"]
    == "Excluded_First_10_Days_Missing"
]

print()
print("=" * 70)
print("AUTOMATIC EXCLUSIONS")
print("=" * 70)

if excluded.empty:

    print("No activity/exercise variable meets the rule.")

else:

    print(
        excluded[
            [
                "Participant",
                "Variable",
                "Total_Days",
                "Missing_Days",
                "Longest_Missing_Streak",
                "Status",
            ]
        ].to_string(index=False)
    )


print()
print("=" * 70)
print(f"Saved: {output_file}")
print("=" * 70)
