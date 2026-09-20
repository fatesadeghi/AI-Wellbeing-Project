import os
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

# Current file:
# AI-Wellbeing-Project/ML/Code/08_prepare_ml_data.py

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

WELLBEING_VARS = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]

FIRST_10_DAYS = 10


# ============================================================
# 3. FIRST 10 DAYS MISSINGNESS RULE
# ============================================================

def first_10_days_missing(df, variable):
    """
    Check whether a behavioral variable is completely
    missing during the first 10 consecutive days.

    If all first 10 days are missing, the variable is
    excluded for that participant.

    Sleep variables are exempt from this rule.
    """

    check = df[
        ["Date", variable]
    ].copy()

    check = check.sort_values(
        "Date"
    ).head(
        FIRST_10_DAYS
    )

    if len(check) < FIRST_10_DAYS:
        return False

    return check[variable].isna().all()


# ============================================================
# 4. FIND PARTICIPANT FILES
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
print("ML DATA PREPARATION")
print("=" * 70)

print(
    f"Participant files found: {len(participant_files)}"
)


# ============================================================
# 5. PROCESS PARTICIPANTS
# ============================================================

all_ml_data = []

feature_status_rows = []


for filename in participant_files:

    participant = os.path.splitext(
        filename
    )[0]

    file_path = os.path.join(
        DATA_DIR,
        filename
    )

    print("\n" + "-" * 70)
    print(f"Processing: {participant}")

    df = pd.read_csv(
        file_path
    )

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

    df = df.dropna(
        subset=["Date"]
    )

    df = df.sort_values(
        "Date"
    )

    df = df.drop_duplicates(
        subset="Date",
        keep="first"
    )

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Determine active behavioral variables
    # --------------------------------------------------------

    active_behavior_vars = []

    for variable in BEHAVIOR_VARS:

        # ----------------------------------------------------
        # Missing column
        # ----------------------------------------------------

        if variable not in df.columns:

            feature_status_rows.append({
                "Participant": participant,
                "Variable": variable,
                "Status": "Missing_Column",
                "First_10_Days_All_Missing": False,
                "Total_Missing": np.nan,
                "Total_Rows": len(df),
                "Missing_Percent": np.nan
            })

            print(
                f"  {variable}: MISSING COLUMN"
            )

            continue

        # ----------------------------------------------------
        # Sleep variables are exempt from first-10 rule
        # ----------------------------------------------------

        is_sleep_variable = (
            variable.startswith("Sleep_")
            or variable == "Deep_Sleep_Minutes"
        )

        total_missing = int(
            df[variable].isna().sum()
        )

        total_rows = len(df)

        missing_percent = (
            total_missing / total_rows * 100
            if total_rows > 0
            else np.nan
        )

        if is_sleep_variable:

            first10_missing = False

            status = (
                "Available_Sleep_Exempt"
            )

            active_behavior_vars.append(
                variable
            )

        else:

            first10_missing = first_10_days_missing(
                df,
                variable
            )

            if first10_missing:

                status = (
                    "Excluded_First_10_Days_Missing"
                )

            else:

                status = "Available"

                active_behavior_vars.append(
                    variable
                )

        # ----------------------------------------------------
        # Save status
        # ----------------------------------------------------

        feature_status_rows.append({
            "Participant": participant,
            "Variable": variable,
            "Status": status,
            "First_10_Days_All_Missing": first10_missing,
            "Total_Missing": total_missing,
            "Total_Rows": total_rows,
            "Missing_Percent": missing_percent
        })

        print(
            f"  {variable}: {status}"
        )

    # --------------------------------------------------------
    # Select wellbeing variables
    # --------------------------------------------------------

    available_wellbeing_vars = [
        variable
        for variable in WELLBEING_VARS
        if variable in df.columns
    ]

    missing_wellbeing = [
        variable
        for variable in WELLBEING_VARS
        if variable not in df.columns
    ]

    if missing_wellbeing:

        print(
            f"  WARNING: Missing wellbeing columns: "
            f"{missing_wellbeing}"
        )

    # --------------------------------------------------------
    # Build participant ML dataframe
    # --------------------------------------------------------

    selected_columns = [
        "Date"
    ]

    selected_columns.extend(
        active_behavior_vars
    )

    selected_columns.extend(
        available_wellbeing_vars
    )

    ml_df = df[
        selected_columns
    ].copy()

    ml_df.insert(
        0,
        "Participant",
        participant
    )

    # --------------------------------------------------------
    # Create next-day wellbeing change targets
    #
    # Target at day t:
    #
    # Wellbeing(t+1) - Wellbeing(t)
    #
    # This is the prediction target for ML.
    # --------------------------------------------------------

    for variable in WELLBEING_VARS:

        if variable not in ml_df.columns:
            continue

        ml_df[
            f"NextDayChange_{variable}"
        ] = (
            ml_df[variable].shift(-1)
            -
            ml_df[variable]
        )

    all_ml_data.append(
        ml_df
    )

    print(
        f"  Active behavioral variables: "
        f"{len(active_behavior_vars)}/{len(BEHAVIOR_VARS)}"
    )

    print(
        f"  Wellbeing variables available: "
        f"{len(available_wellbeing_vars)}/{len(WELLBEING_VARS)}"
    )


# ============================================================
# 6. CHECK PROCESSING
# ============================================================

if not all_ml_data:

    raise RuntimeError(
        "No participant data could be prepared for ML."
    )


# ============================================================
# 7. COMBINE ALL PARTICIPANTS
# ============================================================

ml_data = pd.concat(
    all_ml_data,
    ignore_index=True
)

feature_status = pd.DataFrame(
    feature_status_rows
)


# ============================================================
# 8. SAVE PREPARED ML DATA
# ============================================================

ML_DATA_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_prepared_data.csv"
)

FEATURE_STATUS_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_feature_status.csv"
)

ml_data.to_csv(
    ML_DATA_FILE,
    index=False
)

feature_status.to_csv(
    FEATURE_STATUS_FILE,
    index=False
)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ML DATA PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Rows prepared: {len(ml_data)}"
)

print(
    f"Participants: "
    f"{ml_data['Participant'].nunique()}"
)

print(
    f"Behavioral variables: "
    f"{len(BEHAVIOR_VARS)}"
)

print(
    f"Wellbeing variables: "
    f"{len(WELLBEING_VARS)}"
)

print("\nOutput files:")
print(
    f"  {ML_DATA_FILE}"
)

print(
    f"  {FEATURE_STATUS_FILE}"
)

print("\n" + "=" * 70)
