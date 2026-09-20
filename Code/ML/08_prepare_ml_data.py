import os
import pandas as pd
import numpy as np


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

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. VARIABLES
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

WELLBEING_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]

MIN_BASELINE_N = 7


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def safe_zscore(value, mean, std):
    if pd.isna(value) or pd.isna(mean) or pd.isna(std):
        return np.nan

    if std == 0:
        return np.nan

    return (value - mean) / std


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() == 0:
        return np.nan
    return series.mean()


def safe_std(series):
    series = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() < 2:
        return np.nan
    return series.std()


# ============================================================
# 4. FIND PARTICIPANTS
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
print("ML DATA PREPARATION")
print("=" * 70)
print(f"Participants found: {len(participant_files)}")
print()


# ============================================================
# 5. PROCESS EACH PARTICIPANT
# ============================================================

all_participant_data = []

for filename in participant_files:

    participant = filename.replace(
        "_daily_merged.csv",
        ""
    )

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    print(f"Processing {participant}...")

    df = pd.read_csv(filepath)

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if "Date" not in df.columns:
        print(
            f"WARNING: {participant} has no Date column. Skipping."
        )
        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    ).sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    if len(df) == 0:
        print(
            f"WARNING: {participant} contains no valid dates. Skipping."
        )
        continue

    # --------------------------------------------------------
    # Participant ID
    # --------------------------------------------------------

    df["Participant"] = participant

    # --------------------------------------------------------
    # Check behavioral variables
    # --------------------------------------------------------

    missing_behavioral = [
        v
        for v in BEHAVIORAL_VARIABLES
        if v not in df.columns
    ]

    if missing_behavioral:
        print(
            f"WARNING: Missing columns for {participant}: "
            f"{missing_behavioral}"
        )

        # Create missing columns as NaN.
        # This keeps the feature structure identical
        # across participants.
        for variable in missing_behavioral:
            df[variable] = np.nan

    # --------------------------------------------------------
    # Check wellbeing variables
    # --------------------------------------------------------

    missing_wellbeing = [
        v
        for v in WELLBEING_VARIABLES
        if v not in df.columns
    ]

    if missing_wellbeing:
        print(
            f"WARNING: Missing wellbeing columns for "
            f"{participant}: {missing_wellbeing}"
        )

        for variable in missing_wellbeing:
            df[variable] = np.nan

    # --------------------------------------------------------
    # Convert behavioral and wellbeing variables to numeric
    # --------------------------------------------------------

    for variable in BEHAVIORAL_VARIABLES:
        df[variable] = pd.to_numeric(
            df[variable],
            errors="coerce"
        )

    for variable in WELLBEING_VARIABLES:
        df[variable] = pd.to_numeric(
            df[variable],
            errors="coerce"
        )

    # ========================================================
    # 6. PERSONALIZED BASELINE
    # ========================================================

    n_total = len(df)

    baseline_n = int(
        np.floor(n_total * 0.50)
    )

    baseline_n = max(
        baseline_n,
        MIN_BASELINE_N
    )

    baseline_n = min(
        baseline_n,
        n_total
    )

    baseline = df.iloc[
        :baseline_n
    ].copy()

    print(
        f"  Total observations: {n_total}"
    )

    print(
        f"  Baseline observations: {baseline_n}"
    )

    # --------------------------------------------------------
    # Baseline statistics
    # --------------------------------------------------------

    baseline_stats = {}

    for variable in BEHAVIORAL_VARIABLES:

        series = baseline[variable]

        baseline_stats[variable] = {
            "median": series.median(),
            "mean": safe_mean(series),
            "std": safe_std(series)
        }

    # ========================================================
    # 7. BEHAVIORAL ML FEATURES
    # ========================================================

    for variable in BEHAVIORAL_VARIABLES:

        median = baseline_stats[variable]["median"]
        mean = baseline_stats[variable]["mean"]
        std = baseline_stats[variable]["std"]

        # ----------------------------------------------------
        # Raw value
        # ----------------------------------------------------

        df[f"{variable}_raw"] = df[variable]

        # ----------------------------------------------------
        # Deviation from personalized baseline median
        # ----------------------------------------------------

        df[f"{variable}_deviation"] = (
            df[variable] - median
        )

        # ----------------------------------------------------
        # Personalized Z-score
        # ----------------------------------------------------

        df[f"{variable}_z"] = (
            df[variable].apply(
                lambda x: safe_zscore(
                    x,
                    mean,
                    std
                )
            )
        )

        # ----------------------------------------------------
        # Absolute Z-score
        # ----------------------------------------------------

        df[f"{variable}_abs_z"] = (
            df[f"{variable}_z"].abs()
        )

        # ----------------------------------------------------
        # Previous 7-day mean
        #
        # shift(1) ensures that the current day's
        # information is NOT included.
        # ----------------------------------------------------

        df[f"{variable}_7d_mean"] = (
            df[variable]
            .shift(1)
            .rolling(
                window=7,
                min_periods=1
            )
            .mean()
        )

        # ----------------------------------------------------
        # Previous 7-day standard deviation
        # ----------------------------------------------------

        df[f"{variable}_7d_std"] = (
            df[variable]
            .shift(1)
            .rolling(
                window=7,
                min_periods=2
            )
            .std()
        )

        # ----------------------------------------------------
        # One-day change
        # ----------------------------------------------------

        df[f"{variable}_1d_change"] = (
            df[variable].diff(1)
        )

    # ========================================================
    # 8. WELLBEING TARGET FEATURES
    # ========================================================

    for variable in WELLBEING_VARIABLES:

        # ----------------------------------------------------
        # Current wellbeing value
        # ----------------------------------------------------

        df[f"{variable}_raw"] = df[variable]

        # ----------------------------------------------------
        # Next-day wellbeing
        #
        # Used later for predictive/lagged modelling.
        # ----------------------------------------------------

        df[f"{variable}_next_day"] = (
            df[variable].shift(-1)
        )

        # ----------------------------------------------------
        # Change in wellbeing from current day to next day
        # ----------------------------------------------------

        df[f"{variable}_next_day_change"] = (
            df[variable].shift(-1)
            - df[variable]
        )

    # ========================================================
    # 9. STORE PARTICIPANT DATA
    # ========================================================

    all_participant_data.append(
        df
    )

    print(
        f"  Features created: "
        f"{len(df.columns)} columns"
    )

    print()


# ============================================================
# 10. COMBINE PARTICIPANTS
# ============================================================

print("=" * 70)
print("COMBINE PARTICIPANTS")
print("=" * 70)

if len(all_participant_data) == 0:

    raise RuntimeError(
        "No participant datasets were successfully processed."
    )

ml_dataset = pd.concat(
    all_participant_data,
    ignore_index=True,
    sort=False
)

print(
    f"Combined rows: {len(ml_dataset)}"
)

print(
    f"Combined columns: {len(ml_dataset.columns)}"
)

print(
    f"Participants in final dataset: "
    f"{ml_dataset['Participant'].nunique()}"
)


# ============================================================
# 11. ORDER COLUMNS
# ============================================================

first_columns = [
    "Participant",
    "Date"
]

remaining_columns = [
    c
    for c in ml_dataset.columns
    if c not in first_columns
]

ml_dataset = ml_dataset[
    first_columns + remaining_columns
]


# ============================================================
# 12. SAVE ML DATASET
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "ml_ready_dataset.csv"
)

ml_dataset.to_csv(
    output_file,
    index=False
)


# ============================================================
# 13. SAVE FEATURE LIST
# ============================================================

feature_rows = []

for variable in BEHAVIORAL_VARIABLES:

    feature_rows.extend(
        [
            {
                "Behavioral_Variable": variable,
                "Feature": f"{variable}_raw",
                "Feature_Type": "Raw value"
            },
            {
                "Behavioral_Variable": variable,
                "Feature": f"{variable}_deviation",
                "Feature_Type": "Deviation from baseline median"
            },
            {
                "Behavioral_Variable": variable,
                "Feature": f"{variable}_z",
                "Feature_Type": "Personalized Z-score"
            },
            {
                "Behavioral_Variable": variable,
                "Feature": f"{variable}_abs_z",
                "Feature_Type": "Absolute personalized Z-score"
            },
            {
                "Behavioral_Variable": variable,
                "Feature": f"{variable}_7d_mean",
                "Feature_Type": "Previous 7-day mean"
            },
            {
                "Behavioral_Variable": variable,
                "Feature": f"{variable}_7d_std",
                "Feature_Type": "Previous 7-day standard deviation"
            },
            {
                "Behavioral_Variable": variable,
                "Feature": f"{variable}_1d_change",
                "Feature_Type": "One-day change"
            }
        ]
    )

feature_list = pd.DataFrame(
    feature_rows
)

feature_file = os.path.join(
    OUTPUT_DIR,
    "ml_feature_list.csv"
)

feature_list.to_csv(
    feature_file,
    index=False
)


# ============================================================
# 14. SAVE PARTICIPANT SUMMARY
# ============================================================

participant_summary = (
    ml_dataset
    .groupby("Participant")
    .agg(
        Observations=("Date", "count")
    )
    .reset_index()
)

participant_file = os.path.join(
    OUTPUT_DIR,
    "ml_participant_summary.csv"
)

participant_summary.to_csv(
    participant_file,
    index=False
)


# ============================================================
# 15. FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("SAVE")
print("=" * 70)

print(
    f"Saved: {output_file}"
)

print(
    f"Saved: {feature_file}"
)

print(
    f"Saved: {participant_file}"
)

print()
print(
    "Behavioral variables:",
    len(BEHAVIORAL_VARIABLES)
)

print(
    "Well-being variables:",
    len(WELLBEING_VARIABLES)
)

print(
    "Participants:",
    ml_dataset["Participant"].nunique()
)

print(
    "Rows:",
    len(ml_dataset)
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
