import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/pmdata"

BASELINE_FILE = "results/baseline/personalized_baselines.csv"
DEVIATION_FILE = "results/baseline/daily_personalized_deviations.csv"

OUTPUT_DIR = "results/same_day"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "same_day_behavior_wellbeing_relationships.csv"
)

MIN_N = 10

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
# HELPER FUNCTIONS
# ============================================================

def safe_pearson(x, y):
    """
    Calculate Pearson correlation safely.

    Returns:
        r, p, n
    """

    data = pd.concat(
        [pd.Series(x), pd.Series(y)],
        axis=1
    ).dropna()

    n = len(data)

    if n < MIN_N:
        return np.nan, np.nan, n

    x_values = data.iloc[:, 0].astype(float).values
    y_values = data.iloc[:, 1].astype(float).values

    # Pearson correlation cannot be calculated
    # when one variable has no variation.
    if np.std(x_values, ddof=1) == 0:
        return np.nan, np.nan, n

    if np.std(y_values, ddof=1) == 0:
        return np.nan, np.nan, n

    try:
        r, p = pearsonr(x_values, y_values)
        return r, p, n

    except Exception:
        return np.nan, np.nan, n


# ============================================================
# LOAD DATA
# ============================================================

print("Loading personalized baseline data...")

baselines = pd.read_csv(BASELINE_FILE)

print("Loading personalized daily deviations...")

deviations = pd.read_csv(DEVIATION_FILE)


# ============================================================
# DATE PREPARATION
# ============================================================

deviations["Date"] = pd.to_datetime(
    deviations["Date"],
    errors="coerce"
)

deviations = deviations.dropna(
    subset=["Date"]
).copy()

deviations = deviations.sort_values(
    ["Participant", "Date"]
).reset_index(drop=True)


# ============================================================
# BUILD SAME-DAY WELLBEING DATA
# ============================================================

print("Loading participant files...")

all_wellbeing = []

participant_files = [
    f for f in os.listdir(DATA_DIR)
    if f.endswith("_daily_merged.csv")
]

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
            f"Skipping {participant}: "
            "Date column not found."
        )
        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    ).copy()

    df = df.sort_values("Date")

    # Remove duplicate dates.
    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    )

    # --------------------------------------------------------
    # Use the same chronological 50/50 split as baseline code
    # --------------------------------------------------------

    n_total = len(df)

    if n_total < 2:
        continue

    baseline_n = int(
        np.floor(n_total * 0.50)
    )

    if baseline_n < 7:
        continue

    analysis_df = df.iloc[baseline_n:].copy()

    # Keep only wellbeing variables that actually exist.
    available_wellbeing = [
        v for v in WELLBEING_VARIABLES
        if v in analysis_df.columns
    ]

    if not available_wellbeing:
        continue

    selected = analysis_df[
        ["Date"] + available_wellbeing
    ].copy()

    selected.insert(
        0,
        "Participant",
        participant
    )

    all_wellbeing.append(selected)


if not all_wellbeing:
    raise RuntimeError(
        "No wellbeing data could be loaded "
        "for the analysis period."
    )


wellbeing = pd.concat(
    all_wellbeing,
    ignore_index=True
)

wellbeing["Date"] = pd.to_datetime(
    wellbeing["Date"]
)


# ============================================================
# MERGE BEHAVIORAL DEVIATIONS WITH SAME-DAY WELLBEING
# ============================================================

print("Merging behavioral deviations with same-day wellbeing...")

merged = pd.merge(
    deviations,
    wellbeing,
    on=["Participant", "Date"],
    how="inner"
)


if merged.empty:
    raise RuntimeError(
        "The merge produced no matching participant-date rows."
    )


# ============================================================
# ANALYSIS
# ============================================================

print("Calculating participant-level same-day correlations...")

results = []

participants = sorted(
    merged["Participant"].dropna().unique()
)

for participant in participants:

    participant_data = merged[
        merged["Participant"] == participant
    ].copy()

    for behavior in BEHAVIOR_VARIABLES:

        z_column = f"{behavior}_Z"

        if z_column not in participant_data.columns:
            continue

        for wellbeing_variable in WELLBEING_VARIABLES:

            if wellbeing_variable not in participant_data.columns:
                continue

            x = participant_data[z_column]
            y = participant_data[wellbeing_variable]

            r, p, n = safe_pearson(x, y)

            results.append({
                "Participant": participant,
                "Behavioral_Variable": behavior,
                "Wellbeing_Variable": wellbeing_variable,
                "r": r,
                "p": p,
                "N": n,
                "Included": "Yes" if n >= MIN_N else "No"
            })


results_df = pd.DataFrame(results)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

included = results_df[
    results_df["Included"] == "Yes"
]

print()
print("=" * 60)
print("SAME-DAY ANALYSIS COMPLETED")
print("=" * 60)

print(
    f"Participants analyzed: "
    f"{results_df['Participant'].nunique()}"
)

print(
    f"Total relationships tested: "
    f"{len(results_df)}"
)

print(
    f"Relationships with N >= {MIN_N}: "
    f"{len(included)}"
)

print(
    f"Output saved to:\n{OUTPUT_FILE}"
)

print("=" * 60)
