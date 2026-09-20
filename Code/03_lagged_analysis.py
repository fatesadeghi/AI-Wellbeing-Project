import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


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

DEVIATION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "baseline",
    "daily_personalized_deviations.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "lagged"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "lagged_behavior_wellbeing_relationships.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

MIN_N = 10
BASELINE_FRACTION = 0.50
MIN_BASELINE_N = 7


# ============================================================
# 3. VARIABLES
# ============================================================

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
# 4. SAFE PEARSON CORRELATION
# ============================================================

def safe_pearson(x, y):
    """
    Calculate Pearson correlation after pairwise removal
    of missing and non-finite values.

    Returns:
        r, p, n
    """

    data = pd.concat(
        [
            pd.Series(x, name="x"),
            pd.Series(y, name="y"),
        ],
        axis=1,
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    n = len(data)

    if n < MIN_N:
        return np.nan, np.nan, n

    x_values = data["x"].astype(float).to_numpy()
    y_values = data["y"].astype(float).to_numpy()

    if not np.isfinite(x_values).all():
        return np.nan, np.nan, n

    if not np.isfinite(y_values).all():
        return np.nan, np.nan, n

    if np.std(x_values, ddof=1) == 0:
        return np.nan, np.nan, n

    if np.std(y_values, ddof=1) == 0:
        return np.nan, np.nan, n

    try:
        r, p = pearsonr(
            x_values,
            y_values
        )

        return r, p, n

    except Exception:
        return np.nan, np.nan, n


# ============================================================
# 5. LOAD PERSONALIZED DEVIATIONS
# ============================================================

if not os.path.exists(DEVIATION_FILE):
    raise FileNotFoundError(
        f"Personalized deviation file not found:\n"
        f"{DEVIATION_FILE}"
    )

print("=" * 60)
print("LAGGED BEHAVIOR-WELLBEING ANALYSIS")
print("=" * 60)

print(
    f"Project directory: {BASE_DIR}"
)

print(
    f"Data directory: {DATA_DIR}"
)

print(
    f"Deviation file: {DEVIATION_FILE}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)

print()
print("Loading personalized daily deviations...")

deviations = pd.read_csv(
    DEVIATION_FILE
)

if "Participant" not in deviations.columns:
    raise ValueError(
        "Column 'Participant' is missing from "
        "daily_personalized_deviations.csv"
    )

if "Date" not in deviations.columns:
    raise ValueError(
        "Column 'Date' is missing from "
        "daily_personalized_deviations.csv"
    )

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
# 6. LOAD WELLBEING DATA
# ============================================================

print("Loading participant files...")

all_wellbeing = []

participant_files = sorted(
    [
        filename
        for filename in os.listdir(DATA_DIR)
        if filename.endswith("_daily_merged.csv")
    ]
)

if not participant_files:
    raise RuntimeError(
        f"No participant files found in:\n{DATA_DIR}"
    )


for filename in participant_files:

    participant = filename.replace(
        "_daily_merged.csv",
        ""
    )

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(
        filepath
    )

    if "Date" not in df.columns:
        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    ).copy()

    df = df.sort_values(
        "Date"
    )

    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    )

    n_total = len(df)

    if n_total < 2:
        continue

    # First 50% = personal baseline period
    baseline_n = int(
        np.floor(
            n_total * BASELINE_FRACTION
        )
    )

    if baseline_n < MIN_BASELINE_N:
        continue

    # Second 50% = analysis period
    analysis_df = df.iloc[baseline_n:].copy()

    available_wellbeing = [
        variable
        for variable in WELLBEING_VARIABLES
        if variable in analysis_df.columns
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

    all_wellbeing.append(
        selected
    )


if not all_wellbeing:
    raise RuntimeError(
        "No wellbeing data found for analysis."
    )


wellbeing = pd.concat(
    all_wellbeing,
    ignore_index=True
)

wellbeing["Date"] = pd.to_datetime(
    wellbeing["Date"],
    errors="coerce"
)

wellbeing = wellbeing.dropna(
    subset=["Date"]
).copy()


# ============================================================
# 7. CREATE NEXT-DAY WELLBEING
# ============================================================

print("Creating one-day lag alignment...")

# Behavior on day t
#        ↓
# Wellbeing on day t+1

wellbeing_next = wellbeing.copy()

wellbeing_next["Date"] = (
    wellbeing_next["Date"]
    - pd.Timedelta(days=1)
)

wellbeing_next = wellbeing_next.rename(
    columns={
        variable: f"{variable}_NextDay"
        for variable in WELLBEING_VARIABLES
        if variable in wellbeing_next.columns
    }
)


# ============================================================
# 8. MERGE BEHAVIOR DEVIATIONS WITH NEXT-DAY WELLBEING
# ============================================================

print(
    "Merging behavioral deviations "
    "with next-day wellbeing..."
)

merged = pd.merge(
    deviations,
    wellbeing_next,
    on=[
        "Participant",
        "Date",
    ],
    how="inner"
)

if merged.empty:
    raise RuntimeError(
        "Merged dataset is empty. "
        "Check participant IDs and dates."
    )

merged = merged.sort_values(
    [
        "Participant",
        "Date",
    ]
).reset_index(drop=True)


# ============================================================
# 9. PARTICIPANT-LEVEL LAGGED ANALYSIS
# ============================================================

print(
    "Calculating participant-level "
    "lagged correlations..."
)

results = []

participants = sorted(
    merged["Participant"]
    .dropna()
    .unique()
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

            next_day_column = (
                f"{wellbeing_variable}_NextDay"
            )

            if next_day_column not in participant_data.columns:
                continue

            x = participant_data[
                z_column
            ]

            y = participant_data[
                next_day_column
            ]

            r, p, n = safe_pearson(
                x,
                y
            )

            results.append(
                {
                    "Participant": participant,
                    "Objective_Variable": behavior,
                    "Wellness_Variable": wellbeing_variable,
                    "r": r,
                    "p": p,
                    "N": n,
                    "Time_Lag": "t_to_t+1",
                }
            )


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. SUMMARY
# ============================================================

total_relationships = len(
    results_df
)

valid_relationships = int(
    results_df["r"].notna().sum()
)

participants_analyzed = (
    results_df["Participant"]
    .nunique()
)

print()
print("=" * 60)
print("LAGGED ANALYSIS COMPLETED")
print("=" * 60)

print(
    f"Participants analyzed: "
    f"{participants_analyzed}"
)

print(
    f"Total relationships tested: "
    f"{total_relationships}"
)

print(
    f"Relationships with N >= {MIN_N}: "
    f"{valid_relationships}"
)

print(
    "Time lag: behavior at day t "
    "-> wellbeing at day t+1"
)

print(
    f"Output saved to:\n{OUTPUT_FILE}"
)

print("=" * 60)
