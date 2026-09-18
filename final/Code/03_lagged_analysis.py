import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/pmdata"

DEVIATION_FILE = (
    "results/baseline/daily_personalized_deviations.csv"
)

OUTPUT_DIR = "results/lagged"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "lagged_behavior_wellbeing_relationships.csv"
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
# PEARSON FUNCTION
# ============================================================

def safe_pearson(x, y):

    data = pd.concat(
        [pd.Series(x), pd.Series(y)],
        axis=1
    ).dropna()

    n = len(data)

    if n < MIN_N:
        return np.nan, np.nan, n

    x_values = data.iloc[:, 0].astype(float).values
    y_values = data.iloc[:, 1].astype(float).values

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
# LOAD PERSONALIZED DEVIATIONS
# ============================================================

print("Loading personalized deviations...")

deviations = pd.read_csv(
    DEVIATION_FILE
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
# LOAD WELLBEING DATA
# ============================================================

print("Loading participant wellbeing data...")

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
        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    ).copy()

    df = df.sort_values("Date")

    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    )

    # --------------------------------------------------------
    # Same 50/50 chronological split as baseline construction
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

    available_wellbeing = [
        v
        for v in WELLBEING_VARIABLES
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

    all_wellbeing.append(
        selected
    )


if not all_wellbeing:
    raise RuntimeError(
        "No wellbeing data found."
    )


wellbeing = pd.concat(
    all_wellbeing,
    ignore_index=True
)

wellbeing["Date"] = pd.to_datetime(
    wellbeing["Date"]
)


# ============================================================
# CREATE NEXT-DAY WELLBEING DATA
# ============================================================

print(
    "Creating next-day wellbeing variables..."
)

# Rename Date to make the temporal direction explicit.
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
# MERGE
# ============================================================

print(
    "Matching behavioral deviation on day t "
    "with wellbeing on day t+1..."
)

merged = pd.merge(
    deviations,
    wellbeing_next,
    on=["Participant", "Date"],
    how="inner"
)


if merged.empty:
    raise RuntimeError(
        "No matching participant-date rows were found."
    )


# ============================================================
# ANALYSIS
# ============================================================

print(
    "Calculating participant-level "
    "one-day lagged correlations..."
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

            x = participant_data[z_column]

            y = participant_data[
                next_day_column
            ]

            r, p, n = safe_pearson(
                x,
                y
            )

            results.append({
                "Participant": participant,
                "Behavioral_Variable": behavior,
                "Wellbeing_Variable": wellbeing_variable,
                "Time_Lag": "t_to_t+1",
                "r": r,
                "p": p,
                "N": n,
                "Included": (
                    "Yes"
                    if n >= MIN_N
                    else "No"
                )
            })


results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
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
print("ONE-DAY LAGGED ANALYSIS COMPLETED")
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
