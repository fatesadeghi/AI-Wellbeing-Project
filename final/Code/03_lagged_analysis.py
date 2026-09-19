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

BEHAVIOR_VARS = [
    "Steps",
    "Sleep_Duration_Score",
    "Sleep_Score",
    "Sleep_Deep_Minutes",
    "Sleep_Restlessness",
    "Exercise_Distance",
    "Exercise_Calories",
    "Exercise_Duration",
]

WELLBEING_VARS = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
]


# ============================================================
# 3. SAFE PEARSON CORRELATION
# ============================================================

def safe_pearson(x, y, min_n=MIN_N):
    """
    Calculate Pearson correlation safely.

    Returns:
        r, p, n
    """

    temp = pd.DataFrame({
        "x": x,
        "y": y
    })

    temp = (
        temp
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    n = len(temp)

    if n < min_n:
        return np.nan, np.nan, n

    x_values = temp["x"].to_numpy(
        dtype=float
    )

    y_values = temp["y"].to_numpy(
        dtype=float
    )

    # Prevent invalid Pearson calculations.
    if not np.isfinite(x_values).all():
        return np.nan, np.nan, n

    if not np.isfinite(y_values).all():
        return np.nan, np.nan, n

    # Pearson correlation is undefined
    # when either variable is constant.
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
# 4. LOAD PERSONALIZED DEVIATIONS
# ============================================================

print(
    "Loading personalized deviations..."
)

deviations = pd.read_csv(
    DEVIATION_FILE,
    parse_dates=["Date"]
)

deviations["Date"] = pd.to_datetime(
    deviations["Date"],
    errors="coerce"
)

deviations = (
    deviations
    .dropna(subset=["Date"])
    .sort_values(
        ["Participant", "Date"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 5. LOAD WELLBEING DATA
# ============================================================

print(
    "Loading participant wellbeing data..."
)

all_wellbeing = []

participant_files = sorted(
    [
        f
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
    ]
)

for filename in participant_files:

    participant = os.path.splitext(
        filename
    )[0]

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

    df = (
        df
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    n_total = len(df)

    if n_total == 0:
        continue

    # --------------------------------------------------------
    # Same chronological 50/50 split as baseline construction
    # --------------------------------------------------------

    baseline_n = int(
        np.floor(
            n_total * BASELINE_FRACTION
        )
    )

    if baseline_n < MIN_BASELINE_N:
        print(
            f"Skipping {participant}: "
            f"baseline N={baseline_n} < "
            f"{MIN_BASELINE_N}"
        )
        continue

    analysis_df = df.iloc[
        baseline_n:
    ].copy()

    if analysis_df.empty:
        continue

    # --------------------------------------------------------
    # Keep available wellbeing variables
    # --------------------------------------------------------

    available_wellbeing = [
        variable
        for variable in WELLBEING_VARS
        if variable in analysis_df.columns
    ]

    if not available_wellbeing:
        print(
            f"Skipping {participant}: "
            "no wellbeing variables found."
        )
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
    wellbeing["Date"],
    errors="coerce"
)


# ============================================================
# 6. CREATE NEXT-DAY WELLBEING DATA
# ============================================================

print(
    "Creating next-day wellbeing variables..."
)

wellbeing_next = wellbeing.copy()

# Shift wellbeing backward by one day so that:
#
# behavioral deviation on day t
#        matches
# wellbeing on day t+1
#
# Example:
# wellbeing on Jan 10
# becomes Date = Jan 9
#
# Therefore it merges with behavior on Jan 9.

wellbeing_next["Date"] = (
    wellbeing_next["Date"]
    - pd.Timedelta(days=1)
)

wellbeing_next = wellbeing_next.rename(
    columns={
        variable: f"{variable}_NextDay"
        for variable in WELLBEING_VARS
        if variable in wellbeing_next.columns
    }
)


# ============================================================
# 7. MERGE
# ============================================================

print(
    "Matching behavioral deviation on day t "
    "with wellbeing on day t+1..."
)

merged = pd.merge(
    deviations,
    wellbeing_next,
    on=[
        "Participant",
        "Date"
    ],
    how="inner"
)

if merged.empty:
    raise RuntimeError(
        "No matching participant-date rows "
        "were found."
    )


# ============================================================
# 8. ANALYSIS
# ============================================================

print(
    "Calculating participant-level "
    "one-day lagged correlations..."
)

results = []

participants = sorted(
    merged[
        "Participant"
    ]
    .dropna()
    .unique()
)

for participant in participants:

    participant_data = merged[
        merged["Participant"] == participant
    ].copy()

    for behavior in BEHAVIOR_VARS:

        behavior_column = (
            f"{behavior}_Z"
        )

        if behavior_column not in participant_data.columns:
            continue

        for wellbeing_variable in WELLBEING_VARS:

            next_day_column = (
                f"{wellbeing_variable}_NextDay"
            )

            if (
                next_day_column
                not in participant_data.columns
            ):
                continue

            x = participant_data[
                behavior_column
            ]

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


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

if results_df.empty:
    raise RuntimeError(
        "No lagged analysis results were generated."
    )

results_df = results_df.sort_values(
    [
        "Participant",
        "Behavioral_Variable",
        "Wellbeing_Variable"
    ]
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 10. SUMMARY
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
