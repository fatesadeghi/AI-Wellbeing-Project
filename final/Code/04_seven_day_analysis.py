import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/pmdata"

OUTPUT_DIR = "results/seven_day"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "seven_day_history_wellbeing_relationships.csv"
)

MIN_N = 10
MIN_HISTORY_DAYS = 7

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
# LOAD PARTICIPANT FILES
# ============================================================

print("Loading participant data...")

participant_files = [
    f
    for f in os.listdir(DATA_DIR)
    if f.endswith("_daily_merged.csv")
]

if not participant_files:
    raise RuntimeError(
        "No participant *_daily_merged.csv files found."
    )


# ============================================================
# ANALYSIS
# ============================================================

results = []

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
        f"Processing participant: {participant}"
    )

    df = pd.read_csv(filepath)

    if "Date" not in df.columns:
        print(
            f"Skipping {participant}: "
            "Date column not found."
        )
        continue

    # --------------------------------------------------------
    # DATE PREPARATION
    # --------------------------------------------------------

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
    ).reset_index(drop=True)

    n_total = len(df)

    if n_total < 2:
        continue

    # --------------------------------------------------------
    # SAME 50/50 SPLIT AS BASELINE CODE
    # --------------------------------------------------------

    baseline_n = int(
        np.floor(n_total * 0.50)
    )

    if baseline_n < 7:
        continue

    analysis_df = df.iloc[
        baseline_n:
    ].copy()

    if analysis_df.empty:
        continue

    # --------------------------------------------------------
    # CREATE 7-DAY HISTORY
    # --------------------------------------------------------

    for behavior in BEHAVIOR_VARIABLES:

        if behavior not in analysis_df.columns:
            continue

        # Previous 7 observations/days.
        #
        # shift(1):
        #   excludes the current day.
        #
        # rolling(7):
        #   uses the previous 7 observations.
        #
        # min_periods=7:
        #   requires a complete 7-day history.
        analysis_df[
            f"{behavior}_7day_history"
        ] = (
            analysis_df[behavior]
            .shift(1)
            .rolling(
                window=MIN_HISTORY_DAYS,
                min_periods=MIN_HISTORY_DAYS
            )
            .mean()
        )

    # --------------------------------------------------------
    # CALCULATE PARTICIPANT-LEVEL CORRELATIONS
    # --------------------------------------------------------

    for behavior in BEHAVIOR_VARIABLES:

        history_column = (
            f"{behavior}_7day_history"
        )

        if history_column not in analysis_df.columns:
            continue

        for wellbeing in WELLBEING_VARIABLES:

            if wellbeing not in analysis_df.columns:
                continue

            x = analysis_df[
                history_column
            ]

            y = analysis_df[
                wellbeing
            ]

            r, p, n = safe_pearson(
                x,
                y
            )

            results.append({
                "Participant": participant,
                "Behavioral_Variable": behavior,
                "Wellbeing_Variable": wellbeing,
                "Time_Window": "previous_7_days",
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
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:
    raise RuntimeError(
        "No relationships could be calculated."
    )


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
print("7-DAY HISTORY ANALYSIS COMPLETED")
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
