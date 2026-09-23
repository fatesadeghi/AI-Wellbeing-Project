import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Current project structure:
#
# AI-Wellbeing-Project/
# ├── Code/
# │   └── 05_sensitivity_analysis.py
# ├── data/
# │   └── pmdata/
# └── results/
#
# Therefore:
# dirname(__file__)          -> Code
# dirname(dirname(__file__)) -> AI-Wellbeing-Project

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

BASELINE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "baseline",
    "personalized_baselines.csv"
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
    "sensitivity"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sensitivity_Z1_same_day_relationships.csv"
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

# Sensitivity threshold:
# A day is considered an unusual/deviation day when
# at least one behavioral variable has |Z| >= 1.0.
SENSITIVITY_Z_THRESHOLD = 1.0


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
    Calculate Pearson correlation safely.

    Missing, infinite, and constant values are handled
    before calculating the correlation.

    Returns:
        r, p, n
    """

    data = pd.concat(
        [
            pd.Series(x, name="x"),
            pd.Series(y, name="y"),
        ],
        axis=1
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
# 5. CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(BASELINE_FILE):
    raise FileNotFoundError(
        f"Personalized baseline file not found:\n"
        f"{BASELINE_FILE}"
    )

if not os.path.exists(DEVIATION_FILE):
    raise FileNotFoundError(
        f"Daily personalized deviation file not found:\n"
        f"{DEVIATION_FILE}"
    )

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"PMData directory not found:\n"
        f"{DATA_DIR}"
    )


# ============================================================
# 6. LOAD BASELINE AND DAILY DEVIATIONS
# ============================================================

print("=" * 70)
print("SENSITIVITY ANALYSIS")
print("=" * 70)

print(
    f"Project directory: {BASE_DIR}"
)

print(
    f"Data directory: {DATA_DIR}"
)

print(
    f"Baseline file: {BASELINE_FILE}"
)

print(
    f"Deviation file: {DEVIATION_FILE}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)

print(
    f"Sensitivity threshold: |Z| >= "
    f"{SENSITIVITY_Z_THRESHOLD}"
)

print()

print(
    "Loading personalized baseline data..."
)

baselines = pd.read_csv(
    BASELINE_FILE
)

print(
    "Loading personalized daily deviations..."
)

deviations = pd.read_csv(
    DEVIATION_FILE
)


# ============================================================
# 7. VALIDATE DEVIATION DATA
# ============================================================

required_deviation_columns = [
    "Participant",
    "Date",
]

missing_columns = [
    column
    for column in required_deviation_columns
    if column not in deviations.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns in deviation file: "
        + ", ".join(missing_columns)
    )


# ============================================================
# 8. DATE PREPARATION
# ============================================================

deviations["Date"] = pd.to_datetime(
    deviations["Date"],
    errors="coerce"
)

deviations = deviations.dropna(
    subset=["Date"]
).copy()

deviations = deviations.sort_values(
    [
        "Participant",
        "Date",
    ]
).reset_index(drop=True)


# ============================================================
# 9. IDENTIFY SENSITIVITY DAYS
# ============================================================

print(
    "Identifying days with behavioral "
    "deviations at |Z| >= 1.0..."
)

z_columns = [
    f"{behavior}_Z"
    for behavior in BEHAVIOR_VARIABLES
]

available_z_columns = [
    column
    for column in z_columns
    if column in deviations.columns
]

if not available_z_columns:
    raise RuntimeError(
        "No behavioral Z-score columns were found "
        "in the deviation file."
    )

# A day is selected if at least one available
# behavioral variable has |Z| >= threshold.

abs_z = deviations[
    available_z_columns
].abs()

deviations["Sensitivity_Day"] = (
    abs_z >= SENSITIVITY_Z_THRESHOLD
).any(
    axis=1
)

sensitivity_days = deviations[
    deviations["Sensitivity_Day"]
].copy()

print(
    f"Total analysis-period rows: "
    f"{len(deviations)}"
)

print(
    f"Sensitivity rows selected: "
    f"{len(sensitivity_days)}"
)


if sensitivity_days.empty:
    raise RuntimeError(
        "No days met the sensitivity threshold."
    )


# ============================================================
# 10. LOAD SAME-DAY WELLBEING DATA
# ============================================================

print(
    "Loading participant wellbeing data..."
)

all_wellbeing = []

participant_files = sorted(
    [
        filename
        for filename in os.listdir(DATA_DIR)
        if filename.endswith(
            "_daily_merged.csv"
        )
    ]
)

if not participant_files:
    raise RuntimeError(
        f"No participant files found in:\n"
        f"{DATA_DIR}"
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

    df = df.sort_values(
        "Date"
    )

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
        np.floor(
            n_total * BASELINE_FRACTION
        )
    )

    if baseline_n < MIN_BASELINE_N:
        continue

    # Second half = analysis period
    analysis_df = df.iloc[
        baseline_n:
    ].copy()

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
        "No wellbeing data could be loaded "
        "for the analysis period."
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
# 11. MERGE SENSITIVITY DAYS WITH WELLBEING
# ============================================================

print(
    "Merging sensitivity days "
    "with same-day wellbeing..."
)

merged = pd.merge(
    sensitivity_days,
    wellbeing,
    on=[
        "Participant",
        "Date",
    ],
    how="inner"
)

if merged.empty:
    raise RuntimeError(
        "The merge produced no matching "
        "participant-date rows."
    )

merged = merged.sort_values(
    [
        "Participant",
        "Date",
    ]
).reset_index(drop=True)


# ============================================================
# 12. PARTICIPANT-LEVEL SENSITIVITY ANALYSIS
# ============================================================

print(
    "Calculating participant-level "
    "sensitivity correlations..."
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

            if wellbeing_variable not in participant_data.columns:
                continue

            x = participant_data[
                z_column
            ]

            y = participant_data[
                wellbeing_variable
            ]

            r, p, n = safe_pearson(
                x,
                y
            )

            results.append(
                {
                    "Participant": participant,
                    "Behavioral_Variable": behavior,
                    "Wellbeing_Variable": wellbeing_variable,
                    "r": r,
                    "p": p,
                    "N": n,
                    "Included": (
                        "Yes"
                        if n >= MIN_N
                        else "No"
                    ),
                    "Sensitivity_Threshold": (
                        SENSITIVITY_Z_THRESHOLD
                    ),
                }
            )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. SUMMARY
# ============================================================

included = results_df[
    results_df["Included"] == "Yes"
]

print()
print("=" * 70)
print("SENSITIVITY ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Participants analyzed: "
    f"{results_df['Participant'].nunique()}"
)

print(
    f"Total sensitivity relationships tested: "
    f"{len(results_df)}"
)

print(
    f"Relationships with N >= {MIN_N}: "
    f"{len(included)}"
)

print(
    f"Sensitivity threshold: "
    f"|Z| >= {SENSITIVITY_Z_THRESHOLD}"
)

print(
    "Analysis type: "
    "same-day participant-level Pearson correlation "
    "on sensitivity-selected days"
)

print(
    f"Output saved to:\n"
    f"{OUTPUT_FILE}"
)

print("=" * 70)
