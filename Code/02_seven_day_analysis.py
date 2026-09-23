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
# │   └── 04_seven_day_analysis.py
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
    "seven_day"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "seven_day_history_wellbeing_relationships.csv"
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
HISTORY_DAYS = 7


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
# 6. LOAD PERSONALIZED DEVIATIONS
# ============================================================

print("=" * 70)
print("SEVEN-DAY HISTORY BEHAVIOR-WELLBEING ANALYSIS")
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
    f"History window: previous {HISTORY_DAYS} days"
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
# 9. BUILD WELLBEING DATA
# ============================================================

print(
    "Loading participant files..."
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
    # Same chronological split as baseline code
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
# 10. CREATE PREVIOUS 7-DAY BEHAVIOR HISTORY
# ============================================================

print(
    "Calculating previous 7-day behavioral averages..."
)

history_records = []

for participant in sorted(
    deviations["Participant"]
    .dropna()
    .unique()
):

    participant_data = deviations[
        deviations["Participant"] == participant
    ].copy()

    participant_data = participant_data.sort_values(
        "Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Use raw behavioral values from the deviation file.
    # The history excludes the current day.
    # --------------------------------------------------------

    for behavior in BEHAVIOR_VARIABLES:

        value_column = f"{behavior}_Value"

        if value_column not in participant_data.columns:
            continue

        participant_data[
            value_column
        ] = pd.to_numeric(
            participant_data[value_column],
            errors="coerce"
        )

        participant_data[
            f"{behavior}_7day_mean"
        ] = (
            participant_data[value_column]
            .shift(1)
            .rolling(
                window=HISTORY_DAYS,
                min_periods=HISTORY_DAYS
            )
            .mean()
        )

    history_records.append(
        participant_data[
            [
                "Participant",
                "Date",
            ]
            + [
                f"{behavior}_7day_mean"
                for behavior in BEHAVIOR_VARIABLES
                if f"{behavior}_7day_mean"
                in participant_data.columns
            ]
        ]
    )


if not history_records:
    raise RuntimeError(
        "No behavioral history could be created."
    )


history = pd.concat(
    history_records,
    ignore_index=True
)

history["Date"] = pd.to_datetime(
    history["Date"],
    errors="coerce"
)

history = history.dropna(
    subset=["Date"]
).copy()


# ============================================================
# 11. MERGE 7-DAY HISTORY WITH SAME-DAY WELLBEING
# ============================================================

print(
    "Merging previous 7-day behavior history "
    "with same-day wellbeing..."
)

merged = pd.merge(
    history,
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
# 12. PARTICIPANT-LEVEL 7-DAY ANALYSIS
# ============================================================

print(
    "Calculating participant-level "
    "seven-day correlations..."
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

        history_column = (
            f"{behavior}_7day_mean"
        )

        if history_column not in participant_data.columns:
            continue

        for wellbeing_variable in WELLBEING_VARIABLES:

            if wellbeing_variable not in participant_data.columns:
                continue

            x = participant_data[
                history_column
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
print("SEVEN-DAY ANALYSIS COMPLETED")
print("=" * 70)

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
    "History definition: "
    "previous 7 days -> same-day wellbeing"
)

print(
    f"Output saved to:\n"
    f"{OUTPUT_FILE}"
)

print("=" * 70)
