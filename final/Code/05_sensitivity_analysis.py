import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


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

DEVIATION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "baseline",
    "daily_personalized_deviations.csv"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "pmdata"
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

Z_THRESHOLD = 1.0
MIN_N = 10
BASELINE_FRACTION = 0.50
MIN_BASELINE_N = 7


# ============================================================
# 3. VARIABLES
# ============================================================

WELLBEING_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
]

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
# 5. CHECK REQUIRED PATHS
# ============================================================

if not os.path.exists(DEVIATION_FILE):
    raise FileNotFoundError(
        f"Personalized deviation file not found:\n"
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

print("Loading personalized deviations...")

deviations = pd.read_csv(
    DEVIATION_FILE
)

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
# 7. LOAD WELLBEING DATA
# ============================================================

print("Loading wellbeing data...")

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
    ).reset_index(drop=True)

    n_total = len(df)

    if n_total < 2:
        continue

    # Same chronological 50/50 split
    # used in the other final analyses.
    baseline_n = int(
        np.floor(
            n_total * BASELINE_FRACTION
        )
    )

    if baseline_n < MIN_BASELINE_N:
        continue

    # Second 50% = analysis period
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

wellbeing = wellbeing.dropna(
    subset=["Date"]
).copy()


# ============================================================
# 8. MERGE
# ============================================================

print(
    "Matching behavioral deviations "
    "with same-day wellbeing..."
)

merged = pd.merge(
    deviations,
    wellbeing,
    on=[
        "Participant",
        "Date",
    ],
    how="inner"
)

if merged.empty:
    raise RuntimeError(
        "No matching participant-date rows found."
    )

merged = merged.sort_values(
    [
        "Participant",
        "Date",
    ]
).reset_index(drop=True)


# ============================================================
# 9. SENSITIVITY ANALYSIS
# ============================================================

print(
    f"Running sensitivity analysis "
    f"with |Z| >= {Z_THRESHOLD}..."
)

results = []


for participant in sorted(
    merged["Participant"]
    .dropna()
    .unique()
):

    participant_data = merged[
        merged["Participant"] == participant
    ].copy()

    for behavior in BEHAVIOR_VARIABLES:

        z_column = f"{behavior}_Z"

        if z_column not in participant_data.columns:
            continue

        # ----------------------------------------------------
        # Select unusual behavioral days
        # ----------------------------------------------------

        behavior_data = participant_data[
            participant_data[z_column].abs()
            >= Z_THRESHOLD
        ].copy()

        if behavior_data.empty:
            continue

        for wellbeing in WELLBEING_VARIABLES:

            if wellbeing not in behavior_data.columns:
                continue

            x = behavior_data[
                z_column
            ]

            y = behavior_data[
                wellbeing
            ]

            r, p, n = safe_pearson(
                x,
                y
            )

            results.append(
                {
                    "Participant": participant,
                    "Behavioral_Variable": behavior,
                    "Wellbeing_Variable": wellbeing,
                    "Z_Threshold": Z_THRESHOLD,
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
# 10. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

if results_df.empty:
    raise RuntimeError(
        "No sensitivity relationships were calculated."
    )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 12. SUMMARY
# ============================================================

included = results_df[
    results_df["Included"] == "Yes"
]

print()
print("=" * 60)
print("SENSITIVITY ANALYSIS COMPLETED")
print("=" * 60)

print(
    f"Z threshold: |Z| >= {Z_THRESHOLD}"
)

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
    f"Output saved to:\n"
    f"{OUTPUT_FILE}"
)

print("=" * 60)
