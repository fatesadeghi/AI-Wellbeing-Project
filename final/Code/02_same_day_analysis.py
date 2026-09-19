import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# This file is located at:
# AI-Wellbeing-Project/final/Code/02_same_day_analysis.py
#
# Therefore:
# dirname(__file__)              -> final/Code
# dirname(dirname(__file__))     -> final
# dirname(dirname(dirname(...))) -> AI-Wellbeing-Project

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
    "same_day"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "same_day_behavior_wellbeing_relationships.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

MIN_N = 10

BASELINE_FRACTION = 0.50
MIN_BASELINE_N = 7

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
# 3. CHECK REQUIRED PATHS
# ============================================================

print("=" * 70)
print("SAME-DAY BEHAVIOR–WELLBEING ANALYSIS")
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

print()


if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        f"\nData directory not found:\n{DATA_DIR}"
    )


if not os.path.exists(BASELINE_FILE):

    raise FileNotFoundError(
        f"\nBaseline file not found:\n{BASELINE_FILE}\n\n"
        "Run 01_build_baseline.py first."
    )


if not os.path.exists(DEVIATION_FILE):

    raise FileNotFoundError(
        f"\nDaily deviation file not found:\n"
        f"{DEVIATION_FILE}\n\n"
        "Run 01_build_baseline.py first."
    )


# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def safe_pearson(x, y):
    """
    Calculate Pearson correlation safely.

    Returns:
        r, p, n
    """

    data = pd.concat(
        [
            pd.Series(x),
            pd.Series(y)
        ],
        axis=1
    ).dropna()

    n = len(data)

    # Minimum sample-size requirement.
    if n < MIN_N:
        return np.nan, np.nan, n

    x_values = data.iloc[:, 0].astype(float).values
    y_values = data.iloc[:, 1].astype(float).values

    # Pearson correlation is undefined when
    # either variable has zero variance.
    x_sd = np.std(
        x_values,
        ddof=1
    )

    y_sd = np.std(
        y_values,
        ddof=1
    )

    if not np.isfinite(x_sd) or x_sd == 0:
        return np.nan, np.nan, n

    if not np.isfinite(y_sd) or y_sd == 0:
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
# 5. LOAD PERSONALIZED BASELINE DATA
# ============================================================

print("Loading personalized baseline data...")

baselines = pd.read_csv(
    BASELINE_FILE
)

print(
    f"Baseline records loaded: {len(baselines)}"
)


# ============================================================
# 6. LOAD DAILY PERSONALIZED DEVIATIONS
# ============================================================

print("Loading personalized daily deviations...")

deviations = pd.read_csv(
    DEVIATION_FILE
)

print(
    f"Daily deviation records loaded: {len(deviations)}"
)


# ============================================================
# 7. PREPARE DEVIATION DATES
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
# 8. LOAD WELLBEING DATA FROM REPOSITORY DATA
# ============================================================

print()
print("Loading participant wellbeing data from data/pmdata...")

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


print(
    f"Participant files found: {len(participant_files)}"
)


# ============================================================
# 9. PROCESS EACH PARTICIPANT
# ============================================================

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
        f"Processing {participant}..."
    )

    # --------------------------------------------------------
    # LOAD PARTICIPANT DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        filepath
    )

    # --------------------------------------------------------
    # CHECK DATE COLUMN
    # --------------------------------------------------------

    if "Date" not in df.columns:

        print(
            f"  Skipping {participant}: "
            "Date column not found."
        )

        continue

    # --------------------------------------------------------
    # PREPARE DATE
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
    ).reset_index(drop=True)

    # Remove duplicate dates using the same rule
    # as the baseline construction script.
    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # CHRONOLOGICAL 50/50 SPLIT
    # --------------------------------------------------------

    n_total = len(df)

    if n_total < 2:

        print(
            f"  Skipping {participant}: "
            f"only {n_total} valid days."
        )

        continue

    baseline_n = int(
        np.floor(
            n_total * BASELINE_FRACTION
        )
    )

    if baseline_n < MIN_BASELINE_N:

        print(
            f"  Skipping {participant}: "
            f"baseline has only {baseline_n} observations."
        )

        continue

    analysis_df = df.iloc[
        baseline_n:
    ].copy()

    if analysis_df.empty:

        print(
            f"  Skipping {participant}: "
            "no analysis-period observations."
        )

        continue

    # --------------------------------------------------------
    # SELECT WELLBEING VARIABLES
    # --------------------------------------------------------

    available_wellbeing = [
        variable
        for variable in WELLBEING_VARIABLES
        if variable in analysis_df.columns
    ]

    if not available_wellbeing:

        print(
            f"  Skipping {participant}: "
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


# ============================================================
# 10. COMBINE WELLBEING DATA
# ============================================================

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

wellbeing = wellbeing.sort_values(
    ["Participant", "Date"]
).reset_index(drop=True)


# ============================================================
# 11. MERGE DEVIATIONS WITH SAME-DAY WELLBEING
# ============================================================

print()
print(
    "Merging personalized behavioral deviations "
    "with same-day wellbeing..."
)

merged = pd.merge(
    deviations,
    wellbeing,
    on=[
        "Participant",
        "Date"
    ],
    how="inner"
)


if merged.empty:

    raise RuntimeError(
        "The merge produced no matching "
        "participant-date observations."
    )


print(
    f"Matched participant-day observations: "
    f"{len(merged)}"
)


# ============================================================
# 12. PARTICIPANT-LEVEL SAME-DAY CORRELATIONS
# ============================================================

print()
print(
    "Calculating participant-level "
    "same-day Pearson correlations..."
)

results = []

participants = sorted(
    merged[
        "Participant"
    ].dropna().unique()
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
                }
            )


# ============================================================
# 13. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "No correlation results were generated."
    )


# ============================================================
# 14. SAVE RESULTS
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
# 15. SUMMARY
# ============================================================

included = results_df[
    results_df["Included"] == "Yes"
]

valid = results_df[
    results_df["r"].notna()
    & results_df["p"].notna()
]


print()
print("=" * 70)
print("SAME-DAY ANALYSIS COMPLETED")
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
    f"Valid Pearson correlations: "
    f"{len(valid)}"
)

print(
    f"Output saved to:\n"
    f"{OUTPUT_FILE}"
)

print("=" * 70)
