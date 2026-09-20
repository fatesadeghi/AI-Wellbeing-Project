import os
import warnings

import numpy as np
import pandas as pd
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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

BASELINE_DIR = os.path.join(
    RESULTS_DIR,
    "baseline"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "lagged"
)

BASELINE_FILE = os.path.join(
    BASELINE_DIR,
    "personalized_baselines.csv"
)

DEVIATION_FILE = os.path.join(
    BASELINE_DIR,
    "daily_personalized_deviations.csv"
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
# 2. ANALYSIS SETTINGS
# ============================================================

BASELINE_FRACTION = 0.50
MIN_BASELINE_N = 7
MIN_CORRELATION_N = 10

BEHAVIOR_VARS = [
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

WELLBEING_VARS = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
]


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("LAGGED BEHAVIOR-WELLBEING ANALYSIS")
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


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(BASELINE_FILE):
    raise FileNotFoundError(
        f"Baseline file not found:\n{BASELINE_FILE}"
    )

if not os.path.exists(DEVIATION_FILE):
    raise FileNotFoundError(
        f"Deviation file not found:\n{DEVIATION_FILE}"
    )

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Data directory not found:\n{DATA_DIR}"
    )


# ============================================================
# 5. LOAD BASELINE DATA
# ============================================================

print()
print("Loading personalized baseline data...")

baselines = pd.read_csv(
    BASELINE_FILE
)

if "Participant" not in baselines.columns:
    raise ValueError(
        "Baseline file must contain 'Participant'."
    )


# ============================================================
# 6. LOAD DAILY DEVIATIONS
# ============================================================

print("Loading personalized daily deviations...")

deviations = pd.read_csv(
    DEVIATION_FILE
)

if "Participant" not in deviations.columns:
    raise ValueError(
        "Deviation file must contain 'Participant'."
    )

if "Date" not in deviations.columns:
    raise ValueError(
        "Deviation file must contain 'Date'."
    )

deviations["Date"] = pd.to_datetime(
    deviations["Date"],
    errors="coerce"
)

deviations = deviations.dropna(
    subset=["Date"]
)


# ============================================================
# 7. LOAD PARTICIPANT FILES
# ============================================================

print("Loading participant files...")

participant_files = sorted(
    [
        file_name
        for file_name in os.listdir(DATA_DIR)
        if file_name.lower().endswith(".csv")
    ]
)

if not participant_files:
    raise RuntimeError(
        "No participant CSV files were found."
    )

wellbeing_frames = []

for file_name in participant_files:

    participant_path = os.path.join(
        DATA_DIR,
        file_name
    )

    participant_id = os.path.splitext(
        file_name
    )[0]

    try:
        df = pd.read_csv(
            participant_path
        )
    except Exception as exc:
        warnings.warn(
            f"Could not read {file_name}: {exc}"
        )
        continue

    # --------------------------------------------------------
    # Standardize date column
    # --------------------------------------------------------

    date_column = None

    for candidate in [
        "Date",
        "date",
        "DATE",
    ]:
        if candidate in df.columns:
            date_column = candidate
            break

    if date_column is None:
        warnings.warn(
            f"No date column found in {file_name}. Skipping."
        )
        continue

    if date_column != "Date":
        df = df.rename(
            columns={
                date_column: "Date"
            }
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    # --------------------------------------------------------
    # Participant
    # --------------------------------------------------------

    if "Participant" not in df.columns:
        df["Participant"] = participant_id

    # --------------------------------------------------------
    # Keep wellbeing variables that exist
    # --------------------------------------------------------

    available_wellbeing = [
        variable
        for variable in WELLBEING_VARS
        if variable in df.columns
    ]

    if not available_wellbeing:
        warnings.warn(
            f"No wellbeing variables found in {file_name}. "
            "Skipping."
        )
        continue

    wellbeing_frames.append(
        df[
            [
                "Participant",
                "Date",
            ]
            + available_wellbeing
        ].copy()
    )


if not wellbeing_frames:
    raise RuntimeError(
        "No participant wellbeing data could be loaded."
    )


wellbeing = pd.concat(
    wellbeing_frames,
    ignore_index=True
)


# ============================================================
# 8. STANDARDIZE PARTICIPANT IDENTIFIERS
# ============================================================

deviations["Participant"] = (
    deviations["Participant"]
    .astype(str)
    .str.strip()
)

wellbeing["Participant"] = (
    wellbeing["Participant"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 9. REMOVE DUPLICATE PARTICIPANT-DATE ROWS
# ============================================================

wellbeing = (
    wellbeing
    .sort_values(
        [
            "Participant",
            "Date",
        ]
    )
    .drop_duplicates(
        subset=[
            "Participant",
            "Date",
        ],
        keep="first"
    )
    .reset_index(drop=True)
)


# ============================================================
# 10. CREATE ONE-DAY LAG ALIGNMENT
# ============================================================

print("Creating one-day lag alignment...")

# For each participant:
#
# Behavior Date = t
# Wellbeing Date = t + 1
#
# We therefore shift the wellbeing date backward
# by one day so that it can be merged with behavior.

wellbeing["Behavior_Date"] = (
    wellbeing["Date"]
    - pd.Timedelta(days=1)
)


# ============================================================
# 11. MERGE LAGGED DATA
# ============================================================

print(
    "Merging behavioral deviations "
    "with next-day wellbeing..."
)

merged = pd.merge(
    deviations,
    wellbeing,
    left_on=[
        "Participant",
        "Date",
    ],
    right_on=[
        "Participant",
        "Behavior_Date",
    ],
    how="inner",
    suffixes=(
        "_behavior",
        "_wellbeing"
    )
)

if merged.empty:
    raise RuntimeError(
        "The lagged merge produced no matching "
        "participant-date rows."
    )


# ------------------------------------------------------------
# IMPORTANT:
# After the merge:
#
# deviations Date      -> Date_behavior
# wellbeing Date       -> Date_wellbeing
# Behavior_Date        -> Behavior_Date
#
# The analysis date is the behavior date t.
# ------------------------------------------------------------

if "Date_behavior" in merged.columns:
    merged["Date"] = merged["Date_behavior"]

elif "Date" in merged.columns:
    merged["Date"] = merged["Date"]

else:
    raise RuntimeError(
        "Could not identify the behavioral Date "
        "after the lagged merge."
    )


merged = merged.sort_values(
    [
        "Participant",
        "Date",
    ]
).reset_index(drop=True)


# ============================================================
# 12. PARTICIPANT-LEVEL LAGGED ANALYSIS
# ============================================================

print(
    "Calculating participant-level "
    "lagged correlations..."
)


results = []


# ------------------------------------------------------------
# Safe Pearson correlation
# ------------------------------------------------------------

def safe_pearson(
    x,
    y,
    minimum_n=MIN_CORRELATION_N
):

    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce"
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce"
            ),
        }
    ).dropna()

    n = len(pair)

    if n < minimum_n:
        return (
            np.nan,
            np.nan,
            n
        )

    if (
        pair["x"].nunique() <= 1
        or pair["y"].nunique() <= 1
    ):
        return (
            np.nan,
            np.nan,
            n
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        r, p = pearsonr(
            pair["x"],
            pair["y"]
        )

    return (
        float(r),
        float(p),
        n
    )


# ------------------------------------------------------------
# Participant-level loop
# ------------------------------------------------------------

participants = sorted(
    merged["Participant"]
    .dropna()
    .unique()
)


for participant in participants:

    participant_data = (
        merged[
            merged["Participant"] == participant
        ]
        .copy()
    )

    for behavior in BEHAVIOR_VARS:

        # The behavioral deviation column
        # is expected to be named:
        #
        # <Behavior>_Deviation
        #
        # in daily_personalized_deviations.csv

        behavior_column = (
            f"{behavior}_Deviation"
        )

        if behavior_column not in participant_data.columns:

            # Some baseline-generation versions may
            # store the deviation column simply as
            # the behavioral variable name.
            if behavior in participant_data.columns:
                behavior_column = behavior
            else:
                results.append(
                    {
                        "Participant": participant,
                        "Behavior_Variable": behavior,
                        "Wellbeing_Variable": np.nan,
                        "r": np.nan,
                        "p": np.nan,
                        "N": 0,
                        "Included": "No",
                    }
                )

                continue

        for wellbeing_variable in WELLBEING_VARS:

            if (
                wellbeing_variable
                not in participant_data.columns
            ):
                results.append(
                    {
                        "Participant": participant,
                        "Behavior_Variable": behavior,
                        "Wellbeing_Variable": wellbeing_variable,
                        "r": np.nan,
                        "p": np.nan,
                        "N": 0,
                        "Included": "No",
                    }
                )

                continue

            r, p, n = safe_pearson(
                participant_data[
                    behavior_column
                ],
                participant_data[
                    wellbeing_variable
                ],
                minimum_n=MIN_CORRELATION_N
            )

            included = (
                "Yes"
                if n >= MIN_CORRELATION_N
                and np.isfinite(r)
                and np.isfinite(p)
                else "No"
            )

            results.append(
                {
                    "Participant": participant,
                    "Behavior_Variable": behavior,
                    "Wellbeing_Variable": wellbeing_variable,
                    "r": r,
                    "p": p,
                    "N": n,
                    "Included": included,
                }
            )


# ============================================================
# 13. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 14. SORT RESULTS
# ============================================================

results_df = results_df.sort_values(
    [
        "Participant",
        "Behavior_Variable",
        "Wellbeing_Variable",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 16. SUMMARY
# ============================================================

total_relationships = len(
    results_df
)

included_relationships = int(
    (
        results_df["Included"]
        == "Yes"
    ).sum()
)


print()
print("=" * 70)
print("LAGGED ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Participants analyzed: "
    f"{len(participants)}"
)

print(
    f"Total relationships tested: "
    f"{total_relationships}"
)

print(
    f"Relationships with N >= "
    f"{MIN_CORRELATION_N}: "
    f"{included_relationships}"
)

print(
    "Output saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)
