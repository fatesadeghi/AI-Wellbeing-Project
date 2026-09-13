```python
import os
import pandas as pd
import numpy as np


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "baseline"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

# The first 50% of each participant's chronological data
# is used to learn the participant's personal baseline.

BASELINE_FRACTION = 0.50


OBJECTIVE_VARIABLES = [
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
    "Sleep_Score"
]


# ============================================================
# 3. LOAD PARTICIPANT FILES
# ============================================================

participant_data = {}

print("=" * 70)
print("LOADING PARTICIPANT DATA")
print("=" * 70)

print(
    f"Data directory: {DATA_DIR}"
)

print()


if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        f"Data directory not found:\n{DATA_DIR}"
    )


for file in sorted(
    os.listdir(DATA_DIR)
):

    # --------------------------------------------------------
    # Only use participant daily merged files
    # --------------------------------------------------------

    if not file.endswith(
        "_daily_merged.csv"
    ):

        continue


    # --------------------------------------------------------
    # Extract participant ID
    #
    # Example:
    # p01_daily_merged.csv
    #       ↓
    # p01
    # --------------------------------------------------------

    participant = file.replace(
        "_daily_merged.csv",
        ""
    )


    file_path = os.path.join(
        DATA_DIR,
        file
    )


    try:

        df = pd.read_csv(
            file_path
        )

    except Exception as e:

        print(
            f"Could not read {file}: {e}"
        )

        continue


    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = (
            df
            .sort_values("Date")
            .reset_index(drop=True)
        )


    # Remove duplicate rows

    df = df.drop_duplicates()


    participant_data[
        participant
    ] = df


    print(
        f"{participant}: "
        f"{len(df)} rows"
    )


print()

print(
    "Participants loaded:",
    len(participant_data)
)


# ============================================================
# 4. BUILD PERSONALIZED BASELINES
# ============================================================

baseline_records = []

deviation_records = []


print()
print("=" * 70)
print("BUILDING PERSONALIZED BASELINES")
print("=" * 70)


for participant, df in participant_data.items():

    # --------------------------------------------------------
    # Determine baseline period
    # --------------------------------------------------------

    n_rows = len(df)

    baseline_end = int(
        n_rows * BASELINE_FRACTION
    )


    if baseline_end < 2:

        print(
            f"{participant}: "
            f"not enough data for baseline"
        )

        continue


    baseline_df = (
        df
        .iloc[:baseline_end]
        .copy()
    )


    analysis_df = (
        df
        .iloc[baseline_end:]
        .copy()
    )


    print(
        f"{participant}: "
        f"baseline={len(baseline_df)} days, "
        f"analysis={len(analysis_df)} days"
    )


    # ========================================================
    # 5. CALCULATE BASELINE STATISTICS
    # ========================================================

    baseline_record = {

        "Participant": participant,

        "Baseline_N": len(baseline_df)

    }


    baseline_stats = {}


    for variable in OBJECTIVE_VARIABLES:

        if variable not in baseline_df.columns:

            continue


        values = pd.to_numeric(
            baseline_df[variable],
            errors="coerce"
        ).dropna()


        if len(values) < 2:

            continue


```
