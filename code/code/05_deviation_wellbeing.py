import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "pmdata"
)

DEVIATION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "change_detection",
    "detected_changes.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "wellbeing_analysis"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "deviation_wellbeing_same_day.csv"
)

BASELINE_FRACTION = 0.50
Z_THRESHOLD = 2.0


# ============================================================
# WELLNESS VARIABLES
# ============================================================

WELLNESS_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]


# ============================================================
# LOAD DETECTED BEHAVIORAL CHANGES
# ============================================================

print("=" * 70)
print("DEVIATION → WELL-BEING SAME-DAY ANALYSIS")
print("=" * 70)

print(f"Deviation file: {DEVIATION_FILE}")

changes_df = pd.read_csv(DEVIATION_FILE)

changes_df["Date"] = pd.to_datetime(
    changes_df["Date"],
    errors="coerce"
)

print(f"Detected deviations loaded: {len(changes_df)}")
print(
    f"Participants with deviations: "
    f"{changes_df['Participant'].nunique()}"
)


# ============================================================
# LOAD WELLNESS DATA AND BUILD PERSONAL BASELINES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PERSONAL WELL-BEING BASELINES")
print("=" * 70)

participant_data = {}
wellbeing_baselines = {}

for filename in sorted(os.listdir(DATA_DIR)):

    if not filename.endswith("_daily_merged.csv"):
        continue

    participant = filename.replace(
        "_daily_merged.csv",
        ""
    )

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(filepath)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.sort_values("Date")
    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    )

    participant_data[participant] = df

    # --------------------------------------------------------
    # First 50% = personal wellness baseline
    # --------------------------------------------------------

    split_index = int(
        len(df) * BASELINE_FRACTION
    )

    baseline_df = df.iloc[:split_index].copy()

    baseline_stats = {}

    for variable in WELLNESS_VARIABLES:

        values = pd.to_numeric(
            baseline_df[variable],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            baseline_stats[variable] = {
                "Mean": np.nan,
                "SD": np.nan
            }

        else:
            baseline_stats[variable] = {
                "Mean": values.mean(),
                "SD": values.std()
            }

    wellbeing_baselines[participant] = baseline_stats

    print(
        f"{participant}: "
        f"baseline={len(baseline_df)} days"
    )


# ============================================================
# MATCH DEVIATIONS WITH SAME-DAY WELL-BEING
# ============================================================

print("\n" + "=" * 70)
print("MATCHING DEVIATIONS WITH SAME-DAY WELL-BEING")
print("=" * 70)

records = []

for _, change in changes_df.iterrows():

    participant = change["Participant"]
    date = change["Date"]
    behavioral_variable = change["Variable"]

    if participant not in participant_data:
        continue

    df = participant_data[participant]

    same_day = df[df["Date"] == date]

    if same_day.empty:
        continue

    row = same_day.iloc[0]

    for wellness_variable in WELLNESS_VARIABLES:

        wellness_value = pd.to_numeric(
            row[wellness_variable],
            errors="coerce"
        )

        if pd.isna(wellness_value):
            continue

        baseline_mean = wellbeing_baselines[
            participant
        ][wellness_variable]["Mean"]

        baseline_sd = wellbeing_baselines[
            participant
        ][wellness_variable]["SD"]

        if pd.isna(baseline_mean):
            continue

        if pd.isna(baseline_sd) or baseline_sd == 0:

            wellness_z = np.nan

        else:

            wellness_z = (
                wellness_value - baseline_mean
            ) / baseline_sd

        records.append({
            "Participant": participant,
            "Date": date,
            "Behavioral_Variable": behavioral_variable,
            "Behavioral_Deviation": change["Deviation"],
            "Behavioral_Z": change["Z"],
            "Behavioral_Abs_Z": change["Abs_Z"],
            "Behavioral_Direction": change["Direction"],
            "Wellness_Variable": wellness_variable,
            "Wellness_Value": wellness_value,
            "Wellness_Baseline_Mean": baseline_mean,
            "Wellness_Baseline_SD": baseline_sd,
            "Wellness_Z": wellness_z
        })


# ============================================================
# SAVE OUTPUT
# ============================================================

result_df = pd.DataFrame(records)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SAME-DAY WELL-BEING ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Matched deviation-well-being records: "
    f"{len(result_df)}"
)

if len(result_df) > 0:

    print(
        f"Participants included: "
        f"{result_df['Participant'].nunique()}"
    )

    print("\nBehavioral variables:")

    print(
        result_df["Behavioral_Variable"]
        .value_counts()
        .to_string()
    )

    print("\nWell-being variables:")

    print(
        result_df["Wellness_Variable"]
        .value_counts()
        .to_string()
    )

print("\nOutput file:")
print(OUTPUT_FILE)

print("=" * 70)
