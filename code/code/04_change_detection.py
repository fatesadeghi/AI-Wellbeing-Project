import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "baseline",
    "daily_personalized_deviations.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "change_detection"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "detected_changes.csv"
)

Z_THRESHOLD = 2.0


# ============================================================
# OBJECTIVE VARIABLES
# ============================================================

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
# LOAD DATA
# ============================================================

print("=" * 70)
print("CHANGE / ANOMALY DETECTION")
print("=" * 70)

print(f"Input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print(f"Records loaded: {len(df)}")
print(f"Participants: {df['Participant'].nunique()}")


# ============================================================
# DETECT MEANINGFUL DEVIATIONS
# ============================================================

print("\n" + "=" * 70)
print("DETECTING MEANINGFUL DEVIATIONS")
print("=" * 70)

change_records = []

for _, row in df.iterrows():

    participant = row["Participant"]
    date = row["Date"]

    for variable in OBJECTIVE_VARIABLES:

        abs_z_column = f"{variable}_Abs_Z"
        z_column = f"{variable}_Z"
        deviation_column = f"{variable}_Deviation"

        abs_z = row.get(abs_z_column, np.nan)
        z = row.get(z_column, np.nan)
        deviation = row.get(deviation_column, np.nan)

        if pd.isna(abs_z):
            continue

        if abs_z >= Z_THRESHOLD:

            direction = "Higher than baseline"

            if z < 0:
                direction = "Lower than baseline"

            change_records.append({
                "Participant": participant,
                "Date": date,
                "Variable": variable,
                "Deviation": deviation,
                "Z": z,
                "Abs_Z": abs_z,
                "Direction": direction
            })


# ============================================================
# CREATE OUTPUT
# ============================================================

changes_df = pd.DataFrame(change_records)

os.makedirs(OUTPUT_DIR, exist_ok=True)

changes_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHANGE DETECTION COMPLETE")
print("=" * 70)

print(f"Z-score threshold: {Z_THRESHOLD}")
print(f"Detected meaningful deviations: {len(changes_df)}")

if len(changes_df) > 0:

    print(
        f"Participants with detected changes: "
        f"{changes_df['Participant'].nunique()}"
    )

    print("\nDetected changes by variable:")

    print(
        changes_df["Variable"]
        .value_counts()
        .to_string()
    )

else:

    print("No meaningful deviations were detected.")


print("\nOutput file:")
print(OUTPUT_FILE)

print("=" * 70)
