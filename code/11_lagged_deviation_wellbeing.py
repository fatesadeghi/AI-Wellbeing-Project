import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# --------------------------------------------------
# One-Day Lagged Analysis
# Behavioral Deviation (Day t)
# -> Well-being (Day t+1)
# --------------------------------------------------

BASE_DIR = Path(".")
DEVIATION_FILE = (
    BASE_DIR /
    "results/baseline/daily_personalized_deviations.csv"
)

OUTPUT_FILE = (
    BASE_DIR /
    "results/wellbeing_analysis/lagged_deviation_wellbeing_relationships.csv"
)

WELLNESS_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]

BEHAVIORAL_VARIABLES = [
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


# --------------------------------------------------
# Load personalized deviations
# --------------------------------------------------

deviations = pd.read_csv(DEVIATION_FILE)
deviations["Date"] = pd.to_datetime(deviations["Date"])

print("\nONE-DAY LAGGED DEVIATION × WELL-BEING")
print("--------------------------------------")
print(f"Deviation records: {len(deviations)}")
print(f"Participants: {deviations['Participant'].nunique()}")


# --------------------------------------------------
# Analyze participant by participant
# --------------------------------------------------

results = []

for participant in sorted(deviations["Participant"].unique()):

    participant_dev = deviations[
        deviations["Participant"] == participant
    ].copy()

    # Load participant daily data
    participant_file = (
        BASE_DIR /
        f"data/pmdata/{participant}_daily_merged.csv"
    )

    if not participant_file.exists():
        continue

    wellness = pd.read_csv(participant_file)
    wellness["Date"] = pd.to_datetime(wellness["Date"])

    # Keep analysis period: second half
    wellness = wellness.sort_values("Date").reset_index(drop=True)

    midpoint = len(wellness) // 2
    wellness = wellness.iloc[midpoint:].copy()

    # Shift wellness one day backward:
    # deviation at day t will match wellness at day t+1
    wellness["Deviation_Date"] = (
        wellness["Date"] - pd.Timedelta(days=1)
    )

    merged = participant_dev.merge(
        wellness,
        left_on="Date",
        right_on="Deviation_Date",
        how="inner"
    )

    for behavioral_variable in BEHAVIORAL_VARIABLES:

        z_variable = f"{behavioral_variable}_Z"

        if z_variable not in merged.columns:
            continue

        for wellness_variable in WELLNESS_VARIABLES:

            if wellness_variable not in merged.columns:
                continue

            data = merged[
                [z_variable, wellness_variable]
            ].replace([np.inf, -np.inf], np.nan).dropna()

            N = len(data)

            if N < 10:
                results.append({
                    "Participant": participant,
                    "Behavioral_Variable": behavioral_variable,
                    "Wellness_Variable": wellness_variable,
                    "r": np.nan,
                    "p": np.nan,
                    "N": N,
                    "Included": "No"
                })
                continue

            # Avoid undefined correlation when one variable is constant
            if (
                data[z_variable].nunique() < 2 or
                data[wellness_variable].nunique() < 2
            ):
                results.append({
                    "Participant": participant,
                    "Behavioral_Variable": behavioral_variable,
                    "Wellness_Variable": wellness_variable,
                    "r": np.nan,
                    "p": np.nan,
                    "N": N,
                    "Included": "No"
                })
                continue

            r, p = pearsonr(
                data[z_variable],
                data[wellness_variable]
            )

            results.append({
                "Participant": participant,
                "Behavioral_Variable": behavioral_variable,
                "Wellness_Variable": wellness_variable,
                "r": r,
                "p": p,
                "N": N,
                "Included": "Yes"
            })


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

valid = results_df["Included"] == "Yes"

print("\nANALYSIS COMPLETE")
print("--------------------------------------")
print(f"Total relationship tests: {len(results_df)}")
print(f"Included tests (N >= 10): {valid.sum()}")

print(
    "Raw significant (p < 0.05):",
    (
        (results_df.loc[valid, "p"] < 0.05)
        .sum()
    )
)

print(f"\nOutput:")
print(OUTPUT_FILE)
