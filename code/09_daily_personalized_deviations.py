import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 09 - DAILY PERSONALIZED DEVIATION × WELL-BEING ANALYSIS
# ============================================================

DEVIATION_FILE = Path(
    "results/baseline/daily_personalized_deviations.csv"
)

DATA_DIR = Path("data/pmdata")
OUTPUT_DIR = Path("results/wellbeing_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

WELLNESS_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]

MIN_N = 10

print("=" * 70)
print("DAILY PERSONALIZED DEVIATION × WELL-BEING ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# Load deviation data
# ------------------------------------------------------------

deviation_df = pd.read_csv(DEVIATION_FILE)

deviation_df["Date"] = pd.to_datetime(deviation_df["Date"])

print()
print("Deviation records:", len(deviation_df))
print(
    "Participants:",
    deviation_df["Participant"].nunique()
)

# ------------------------------------------------------------
# Load original participant data
# ------------------------------------------------------------

participants = {}

files = sorted(DATA_DIR.glob("p*_daily_merged.csv"))

for file in files:

    participant = file.name.split("_")[0]

    df = pd.read_csv(file)

    if "Date" not in df.columns:
        continue

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    participants[participant] = df

print(
    "Participants loaded:",
    len(participants)
)

# ------------------------------------------------------------
# Calculate correlations
# ------------------------------------------------------------

results = []

for participant, participant_deviations in deviation_df.groupby(
    "Participant"
):

    if participant not in participants:
        continue

    original_df = participants[participant].copy()

    # Keep only the analysis period
    total_days = len(original_df)

    baseline_days = int(
        total_days * 0.5
    )

    analysis_df = original_df.iloc[baseline_days:].copy()

    # Merge deviation data with well-being data
    merged = pd.merge(
        participant_deviations,
        analysis_df[
            ["Date"] + WELLNESS_VARIABLES
        ],
        on="Date",
        how="inner"
    )

    for behavioral_variable in OBJECTIVE_VARIABLES:

        z_column = behavioral_variable + "_Z"

        if z_column not in merged.columns:
            continue

        for wellness_variable in WELLNESS_VARIABLES:

            if wellness_variable not in merged.columns:
                continue

            data = merged[
                [z_column, wellness_variable]
            ].copy()

            data[z_column] = pd.to_numeric(
                data[z_column],
                errors="coerce"
            )

            data[wellness_variable] = pd.to_numeric(
                data[wellness_variable],
                errors="coerce"
            )

            data = data.dropna()

            n = len(data)

            if n < MIN_N:
                r = np.nan
                p = np.nan
                included = "No"

            else:

                r = data[z_column].corr(
                    data[wellness_variable]
                )

                if pd.isna(r):
                    p = np.nan
                else:

                    # Pearson correlation p-value
                    from scipy.stats import pearsonr

                    r, p = pearsonr(
                        data[z_column],
                        data[wellness_variable]
                    )

                included = "Yes"

            results.append(
                {
                    "Participant": participant,
                    "Behavioral_Variable": behavioral_variable,
                    "Wellness_Variable": wellness_variable,
                    "r": r,
                    "p": p,
                    "N": n,
                    "Included": included
                }
            )

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

output_file = (
    OUTPUT_DIR
    / "daily_deviation_wellbeing_relationships.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

included_count = (
    results_df["Included"] == "Yes"
).sum()

significant_count = (
    (results_df["Included"] == "Yes")
    & (results_df["p"] < 0.05)
).sum()

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    "Total relationship tests:",
    len(results_df)
)

print(
    "Included tests (N >= 10):",
    included_count
)

print(
    "Raw significant (p < 0.05):",
    significant_count
)

print()
print("Output:")
print(output_file)
