import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "wellbeing_analysis",
    "deviation_wellbeing_same_day.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "wellbeing_analysis"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "deviation_wellbeing_statistics.csv"
)

MIN_N = 10


# ============================================================
# VARIABLES
# ============================================================

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

WELLNESS_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STATISTICAL ANALYSIS: BEHAVIORAL DEVIATION × WELL-BEING")
print("=" * 70)

print(f"Input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df)}")
print(f"Participants: {df['Participant'].nunique()}")


# ============================================================
# PARTICIPANT-LEVEL CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING PARTICIPANT-LEVEL CORRELATIONS")
print("=" * 70)

results = []

for participant in sorted(df["Participant"].dropna().unique()):

    participant_df = df[
        df["Participant"] == participant
    ].copy()

    for behavioral_variable in BEHAVIORAL_VARIABLES:

        behavioral_df = participant_df[
            participant_df["Behavioral_Variable"]
            == behavioral_variable
        ].copy()

        if behavioral_df.empty:
            continue

        for wellness_variable in WELLNESS_VARIABLES:

            pair_df = behavioral_df[
                behavioral_df["Wellness_Variable"]
                == wellness_variable
            ][
                ["Behavioral_Z", "Wellness_Z"]
            ].copy()

            pair_df["Behavioral_Z"] = pd.to_numeric(
                pair_df["Behavioral_Z"],
                errors="coerce"
            )

            pair_df["Wellness_Z"] = pd.to_numeric(
                pair_df["Wellness_Z"],
                errors="coerce"
            )

            pair_df = pair_df.dropna()

            n = len(pair_df)

            if n < MIN_N:

                results.append({
                    "Participant": participant,
                    "Behavioral_Variable": behavioral_variable,
                    "Wellness_Variable": wellness_variable,
                    "r": np.nan,
                    "p": np.nan,
                    "N": n,
                    "Included": "No"
                })

                continue

            x = pair_df["Behavioral_Z"]
            y = pair_df["Wellness_Z"]

            # Correlation is undefined when one variable has no variation
            if x.nunique() < 2 or y.nunique() < 2:

                results.append({
                    "Participant": participant,
                    "Behavioral_Variable": behavioral_variable,
                    "Wellness_Variable": wellness_variable,
                    "r": np.nan,
                    "p": np.nan,
                    "N": n,
                    "Included": "No"
                })

                continue

            r, p = pearsonr(x, y)

            results.append({
                "Participant": participant,
                "Behavioral_Variable": behavioral_variable,
                "Wellness_Variable": wellness_variable,
                "r": r,
                "p": p,
                "N": n,
                "Included": "Yes"
            })


results_df = pd.DataFrame(results)


# ============================================================
# FDR CORRECTION
# ============================================================

print("\n" + "=" * 70)
print("APPLYING BENJAMINI-HOCHBERG FDR CORRECTION")
print("=" * 70)

valid = (
    results_df["Included"] == "Yes"
) & results_df["p"].notna()

results_df["p_FDR"] = np.nan
results_df["Significant_raw"] = False
results_df["Significant_FDR"] = False

if valid.sum() > 0:

    p_values = results_df.loc[
        valid,
        "p"
    ].values

    reject, p_corrected, _, _ = multipletests(
        p_values,
        method="fdr_bh"
    )

    results_df.loc[
        valid,
        "p_FDR"
    ] = p_corrected

    results_df.loc[
        valid,
        "Significant_raw"
    ] = p_values < 0.05

    results_df.loc[
        valid,
        "Significant_FDR"
    ] = reject


# ============================================================
# SAVE
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
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Total relationship tests: "
    f"{len(results_df)}"
)

print(
    f"Included tests (N >= {MIN_N}): "
    f"{valid.sum()}"
)

print(
    f"Raw significant (p < 0.05): "
    f"{results_df['Significant_raw'].sum()}"
)

print(
    f"FDR significant: "
    f"{results_df['Significant_FDR'].sum()}"
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("=" * 70)
