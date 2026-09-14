import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# ============================================================
# 7-DAY HISTORY → CURRENT WELL-BEING ANALYSIS
# ============================================================

data_dir = Path("data/pmdata")
results_dir = Path("results/wellbeing_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

# Behavioral / sleep variables
behavioral_variables = [
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

# Well-being variables
wellness_variables = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]

MIN_N = 10
WINDOW_DAYS = 7

all_results = []

# ============================================================
# 1. LOAD PARTICIPANTS
# ============================================================

files = sorted(data_dir.glob("p*_daily_merged.csv"))

print("==============================================")
print("7-DAY HISTORY → CURRENT WELL-BEING ANALYSIS")
print("==============================================")

print(f"Participants found: {len(files)}")

# ============================================================
# 2. ANALYZE EACH PARTICIPANT
# ============================================================

for file in files:

    participant = file.stem.replace("_daily_merged", "")

    df = pd.read_csv(file)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    # Convert variables to numeric
    for col in behavioral_variables + wellness_variables:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # Create 7-day PREVIOUS history
    #
    # shift(1) excludes the current day.
    # rolling(7) therefore represents the previous 7 days.
    # --------------------------------------------------------

    for variable in behavioral_variables:

        if variable not in df.columns:
            continue

        df[f"{variable}_7day_mean"] = (
            df[variable]
            .shift(1)
            .rolling(
                WINDOW_DAYS,
                min_periods=1
            )
            .mean()
        )

    # ========================================================
    # 3. CORRELATIONS
    # ========================================================

    for behavioral_variable in behavioral_variables:

        history_variable = f"{behavioral_variable}_7day_mean"

        if history_variable not in df.columns:
            continue

        for wellness_variable in wellness_variables:

            if wellness_variable not in df.columns:
                continue

            analysis_df = df[
                [
                    "Date",
                    history_variable,
                    wellness_variable
                ]
            ].dropna()

            N = len(analysis_df)

            r = np.nan
            p = np.nan
            included = False

            if N >= MIN_N:

                x = analysis_df[history_variable]
                y = analysis_df[wellness_variable]

                # Avoid constant-input errors
                if x.nunique() > 1 and y.nunique() > 1:

                    r, p = pearsonr(x, y)
                    included = True

            all_results.append({
                "Participant": participant,
                "Behavioral_Variable": behavioral_variable,
                "Wellness_Variable": wellness_variable,
                "Window": "Previous 7 days",
                "Relationship": "7-day history → current well-being",
                "r": r,
                "p": p,
                "N": N,
                "Included": included
            })


# ============================================================
# 4. CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(all_results)

# ============================================================
# 5. BENJAMINI-HOCHBERG FDR
# ============================================================

results["p_FDR"] = np.nan
results["Significant_raw"] = False
results["Significant_FDR"] = False

valid = (
    results["Included"]
    & results["p"].notna()
)

if valid.sum() > 0:

    p_values = results.loc[valid, "p"].values

    reject, corrected_p, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="fdr_bh"
    )

    results.loc[valid, "p_FDR"] = corrected_p
    results.loc[valid, "Significant_raw"] = (
        results.loc[valid, "p"] < 0.05
    )
    results.loc[valid, "Significant_FDR"] = reject


# ============================================================
# 6. SAVE RESULTS
# ============================================================

output_file = (
    results_dir /
    "seven_day_history_wellbeing_relationships.csv"
)

results.to_csv(output_file, index=False)


# ============================================================
# 7. PRINT SUMMARY
# ============================================================

print()
print("Total relationship tests:", len(results))
print(
    "Included tests (N >= 10):",
    results["Included"].sum()
)
print(
    "Raw significant (p < 0.05):",
    results["Significant_raw"].sum()
)
print(
    "FDR significant:",
    results["Significant_FDR"].sum()
)

print()
print("Saved to:")
print(output_file)

print("==============================================")
