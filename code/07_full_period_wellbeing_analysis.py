import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "pmdata"
)

BASELINE_FRACTION = 0.50
MIN_N = 10

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "wellbeing_analysis"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "full_period_behavior_wellbeing_statistics.csv"
)


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
# LOAD PARTICIPANT DATA
# ============================================================

print("=" * 70)
print("FULL-PERIOD BEHAVIOR × WELL-BEING ANALYSIS")
print("=" * 70)

print(f"Data directory: {DATA_DIR}")

participant_data = {}

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

    print(
        f"{participant}: {len(df)} rows"
    )


print(
    f"\nParticipants loaded: "
    f"{len(participant_data)}"
)


# ============================================================
# BUILD PERSONALIZED BASELINES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PERSONALIZED BEHAVIOR AND WELL-BEING BASELINES")
print("=" * 70)

all_results = []

for participant in sorted(participant_data.keys()):

    df = participant_data[participant].copy()

    split_index = int(
        len(df) * BASELINE_FRACTION
    )

    baseline_df = df.iloc[:split_index].copy()
    analysis_df = df.iloc[split_index:].copy()

    print(
        f"{participant}: "
        f"baseline={len(baseline_df)} days, "
        f"analysis={len(analysis_df)} days"
    )

    # --------------------------------------------------------
    # Behavioral baseline statistics
    # --------------------------------------------------------

    behavioral_stats = {}

    for variable in BEHAVIORAL_VARIABLES:

        values = pd.to_numeric(
            baseline_df[variable],
            errors="coerce"
        ).dropna()

        if len(values) == 0:

            behavioral_stats[variable] = {
                "mean": np.nan,
                "sd": np.nan
            }

        else:

            behavioral_stats[variable] = {
                "mean": values.mean(),
                "sd": values.std()
            }

    # --------------------------------------------------------
    # Well-being baseline statistics
    # --------------------------------------------------------

    wellness_stats = {}

    for variable in WELLNESS_VARIABLES:

        values = pd.to_numeric(
            baseline_df[variable],
            errors="coerce"
        ).dropna()

        if len(values) == 0:

            wellness_stats[variable] = {
                "mean": np.nan,
                "sd": np.nan
            }

        else:

            wellness_stats[variable] = {
                "mean": values.mean(),
                "sd": values.std()
            }

    # --------------------------------------------------------
    # Calculate daily personalized Z-scores
    # --------------------------------------------------------

    for _, row in analysis_df.iterrows():

        date = row["Date"]

        behavioral_z = {}

        for variable in BEHAVIORAL_VARIABLES:

            value = pd.to_numeric(
                row[variable],
                errors="coerce"
            )

            mean = behavioral_stats[variable]["mean"]
            sd = behavioral_stats[variable]["sd"]

            if (
                pd.isna(value)
                or pd.isna(mean)
                or pd.isna(sd)
                or sd == 0
            ):

                behavioral_z[variable] = np.nan

            else:

                behavioral_z[variable] = (
                    value - mean
                ) / sd

        wellness_z = {}

        for variable in WELLNESS_VARIABLES:

            value = pd.to_numeric(
                row[variable],
                errors="coerce"
            )

            mean = wellness_stats[variable]["mean"]
            sd = wellness_stats[variable]["sd"]

            if (
                pd.isna(value)
                or pd.isna(mean)
                or pd.isna(sd)
                or sd == 0
            ):

                wellness_z[variable] = np.nan

            else:

                wellness_z[variable] = (
                    value - mean
                ) / sd

        # ----------------------------------------------------
        # Store one daily record
        # ----------------------------------------------------

        record = {
            "Participant": participant,
            "Date": date
        }

        for variable in BEHAVIORAL_VARIABLES:

            record[
                f"{variable}_Z"
            ] = behavioral_z[variable]

        for variable in WELLNESS_VARIABLES:

            record[
                f"{variable}_Z"
            ] = wellness_z[variable]

        all_results.append(record)


# ============================================================
# CREATE ANALYSIS DATAFRAME
# ============================================================

analysis_df = pd.DataFrame(
    all_results
)

print("\nAnalysis records created:")
print(len(analysis_df))


# ============================================================
# PARTICIPANT-LEVEL CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING PARTICIPANT-LEVEL CORRELATIONS")
print("=" * 70)

results = []

for participant in sorted(
    analysis_df["Participant"].unique()
):

    participant_df = analysis_df[
        analysis_df["Participant"] == participant
    ].copy()

    for behavioral_variable in BEHAVIORAL_VARIABLES:

        behavioral_column = (
            f"{behavioral_variable}_Z"
        )

        for wellness_variable in WELLNESS_VARIABLES:

            wellness_column = (
                f"{wellness_variable}_Z"
            )

            pair_df = participant_df[
                [
                    behavioral_column,
                    wellness_column
                ]
            ].dropna()

            n = len(pair_df)

            if n < MIN_N:

                results.append({
                    "Participant": participant,
                    "Behavioral_Variable":
                        behavioral_variable,
                    "Wellness_Variable":
                        wellness_variable,
                    "r": np.nan,
                    "p": np.nan,
                    "N": n,
                    "Included": "No"
                })

                continue

            x = pair_df[behavioral_column]
            y = pair_df[wellness_column]

            if (
                x.nunique() < 2
                or y.nunique() < 2
            ):

                results.append({
                    "Participant": participant,
                    "Behavioral_Variable":
                        behavioral_variable,
                    "Wellness_Variable":
                        wellness_variable,
                    "r": np.nan,
                    "p": np.nan,
                    "N": n,
                    "Included": "No"
                })

                continue

            r, p = pearsonr(x, y)

            results.append({
                "Participant": participant,
                "Behavioral_Variable":
                    behavioral_variable,
                "Wellness_Variable":
                    wellness_variable,
                "r": r,
                "p": p,
                "N": n,
                "Included": "Yes"
            })


results_df = pd.DataFrame(
    results
)


# ============================================================
# FDR CORRECTION
# ============================================================

print("\n" + "=" * 70)
print("APPLYING BENJAMINI-HOCHBERG FDR CORRECTION")
print("=" * 70)

valid = (
    (results_df["Included"] == "Yes")
    & results_df["p"].notna()
)

results_df["p_FDR"] = np.nan
results_df["Significant_raw"] = False
results_df["Significant_FDR"] = False

if valid.sum() > 0:

    p_values = results_df.loc[
        valid,
        "p"
    ].values

    reject, corrected_p, _, _ = multipletests(
        p_values,
        method="fdr_bh"
    )

    results_df.loc[
        valid,
        "p_FDR"
    ] = corrected_p

    results_df.loc[
        valid,
        "Significant_raw"
    ] = p_values < 0.05

    results_df.loc[
        valid,
        "Significant_FDR"
    ] = reject


# ============================================================
# SAVE RESULTS
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
print("FULL-PERIOD ANALYSIS COMPLETE")
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
