```python
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ============================================================
# 0. PATHS
# ============================================================

# Repository root = one level above the code folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "pmdata")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "week2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_N = 10


# ============================================================
# 1. VARIABLES
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

WELLNESS_VARIABLES = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]


# ============================================================
# 2. BENJAMINI-HOCHBERG FDR
# ============================================================

def bh_fdr(p_values):
    """
    Benjamini-Hochberg False Discovery Rate correction.
    """

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    if n == 0:
        return np.array([])

    order = np.argsort(p_values)

    ranked_p = p_values[order]

    q_values = (
        ranked_p * n /
        np.arange(1, n + 1)
    )

    # Ensure monotonicity
    q_values = np.minimum.accumulate(
        q_values[::-1]
    )[::-1]

    q_values = np.clip(
        q_values,
        0,
        1
    )

    adjusted = np.empty(n)

    adjusted[order] = q_values

    return adjusted


# ============================================================
# 3. LOAD PARTICIPANT DATA
# ============================================================

participant_data = {}

print("=" * 70)
print("LOADING PMDATA")
print("=" * 70)

print(f"Data directory: {DATA_DIR}")
print()

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"PMData directory not found:\n{DATA_DIR}"
    )


for participant in sorted(os.listdir(DATA_DIR)):

    participant_path = os.path.join(
        DATA_DIR,
        participant
    )

    if not os.path.isdir(participant_path):
        continue

    # New repository structure:
    # data/pmdata/p01/p01_daily_merged.csv

    csv_files = []

    for file in os.listdir(participant_path):

        if file.lower().endswith(".csv"):

            csv_files.append(
                os.path.join(
                    participant_path,
                    file
                )
            )

    if not csv_files:
        continue

    dataframes = []

    for file in sorted(csv_files):

        try:

            df = pd.read_csv(file)

            dataframes.append(df)

            print(
                f"{participant}: "
                f"{os.path.basename(file)} "
                f"({len(df)} rows)"
            )

        except Exception as e:

            print(
                f"Could not read {file}: {e}"
            )

    if dataframes:

        participant_df = pd.concat(
            dataframes,
            ignore_index=True
        )

        # Remove exact duplicate rows
        participant_df = participant_df.drop_duplicates()

        participant_data[participant] = participant_df


print()
print(
    "Participants loaded:",
    len(participant_data)
)

print()


# ============================================================
# 4. PARTICIPANT-LEVEL PEARSON CORRELATIONS
# ============================================================

results = []

print("=" * 70)
print("CALCULATING PARTICIPANT-LEVEL CORRELATIONS")
print("=" * 70)


for participant, df in participant_data.items():

    for objective in OBJECTIVE_VARIABLES:

        if objective not in df.columns:
            continue

        for wellness in WELLNESS_VARIABLES:

            if wellness not in df.columns:
                continue

            # Keep only complete pairs
            temp = df[
                [objective, wellness]
            ].dropna()

            n = len(temp)

            # ------------------------------------------------
            # Minimum sample size
            # ------------------------------------------------

            if n < MIN_N:

                results.append({

                    "Participant": participant,

                    "Objective_Variable": objective,

                    "Wellness_Variable": wellness,

                    "r": np.nan,

                    "p": np.nan,

                    "N": n,

                    "Included": "No"

                })

                continue

            # ------------------------------------------------
            # Check for constant variables
            # ------------------------------------------------

            if (
                temp[objective].nunique() < 2
                or
                temp[wellness].nunique() < 2
            ):

                results.append({

                    "Participant": participant,

                    "Objective_Variable": objective,

                    "Wellness_Variable": wellness,

                    "r": np.nan,

                    "p": np.nan,

                    "N": n,

                    "Included": "No"

                })

                continue

            # ------------------------------------------------
            # Pearson correlation
            # ------------------------------------------------

            r, p = pearsonr(
                temp[objective],
                temp[wellness]
            )

            results.append({

                "Participant": participant,

                "Objective_Variable": objective,

                "Wellness_Variable": wellness,

                "r": r,

                "p": p,

                "N": n,

                "Included": "Yes"

            })


participant_results = pd.DataFrame(results)


# ============================================================
# 5. FDR CORRECTION
# ============================================================

print("=" * 70)
print("APPLYING BENJAMINI-HOCHBERG FDR CORRECTION")
print("=" * 70)


valid = participant_results["p"].notna()

participant_results["p_FDR"] = np.nan

participant_results.loc[valid, "p_FDR"] = bh_fdr(
    participant_results.loc[valid, "p"].values
)


participant_results["Significant_raw"] = (
    participant_results["p"] < 0.05
)

participant_results["Significant_FDR"] = (
    participant_results["p_FDR"] < 0.05
)


# ============================================================
# 6. RELATIONSHIP-LEVEL SUMMARY
# ============================================================

summary = []

included_results = participant_results[
    participant_results["Included"] == "Yes"
]


grouped = included_results.groupby(
    [
        "Objective_Variable",
        "Wellness_Variable"
    ]
)


for (
    objective,
    wellness
), group in grouped:

    r_values = group["r"].dropna()

    if len(r_values) == 0:
        continue

    positive = (
        r_values > 0
    ).sum()

    negative = (
        r_values < 0
    ).sum()

    n_participants = len(r_values)

    positive_percent = (
        positive /
        n_participants *
        100
    )

    negative_percent = (
        negative /
        n_participants *
        100
    )

    raw_sig_percent = (
        group["Significant_raw"].mean()
        * 100
    )

    fdr_sig_percent = (
        group["Significant_FDR"].mean()
        * 100
    )

    summary.append({

        "Objective_Variable": objective,

        "Wellness_Variable": wellness,

        "N_Participants": n_participants,

        "Mean_r": r_values.mean(),

        "Median_r": r_values.median(),

        "SD_r": r_values.std(),

        "Mean_abs_r": r_values.abs().mean(),

        "Median_abs_r": r_values.abs().median(),

        "Positive_n": positive,

        "Positive_%": positive_percent,

        "Negative_n": negative,

        "Negative_%": negative_percent,

        "Raw_p<0.05_%": raw_sig_percent,

        "FDR_q<0.05_%": fdr_sig_percent,

        "Median_N": group["N"].median(),

        "Min_N": group["N"].min(),

        "Max_N": group["N"].max()

    })


summary_df = pd.DataFrame(summary)


# ============================================================
# 7. EXPLORATORY RELATIONSHIP RANKING
# ============================================================

if len(summary_df) > 0:

    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------

    summary_df["Effect_Size_Rank"] = (
        summary_df["Mean_abs_r"].rank(
            pct=True
        )
    )

    # --------------------------------------------------------
    # Sign consistency
    #
    # If mean r >= 0:
    #     positive percentage = consistency
    #
    # If mean r < 0:
    #     negative percentage = consistency
    # --------------------------------------------------------

    summary_df["Consistency_Rank"] = np.where(
        summary_df["Mean_r"] >= 0,
        summary_df["Positive_%"] / 100,
        summary_df["Negative_%"] / 100
    )

    # --------------------------------------------------------
    # FDR significance
    # --------------------------------------------------------

    summary_df["FDR_Rank"] = (
        summary_df["FDR_q<0.05_%"] / 100
    )

    # --------------------------------------------------------
    # Exploratory combined score
    #
    # IMPORTANT:
    # This is exploratory ranking, not a statistical test.
    # --------------------------------------------------------

    summary_df["Exploratory_Score"] = (

        0.40 *
        summary_df["Effect_Size_Rank"]

        +

        0.35 *
        summary_df["Consistency_Rank"]

        +

        0.25 *
        summary_df["FDR_Rank"]

    )

    summary_df["Rank"] = (
        summary_df["Exploratory_Score"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    summary_df = summary_df.sort_values(
        "Rank"
    )


# ============================================================
# 8. EXCLUDED RESULTS
# ============================================================

excluded_df = participant_results[
    participant_results["Included"] == "No"
].copy()


# ============================================================
# 9. SAVE RESULTS
# ============================================================

participant_output = os.path.join(
    OUTPUT_DIR,
    "PMData_Participant_Level_Correlations_MIN_N10.xlsx"
)

summary_output = os.path.join(
    OUTPUT_DIR,
    "PMData_Relationship_Summary_MIN_N10.xlsx"
)

excluded_output = os.path.join(
    OUTPUT_DIR,
    "PMData_Excluded_Below_N10.xlsx"
)


participant_results.to_excel(
    participant_output,
    index=False
)

summary_df.to_excel(
    summary_output,
    index=False
)

excluded_df.to_excel(
    excluded_output,
    index=False
)


# ============================================================
# 10. FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()
print(
    f"Participants analyzed: "
    f"{len(participant_data)}"
)

print(
    f"Valid correlation tests: "
    f"{valid.sum()}"
)

print(
    f"FDR-significant relationships: "
    f"{participant_results['Significant_FDR'].sum()}"
)

print()
print("Output files:")
print()
print(participant_output)
print(summary_output)
print(excluded_output)

print()
print("=" * 70)
```
