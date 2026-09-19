import os
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "fdr",
    "all_final_fdr_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "final_summary"
)


# ============================================================
# 2. SETTINGS
# ============================================================

ALPHA = 0.05


# ============================================================
# 3. CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"FDR results file not found:\n{INPUT_FILE}"
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOAD RESULTS
# ============================================================

print("=" * 70)
print("FINAL FDR SUMMARY")
print("=" * 70)

print(
    f"Input:\n{INPUT_FILE}"
)

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# 5. BASIC CLEANING
# ============================================================

required_columns = [
    "Analysis",
    "Participant",
    "Behavioral_Variable",
    "Wellbeing_Variable",
    "r",
    "p",
    "p_FDR",
    "N",
    "Significant_FDR",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


df["r"] = pd.to_numeric(
    df["r"],
    errors="coerce"
)

df["p"] = pd.to_numeric(
    df["p"],
    errors="coerce"
)

df["p_FDR"] = pd.to_numeric(
    df["p_FDR"],
    errors="coerce"
)

df["N"] = pd.to_numeric(
    df["N"],
    errors="coerce"
)


# ============================================================
# 6. FDR-SIGNIFICANT RESULTS
# ============================================================

significant = df[
    (df["Significant_FDR"] == True)
    & df["p_FDR"].notna()
].copy()


if significant.empty:

    print()
    print(
        "No FDR-significant relationships were found."
    )

else:

    # --------------------------------------------------------
    # Absolute correlation
    # --------------------------------------------------------

    significant["Abs_r"] = (
        significant["r"].abs()
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    significant["Direction"] = np.where(
        significant["r"] > 0,
        "Positive",
        np.where(
            significant["r"] < 0,
            "Negative",
            "Zero"
        )
    )

    # --------------------------------------------------------
    # Sort by analysis and FDR p-value
    # --------------------------------------------------------

    significant = significant.sort_values(
        [
            "Analysis",
            "p_FDR",
            "Abs_r"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )


# ============================================================
# 7. TABLE 1 — ALL SIGNIFICANT RELATIONSHIPS
# ============================================================

table_relationships = significant[
    [
        "Analysis",
        "Participant",
        "Behavioral_Variable",
        "Wellbeing_Variable",
        "r",
        "Abs_r",
        "Direction",
        "p",
        "p_FDR",
        "N"
    ]
].copy()

table_relationships_file = os.path.join(
    OUTPUT_DIR,
    "TABLE_1_FDR_significant_relationships.csv"
)

table_relationships.to_csv(
    table_relationships_file,
    index=False
)


# ============================================================
# 8. TABLE 2 — SUMMARY BY ANALYSIS
# ============================================================

analysis_summary = []

for analysis in sorted(
    df["Analysis"].dropna().unique()
):

    analysis_df = df[
        df["Analysis"] == analysis
    ]

    valid_df = analysis_df[
        analysis_df["p_FDR"].notna()
    ]

    significant_df = analysis_df[
        analysis_df["Significant_FDR"] == True
    ]

    analysis_summary.append({

        "Analysis": analysis,

        "Total_Tests": len(
            analysis_df
        ),

        "Valid_Tests": len(
            valid_df
        ),

        "Raw_Significant": int(
            (
                valid_df["p"] < ALPHA
            ).sum()
        ),

        "FDR_Significant": len(
            significant_df
        ),

        "Unique_Participants_with_FDR_Significant":
            significant_df[
                "Participant"
            ].nunique(),

        "Mean_Abs_r_FDR_Significant":
            (
                significant_df["r"].abs().mean()
                if not significant_df.empty
                else np.nan
            ),

        "Mean_r_FDR_Significant":
            (
                significant_df["r"].mean()
                if not significant_df.empty
                else np.nan
            )
    })


analysis_summary_df = pd.DataFrame(
    analysis_summary
)

analysis_summary_file = os.path.join(
    OUTPUT_DIR,
    "TABLE_2_analysis_summary.csv"
)

analysis_summary_df.to_csv(
    analysis_summary_file,
    index=False
)


# ============================================================
# 9. TABLE 3 — SUMMARY BY BEHAVIORAL VARIABLE
# ============================================================

behavior_summary = (
    significant
    .groupby(
        [
            "Analysis",
            "Behavioral_Variable"
        ],
        dropna=False
    )
    .agg(
        FDR_Significant_Relationships=(
            "Participant",
            "count"
        ),

        Unique_Participants=(
            "Participant",
            "nunique"
        ),

        Mean_r=(
            "r",
            "mean"
        ),

        Mean_Abs_r=(
            "Abs_r",
            "mean"
        )
    )
    .reset_index()
)

behavior_summary_file = os.path.join(
    OUTPUT_DIR,
    "TABLE_3_behavior_summary.csv"
)

behavior_summary.to_csv(
    behavior_summary_file,
    index=False
)


# ============================================================
# 10. TABLE 4 — SUMMARY BY WELLBEING VARIABLE
# ============================================================

wellbeing_summary = (
    significant
    .groupby(
        [
            "Analysis",
            "Wellbeing_Variable"
        ],
        dropna=False
    )
    .agg(
        FDR_Significant_Relationships=(
            "Participant",
            "count"
        ),

        Unique_Participants=(
            "Participant",
            "nunique"
        ),

        Mean_r=(
            "r",
            "mean"
        ),

        Mean_Abs_r=(
            "Abs_r",
            "mean"
        )
    )
    .reset_index()
)

wellbeing_summary_file = os.path.join(
    OUTPUT_DIR,
    "TABLE_4_wellbeing_summary.csv"
)

wellbeing_summary.to_csv(
    wellbeing_summary_file,
    index=False
)


# ============================================================
# 11. PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY COMPLETED")
print("=" * 70)

print()

for _, row in analysis_summary_df.iterrows():

    print(
        f"{row['Analysis']}:"
    )

    print(
        f"  Valid tests: "
        f"{int(row['Valid_Tests'])}"
    )

    print(
        f"  Raw significant: "
        f"{int(row['Raw_Significant'])}"
    )

    print(
        f"  FDR significant: "
        f"{int(row['FDR_Significant'])}"
    )

    print()


print(
    f"Total FDR-significant relationships: "
    f"{len(significant)}"
)

print()
print("Saved files:")
print(
    f"  {table_relationships_file}"
)
print(
    f"  {analysis_summary_file}"
)
print(
    f"  {behavior_summary_file}"
)
print(
    f"  {wellbeing_summary_file}"
)

print("=" * 70)
