import os
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

# Current file:
# AI-Wellbeing-Project/Code/07_final_summary.py

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

FDR_FILE = os.path.join(
    RESULTS_DIR,
    "fdr",
    "all_final_fdr_results.csv"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "Final_Summary"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD FINAL FDR RESULTS
# ============================================================

if not os.path.exists(FDR_FILE):
    raise FileNotFoundError(
        f"FDR results file not found:\n{FDR_FILE}"
    )

df = pd.read_csv(FDR_FILE)

print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Rows loaded: {len(df)}"
)


# ============================================================
# 3. BASIC FINAL SUMMARY
# ============================================================

total_tests = len(df)

valid_tests = (
    df["p"].notna()
    &
    df["N"].notna()
    &
    (df["N"] >= 10)
).sum()

raw_significant = (
    df["Significant_raw"] == True
).sum()

fdr_significant = (
    df["Significant_FDR"] == True
).sum()


summary = pd.DataFrame([{
    "Total_Tests": total_tests,
    "Valid_Tests_N>=10": valid_tests,
    "Raw_Significant_p<0.05": raw_significant,
    "FDR_Significant_pFDR<0.05": fdr_significant
}])


# ============================================================
# 4. TABLE 1 — FDR-SIGNIFICANT RELATIONSHIPS
# ============================================================

fdr_df = df[
    df["Significant_FDR"] == True
].copy()

table_1_columns = [
    col
    for col in [
        "Analysis",
        "Participant",
        "Objective_Variable",
        "Wellness_Variable",
        "r",
        "p",
        "N",
        "p_FDR"
    ]
    if col in fdr_df.columns
]

table_1 = fdr_df[
    table_1_columns
].copy()

table_1 = table_1.sort_values(
    by=[
        "Analysis",
        "Participant"
    ]
)


# ============================================================
# 5. TABLE 2 — SUMMARY BY ANALYSIS
# ============================================================

table_2 = (
    df.groupby("Analysis")
    .agg(
        Total_Tests=("Analysis", "size"),
        Valid_Tests_N10=(
            "N",
            lambda x: (x >= 10).sum()
        ),
        Raw_Significant=(
            "Significant_raw",
            "sum"
        ),
        FDR_Significant=(
            "Significant_FDR",
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# 6. TABLE 3 — SUMMARY BY WELLBEING VARIABLE
# ============================================================

table_3 = (
    fdr_df.groupby("Wellness_Variable")
    .agg(
        FDR_Significant_Relationships=(
            "Wellness_Variable",
            "size"
        ),
        Participants=(
            "Participant",
            "nunique"
        ),
        Mean_Abs_r=(
            "r",
            lambda x: x.abs().mean()
        )
    )
    .reset_index()
)

table_3 = table_3.sort_values(
    by="FDR_Significant_Relationships",
    ascending=False
)


# ============================================================
# 7. TABLE 4 — SUMMARY BY BEHAVIORAL VARIABLE
# ============================================================

table_4 = (
    fdr_df.groupby("Objective_Variable")
    .agg(
        FDR_Significant_Relationships=(
            "Objective_Variable",
            "size"
        ),
        Participants=(
            "Participant",
            "nunique"
        ),
        Analyses=(
            "Analysis",
            "nunique"
        ),
        Mean_Abs_r=(
            "r",
            lambda x: x.abs().mean()
        )
    )
    .reset_index()
)

table_4 = table_4.sort_values(
    by="FDR_Significant_Relationships",
    ascending=False
)


# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "final_summary.csv"
)

table_1_file = os.path.join(
    OUTPUT_DIR,
    "TABLE_1_FDR_significant_relationships.csv"
)

table_2_file = os.path.join(
    OUTPUT_DIR,
    "TABLE_2_summary_by_analysis.csv"
)

table_3_file = os.path.join(
    OUTPUT_DIR,
    "TABLE_3_summary_by_wellbeing.csv"
)

table_4_file = os.path.join(
    OUTPUT_DIR,
    "TABLE_4_summary_by_behavior.csv"
)


summary.to_csv(
    summary_file,
    index=False
)

table_1.to_csv(
    table_1_file,
    index=False
)

table_2.to_csv(
    table_2_file,
    index=False
)

table_3.to_csv(
    table_3_file,
    index=False
)

table_4.to_csv(
    table_4_file,
    index=False
)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\nFINAL COUNTS")
print("-" * 70)

print(
    f"Total tests: {total_tests}"
)

print(
    f"Valid tests (N >= 10): {valid_tests}"
)

print(
    f"Raw significant: {raw_significant}"
)

print(
    f"FDR significant: {fdr_significant}"
)

print("\nFiles saved:")
print(summary_file)
print(table_1_file)
print(table_2_file)
print(table_3_file)
print(table_4_file)

print("\n" + "=" * 70)
print("FINAL SUMMARY COMPLETE")
print("=" * 70)
