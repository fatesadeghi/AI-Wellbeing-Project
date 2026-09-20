import os
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

# Current file:
# AI-Wellbeing-Project/Code/06_fdr_correction.py
#
# Therefore BASE_DIR = AI-Wellbeing-Project

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "fdr"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

ANALYSIS_FILES = {
    "same_day": os.path.join(
        RESULTS_DIR,
        "same_day",
        "same_day_behavior_wellbeing_relationships.csv"
    ),

    "lagged": os.path.join(
        RESULTS_DIR,
        "lagged",
        "lagged_behavior_wellbeing_relationships.csv"
    ),

    "seven_day": os.path.join(
        RESULTS_DIR,
        "seven_day",
        "seven_day_behavior_wellbeing_relationships.csv"
    ),

    "sensitivity_Z1": os.path.join(
        RESULTS_DIR,
        "sensitivity",
        "sensitivity_Z1_same_day_relationships.csv"
    )
}


# ============================================================
# 3. FDR FUNCTION
# ============================================================

def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted p-values in the original order.
    """

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    if n == 0:
        return np.array([])

    order = np.argsort(p_values)

    ranked_p = p_values[order]

    adjusted = np.empty(n)

    cumulative_min = 1.0

    for i in range(n - 1, -1, -1):

        rank = i + 1

        value = ranked_p[i] * n / rank

        cumulative_min = min(
            cumulative_min,
            value
        )

        adjusted[i] = cumulative_min

    result = np.empty(n)

    result[order] = adjusted

    return result


# ============================================================
# 4. PROCESS EACH ANALYSIS FAMILY
# ============================================================

all_results = []

summary_rows = []


for analysis_name, file_path in ANALYSIS_FILES.items():

    print("\n" + "=" * 70)
    print(f"Processing: {analysis_name}")
    print("=" * 70)

    if not os.path.exists(file_path):

        print(
            f"WARNING: File not found:\n{file_path}"
        )

        continue

    df = pd.read_csv(file_path)

    print(
        f"Rows loaded: {len(df)}"
    )

    # --------------------------------------------------------
    # Ensure required columns exist
    # --------------------------------------------------------

    required_columns = [
        "r",
        "p",
        "N"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        print(
            f"WARNING: Missing columns: {missing_columns}"
        )

        continue

    # --------------------------------------------------------
    # Valid statistical tests
    # --------------------------------------------------------

    valid_mask = (
        df["p"].notna()
        &
        np.isfinite(
            pd.to_numeric(
                df["p"],
                errors="coerce"
            )
        )
        &
        (df["N"] >= 10)
    )

    valid_df = df.loc[
        valid_mask
    ].copy()

    print(
        f"Valid tests (N >= 10): {len(valid_df)}"
    )

    if len(valid_df) == 0:

        df["p_FDR"] = np.nan
        df["Significant_raw"] = False
        df["Significant_FDR"] = False

        df["Analysis"] = analysis_name

        all_results.append(df)

        continue

    # --------------------------------------------------------
    # Raw significance
    # --------------------------------------------------------

    valid_df["Significant_raw"] = (
        valid_df["p"] < 0.05
    )

    raw_significant = int(
        valid_df["Significant_raw"].sum()
    )

    # --------------------------------------------------------
    # Benjamini-Hochberg FDR
    # Applied separately within each analysis family
    # --------------------------------------------------------

    valid_df["p_FDR"] = benjamini_hochberg(
        valid_df["p"].values
    )

    valid_df["Significant_FDR"] = (
        valid_df["p_FDR"] < 0.05
    )

    fdr_significant = int(
        valid_df["Significant_FDR"].sum()
    )

    # --------------------------------------------------------
    # Put corrected values back into original dataframe
    # --------------------------------------------------------

    df["p_FDR"] = np.nan
    df["Significant_raw"] = False
    df["Significant_FDR"] = False

    df.loc[
        valid_df.index,
        "p_FDR"
    ] = valid_df["p_FDR"]

    df.loc[
        valid_df.index,
        "Significant_raw"
    ] = valid_df["Significant_raw"]

    df.loc[
        valid_df.index,
        "Significant_FDR"
    ] = valid_df["Significant_FDR"]

    df["Analysis"] = analysis_name

    # --------------------------------------------------------
    # Save corrected results for this analysis
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{analysis_name}_fdr_results.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Raw significant: {raw_significant}"
    )

    print(
        f"FDR significant: {fdr_significant}"
    )

    print(
        f"Saved: {output_file}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows.append({
        "Analysis": analysis_name,
        "Total_Tests": len(df),
        "Valid_Tests_N>=10": len(valid_df),
        "Raw_Significant_p<0.05": raw_significant,
        "FDR_Significant_pFDR<0.05": fdr_significant
    })

    all_results.append(df)


# ============================================================
# 5. COMBINE ALL RESULTS
# ============================================================

if all_results:

    all_final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    combined_file = os.path.join(
        OUTPUT_DIR,
        "all_final_fdr_results.csv"
    )

    all_final_results.to_csv(
        combined_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("Combined FDR results saved:")
    print(combined_file)


# ============================================================
# 6. SAVE SUMMARY
# ============================================================

if summary_rows:

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "fdr_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\nFDR SUMMARY")
    print(summary_df.to_string(index=False))

    print(
        f"\nSaved summary: {summary_file}"
    )


# ============================================================
# 7. FINAL TOTAL
# ============================================================

if summary_rows:

    total_fdr = sum(
        row["FDR_Significant_pFDR<0.05"]
        for row in summary_rows
    )

    print("\n" + "=" * 70)
    print(
        f"TOTAL FDR-SIGNIFICANT RELATIONSHIPS: {total_fdr}"
    )
    print("=" * 70)
