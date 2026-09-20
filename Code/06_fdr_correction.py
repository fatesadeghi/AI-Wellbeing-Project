import os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

FDR_DIR = os.path.join(
    RESULTS_DIR,
    "fdr"
)

os.makedirs(
    FDR_DIR,
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
        "seven_day_history_wellbeing_relationships.csv"
    ),

    "sensitivity_Z1": os.path.join(
        RESULTS_DIR,
        "sensitivity",
        "sensitivity_Z1_same_day_relationships.csv"
    ),
}


# ============================================================
# 3. SETTINGS
# ============================================================

MIN_N = 10
FDR_ALPHA = 0.05


# ============================================================
# 4. FUNCTION
# ============================================================

def apply_fdr(
    df,
    analysis_name
):

    df = df.copy()

    # --------------------------------------------------------
    # Ensure required columns exist
    # --------------------------------------------------------

    required_columns = [
        "p",
        "N",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{analysis_name}: missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    df["p"] = pd.to_numeric(
        df["p"],
        errors="coerce"
    )

    df["N"] = pd.to_numeric(
        df["N"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Valid tests
    # --------------------------------------------------------

    valid_mask = (
        (df["N"] >= MIN_N)
        & df["p"].notna()
        & np.isfinite(df["p"])
        & (df["p"] >= 0)
        & (df["p"] <= 1)
    )

    valid_df = df.loc[
        valid_mask
    ].copy()

    # --------------------------------------------------------
    # Initialize FDR columns
    # --------------------------------------------------------

    df["p_FDR"] = np.nan

    df["Significant_raw"] = False

    df["Significant_FDR"] = False

    # --------------------------------------------------------
    # Benjamini-Hochberg FDR
    # --------------------------------------------------------

    if len(valid_df) > 0:

        reject, corrected_p, _, _ = multipletests(
            valid_df["p"].to_numpy(),
            alpha=FDR_ALPHA,
            method="fdr_bh"
        )

        df.loc[
            valid_df.index,
            "p_FDR"
        ] = corrected_p

        df.loc[
            valid_df.index,
            "Significant_raw"
        ] = (
            valid_df["p"].to_numpy()
            < FDR_ALPHA
        )

        df.loc[
            valid_df.index,
            "Significant_FDR"
        ] = reject

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    df["Analysis"] = analysis_name

    return (
        df,
        len(valid_df),
        int(
            df["Significant_raw"].sum()
        ),
        int(
            df["Significant_FDR"].sum()
        ),
    )


# ============================================================
# 5. HEADER
# ============================================================

print()
print("=" * 70)
print("FDR CORRECTION")
print("=" * 70)


# ============================================================
# 6. PROCESS EACH ANALYSIS
# ============================================================

all_results = []

summary_rows = []


for analysis_name, input_file in ANALYSIS_FILES.items():

    print()
    print("=" * 70)
    print(
        f"Processing: {analysis_name}"
    )
    print("=" * 70)

    if not os.path.exists(input_file):

        print(
            "WARNING: File not found:"
        )

        print(
            input_file
        )

        continue

    df = pd.read_csv(
        input_file
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    processed_df, valid_n, raw_sig, fdr_sig = apply_fdr(
        df,
        analysis_name
    )

    output_file = os.path.join(
        FDR_DIR,
        f"{analysis_name}_fdr_results.csv"
    )

    processed_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Valid tests (N >= {MIN_N}): "
        f"{valid_n}"
    )

    print(
        f"Raw significant: "
        f"{raw_sig}"
    )

    print(
        f"FDR significant: "
        f"{fdr_sig}"
    )

    print(
        f"Saved: {output_file}"
    )

    all_results.append(
        processed_df
    )

    summary_rows.append(
        {
            "Analysis": analysis_name,
            "Total_Tests": len(df),
            "Valid_Tests_N>=10": valid_n,
            "Raw_Significant_p<0.05": raw_sig,
            "FDR_Significant_pFDR<0.05": fdr_sig,
        }
    )


# ============================================================
# 7. COMBINE RESULTS
# ============================================================

if all_results:

    combined_results = pd.concat(
        all_results,
        ignore_index=True
    )

else:

    raise RuntimeError(
        "No analysis result files were available "
        "for FDR correction."
    )


combined_output = os.path.join(
    FDR_DIR,
    "all_final_fdr_results.csv"
)

combined_results.to_csv(
    combined_output,
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_output = os.path.join(
    FDR_DIR,
    "fdr_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)


total_fdr_significant = int(
    combined_results[
        "Significant_FDR"
    ].sum()
)


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("Combined FDR results saved:")
print(combined_output)
print("=" * 70)

print()
print("FDR SUMMARY")

if not summary_df.empty:
    print(
        summary_df.to_string(
            index=False
        )
    )

print()
print(
    f"Saved summary: "
    f"{summary_output}"
)

print()
print("=" * 70)
print(
    "TOTAL FDR-SIGNIFICANT RELATIONSHIPS: "
    f"{total_fdr_significant}"
)
print("=" * 70)
