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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "fdr"
)


# ============================================================
# 2. SETTINGS
# ============================================================

MIN_N = 10
ALPHA = 0.05


# ============================================================
# 3. INPUT FILES
# ============================================================

INPUT_FILES = {

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
# 4. BENJAMINI-HOCHBERG FDR
# ============================================================

def benjamini_hochberg(
    p_values,
    alpha=0.05
):
    """
    Benjamini-Hochberg FDR correction.

    Returns:
        adjusted p-values
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan
    )

    valid_mask = np.isfinite(
        p_values
    )

    if not valid_mask.any():
        return adjusted

    valid_p = p_values[
        valid_mask
    ]

    m = len(valid_p)

    order = np.argsort(
        valid_p
    )

    sorted_p = valid_p[
        order
    ]

    ranks = np.arange(
        1,
        m + 1
    )

    adjusted_sorted = (
        sorted_p * m / ranks
    )

    # Ensure monotonicity.
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    # Adjusted p-values cannot exceed 1.
    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0
    )

    # Restore original order.
    valid_adjusted = np.empty(
        m
    )

    valid_adjusted[
        order
    ] = adjusted_sorted

    adjusted[
        valid_mask
    ] = valid_adjusted

    return adjusted


# ============================================================
# 5. PROCESS ONE ANALYSIS
# ============================================================

def process_analysis(
    analysis_name,
    input_file
):

    print()
    print("-" * 60)

    print(
        f"Processing: {analysis_name}"
    )

    print("-" * 60)

    if not os.path.exists(
        input_file
    ):

        print(
            f"Input file not found:\n"
            f"{input_file}"
        )

        return None

    df = pd.read_csv(
        input_file
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "p",
        "N",
        "Included",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns in "
            f"{input_file}: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # VALID TESTS
    # --------------------------------------------------------

    valid = (
        (df["Included"] == "Yes")
        & (df["N"] >= MIN_N)
        & df["p"].notna()
        & np.isfinite(df["p"])
    )

    # --------------------------------------------------------
    # INITIALIZE FDR COLUMN
    # --------------------------------------------------------

    df["p_FDR"] = np.nan

    # --------------------------------------------------------
    # FDR FAMILY
    # --------------------------------------------------------
    #
    # FDR is applied separately within each analysis.
    #
    # same_day       -> one FDR family
    # lagged         -> one FDR family
    # seven_day      -> one FDR family
    # sensitivity_Z1 -> one FDR family
    #
    # --------------------------------------------------------

    p_values = df.loc[
        valid,
        "p"
    ].astype(float).to_numpy()

    if len(p_values) > 0:

        corrected = benjamini_hochberg(
            p_values,
            alpha=ALPHA
        )

        df.loc[
            valid,
            "p_FDR"
        ] = corrected

    # --------------------------------------------------------
    # RAW SIGNIFICANCE
    # --------------------------------------------------------

    df["Significant_raw"] = (
        valid
        & (df["p"] < ALPHA)
    )

    # --------------------------------------------------------
    # FDR SIGNIFICANCE
    # --------------------------------------------------------

    df["Significant_FDR"] = (
        valid
        & (df["p_FDR"] < ALPHA)
    )

    # --------------------------------------------------------
    # ANALYSIS LABEL
    # --------------------------------------------------------

    df.insert(
        0,
        "Analysis",
        analysis_name
    )

    return df


# ============================================================
# 6. RUN ALL ANALYSES
# ============================================================

all_results = []

for analysis_name, input_file in INPUT_FILES.items():

    result = process_analysis(
        analysis_name,
        input_file
    )

    if result is not None:

        all_results.append(
            result
        )


if not all_results:

    raise RuntimeError(
        "No analysis result files were found."
    )


# ============================================================
# 7. COMBINE RESULTS
# ============================================================

combined = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# 8. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 9. SAVE COMBINED RESULTS
# ============================================================

combined_file = os.path.join(
    OUTPUT_DIR,
    "all_final_fdr_results.csv"
)

combined.to_csv(
    combined_file,
    index=False
)


# ============================================================
# 10. SAVE FDR-SIGNIFICANT RESULTS
# ============================================================

significant = combined[
    combined["Significant_FDR"] == True
].copy()

significant_file = os.path.join(
    OUTPUT_DIR,
    "all_fdr_significant_results.csv"
)

significant.to_csv(
    significant_file,
    index=False
)


# ============================================================
# 11. SAVE EACH ANALYSIS SEPARATELY
# ============================================================

analysis_names = (
    combined["Analysis"]
    .dropna()
    .unique()
)

for analysis_name in analysis_names:

    analysis_df = combined[
        combined["Analysis"] == analysis_name
    ].copy()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{analysis_name}_fdr_results.csv"
    )

    analysis_df.to_csv(
        output_file,
        index=False
    )


# ============================================================
# 12. SUMMARY
# ============================================================

print()
print("=" * 60)
print("FDR CORRECTION COMPLETED")
print("=" * 60)


for analysis_name in analysis_names:

    analysis_df = combined[
        combined["Analysis"] == analysis_name
    ]

    valid_count = int(
        analysis_df["p_FDR"].notna().sum()
    )

    raw_count = int(
        analysis_df["Significant_raw"].sum()
    )

    fdr_count = int(
        analysis_df["Significant_FDR"].sum()
    )

    print()
    print(
        f"{analysis_name}:"
    )

    print(
        f"  Valid tests: {valid_count}"
    )

    print(
        f"  Raw significant: {raw_count}"
    )

    print(
        f"  FDR significant: {fdr_count}"
    )


print()
print(
    f"Combined results:\n"
    f"{combined_file}"
)

print(
    f"FDR-significant results:\n"
    f"{significant_file}"
)

print("=" * 60)
