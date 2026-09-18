import os
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MIN_N = 10
ALPHA = 0.05

INPUT_FILES = {
    "same_day": (
        "results/same_day/"
        "same_day_behavior_wellbeing_relationships.csv"
    ),

    "lagged": (
        "results/lagged/"
        "lagged_behavior_wellbeing_relationships.csv"
    ),

    "seven_day": (
        "results/seven_day/"
        "seven_day_history_wellbeing_relationships.csv"
    ),

    "sensitivity_Z1": (
        "results/sensitivity/"
        "sensitivity_Z1_same_day_relationships.csv"
    ),
}

OUTPUT_DIR = "results/fdr"


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================

def benjamini_hochberg(p_values, alpha=0.05):
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

    valid_mask = np.isfinite(p_values)

    if not valid_mask.any():
        return adjusted

    valid_p = p_values[valid_mask]

    m = len(valid_p)

    order = np.argsort(valid_p)

    sorted_p = valid_p[order]

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

    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0
    )

    valid_adjusted = np.empty(m)

    valid_adjusted[order] = adjusted_sorted

    adjusted[valid_mask] = valid_adjusted

    return adjusted


# ============================================================
# PROCESS ONE ANALYSIS
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

    if not os.path.exists(input_file):

        print(
            f"Input file not found:\n{input_file}"
        )

        return None

    df = pd.read_csv(
        input_file
    )

    if "p" not in df.columns:
        raise ValueError(
            f"'p' column not found in {input_file}"
        )

    if "N" not in df.columns:
        raise ValueError(
            f"'N' column not found in {input_file}"
        )

    if "Included" not in df.columns:
        raise ValueError(
            f"'Included' column not found in {input_file}"
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

    df["p_FDR"] = np.nan

    # --------------------------------------------------------
    # FDR FAMILY
    # --------------------------------------------------------
    #
    # FDR is applied across all valid participant-level
    # relationships within each analysis separately.
    #
    # This means:
    #
    # same_day       → one FDR family
    # lagged         → one FDR family
    # seven_day      → one FDR family
    # sensitivity_Z1 → one FDR family
    #
    # --------------------------------------------------------

    p_values = df.loc[
        valid,
        "p"
    ].astype(float).values

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
    # SIGNIFICANCE
    # --------------------------------------------------------

    df["Significant_raw"] = (
        valid
        & (df["p"] < ALPHA)
    )

    df["Significant_FDR"] = (
        valid
        & (df["p_FDR"] < ALPHA)
    )

    # --------------------------------------------------------
    # ADD ANALYSIS LABEL
    # --------------------------------------------------------

    df.insert(
        0,
        "Analysis",
        analysis_name
    )

    return df


# ============================================================
# RUN ALL ANALYSES
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
# COMBINE RESULTS
# ============================================================

combined = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

combined_file = os.path.join(
    OUTPUT_DIR,
    "all_final_fdr_results.csv"
)

combined.to_csv(
    combined_file,
    index=False
)


# ============================================================
# SAVE SIGNIFICANT RESULTS
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
# SAVE EACH ANALYSIS SEPARATELY
# ============================================================

for analysis_name in combined[
    "Analysis"
].dropna().unique():

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
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("FDR CORRECTION COMPLETED")
print("=" * 60)

for analysis_name in combined[
    "Analysis"
].dropna().unique():

    analysis_df = combined[
        combined["Analysis"] == analysis_name
    ]

    valid_count = (
        analysis_df["p_FDR"].notna()
    ).sum()

    raw_count = (
        analysis_df["Significant_raw"]
    ).sum()

    fdr_count = (
        analysis_df["Significant_FDR"]
    ).sum()

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
    f"Combined results:\n{combined_file}"
)

print(
    f"FDR-significant results:\n{significant_file}"
)

print("=" * 60)
