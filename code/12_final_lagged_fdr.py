import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Final FDR correction for one-day lagged analysis
# Behavioral Deviation (Day t) -> Well-being (Day t+1)
# --------------------------------------------------

INPUT = Path(
    "results/wellbeing_analysis/lagged_deviation_wellbeing_relationships.csv"
)

OUTPUT = Path(
    "results/wellbeing_analysis/final_lagged_deviation_wellbeing_results.csv"
)

df = pd.read_csv(INPUT)

# Valid tests
valid = (
    (df["Included"] == "Yes") &
    (df["p"].notna())
)

p_values = df.loc[valid, "p"].values
n = len(p_values)

df["p_FDR"] = np.nan
df["Significant_FDR"] = False

# --------------------------------------------------
# Benjamini-Hochberg FDR
# --------------------------------------------------

if n > 0:

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    ranks = np.arange(1, n + 1)

    adjusted = sorted_p * n / ranks

    # Make adjusted p-values monotonic
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(adjusted, 1.0)

    # Return to original order
    p_fdr = np.empty(n)
    p_fdr[order] = adjusted

    df.loc[valid, "p_FDR"] = p_fdr

    df.loc[valid, "Significant_FDR"] = (
        p_fdr < 0.05
    )

# --------------------------------------------------
# Save final results
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\nFINAL LAGGED FDR ANALYSIS")
print("--------------------------------------")

print(
    f"Total relationship tests: {len(df)}"
)

print(
    f"Included tests (N >= 10): {valid.sum()}"
)

print(
    f"Raw significant (p < 0.05): "
    f"{(df.loc[valid, 'p'] < 0.05).sum()}"
)

print(
    f"FDR significant (q < 0.05): "
    f"{df['Significant_FDR'].sum()}"
)

print("\nFDR-significant lagged relationships:")

print(
    df[df["Significant_FDR"]]
    .sort_values("p_FDR")
    .to_string(index=False)
)

print(
    f"\nSaved to: {OUTPUT}"
)
