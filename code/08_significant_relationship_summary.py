import pandas as pd
from pathlib import Path

# ============================================================
# 08 - SUMMARY OF FDR-SIGNIFICANT RELATIONSHIPS
# ============================================================

INPUT_FILE = Path(
    "results/wellbeing_analysis/full_period_behavior_wellbeing_statistics.csv"
)

OUTPUT_DIR = Path("results/wellbeing_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SUMMARY OF FDR-SIGNIFICANT RELATIONSHIPS")
print("=" * 70)

# ------------------------------------------------------------
# Load analysis results
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Total relationship tests: {len(df)}")

# Keep only FDR-significant relationships
sig = df[df["Significant_FDR"] == True].copy()

print(f"FDR-significant relationships: {len(sig)}")

# ------------------------------------------------------------
# 1. Summary by behavioral variable
# ------------------------------------------------------------

behavior_summary = (
    sig.groupby("Behavioral_Variable")
    .agg(
        Significant_Relationships=("Participant", "count"),
        Participants=("Participant", "nunique"),
        Mean_Abs_r=("r", lambda x: x.abs().mean()),
        Mean_r=("r", "mean"),
    )
    .reset_index()
    .sort_values(
        ["Participants", "Significant_Relationships", "Mean_Abs_r"],
        ascending=False
    )
)

behavior_summary.to_csv(
    OUTPUT_DIR / "significant_behavior_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 2. Summary by well-being variable
# ------------------------------------------------------------

wellness_summary = (
    sig.groupby("Wellness_Variable")
    .agg(
        Significant_Relationships=("Participant", "count"),
        Participants=("Participant", "nunique"),
        Mean_Abs_r=("r", lambda x: x.abs().mean()),
        Mean_r=("r", "mean"),
    )
    .reset_index()
    .sort_values(
        ["Participants", "Significant_Relationships", "Mean_Abs_r"],
        ascending=False
    )
)

wellness_summary.to_csv(
    OUTPUT_DIR / "significant_wellness_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 3. Summary of behavioral × well-being relationships
# ------------------------------------------------------------

relationship_summary = (
    sig.groupby(["Behavioral_Variable", "Wellness_Variable"])
    .agg(
        Significant_Participants=("Participant", "nunique"),
        Mean_r=("r", "mean"),
        Mean_Abs_r=("r", lambda x: x.abs().mean()),
        Min_r=("r", "min"),
        Max_r=("r", "max"),
    )
    .reset_index()
    .sort_values(
        ["Significant_Participants", "Mean_Abs_r"],
        ascending=False
    )
)

relationship_summary.to_csv(
    OUTPUT_DIR / "significant_relationship_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 4. Print behavioral summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BY BEHAVIORAL VARIABLE")
print("=" * 70)

if len(behavior_summary) > 0:
    print(behavior_summary.to_string(index=False))
else:
    print("No FDR-significant behavioral relationships found.")

# ------------------------------------------------------------
# 5. Print well-being summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BY WELL-BEING VARIABLE")
print("=" * 70)

if len(wellness_summary) > 0:
    print(wellness_summary.to_string(index=False))
else:
    print("No FDR-significant well-being relationships found.")

# ------------------------------------------------------------
# 6. Print repeated relationships
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("
