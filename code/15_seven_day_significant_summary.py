import pandas as pd
from pathlib import Path

# ============================================================
# 7-DAY SIGNIFICANT RELATIONSHIP SUMMARY
# ============================================================

results_dir = Path("results/wellbeing_analysis")

input_file = (
    results_dir /
    "seven_day_history_wellbeing_relationships.csv"
)

# Load results
results = pd.read_csv(input_file)

# Keep only FDR-significant relationships
significant = results[
    results["Significant_FDR"] == True
].copy()

# ============================================================
# 1. SUMMARY BY BEHAVIORAL VARIABLE
# ============================================================

behavior_summary = (
    significant
    .groupby("Behavioral_Variable")
    .agg(
        Significant_Relationships=("Behavioral_Variable", "size"),
        Participants=("Participant", "nunique"),
        Mean_Abs_r=("r", lambda x: x.abs().mean()),
        Mean_r=("r", "mean")
    )
    .sort_values(
        "Significant_Relationships",
        ascending=False
    )
    .reset_index()
)

# ============================================================
# 2. SUMMARY BY WELL-BEING VARIABLE
# ============================================================

wellness_summary = (
    significant
    .groupby("Wellness_Variable")
    .agg(
        Significant_Relationships=("Wellness_Variable", "size"),
        Participants=("Participant", "nunique"),
        Mean_Abs_r=("r", lambda x: x.abs().mean()),
        Mean_r=("r", "mean")
    )
    .sort_values(
        "Significant_Relationships",
        ascending=False
    )
    .reset_index()
)

# ============================================================
# 3. MOST REPEATED BEHAVIOR × WELL-BEING RELATIONSHIPS
# ============================================================

relationship_summary = (
    significant
    .groupby(
        ["Behavioral_Variable", "Wellness_Variable"]
    )
    .agg(
        Significant_Relationships=("Participant", "size"),
        Participants=("Participant", "nunique"),
        Mean_Abs_r=("r", lambda x: x.abs().mean()),
        Mean_r=("r", "mean")
    )
    .sort_values(
        ["Significant_Relationships", "Mean_Abs_r"],
        ascending=[False, False]
    )
    .reset_index()
)

# Round correlations
for df in [
    behavior_summary,
    wellness_summary,
    relationship_summary
]:
    df["Mean_Abs_r"] = df["Mean_Abs_r"].round(3)
    df["Mean_r"] = df["Mean_r"].round(3)


# ============================================================
# 4. SAVE OUTPUTS
# ============================================================

behavior_file = (
    results_dir /
    "seven_day_significant_behavior_summary.csv"
)

wellness_file = (
    results_dir /
    "seven_day_significant_wellness_summary.csv"
)

relationship_file = (
    results_dir /
    "seven_day_significant_relationship_summary.csv"
)

behavior_summary.to_csv(
    behavior_file,
    index=False
)

wellness_summary.to_csv(
    wellness_file,
    index=False
)

relationship_summary.to_csv(
    relationship_file,
    index=False
)


# ============================================================
# 5. PRINT RESULTS
# ============================================================

print()
print("==============================================")
print("7-DAY SIGNIFICANT RELATIONSHIP SUMMARY")
print("==============================================")

print(
    f"FDR-significant relationships: {len(significant)}"
)

print()
print("BY BEHAVIORAL VARIABLE")
print("----------------------------------------------")
print(behavior_summary.to_string(index=False))

print()
print("BY WELL-BEING VARIABLE")
print("----------------------------------------------")
print(wellness_summary.to_string(index=False))

print()
print("MOST REPEATED BEHAVIOR × WELL-BEING RELATIONSHIPS")
print("----------------------------------------------")
print(relationship_summary.to_string(index=False))

print()
print("Files saved:")
print(behavior_file)
print(wellness_file)
print(relationship_file)

print("==============================================")
