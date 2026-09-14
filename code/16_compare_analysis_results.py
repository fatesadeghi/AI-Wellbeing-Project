import pandas as pd
from pathlib import Path

# ============================================================
# COMPARE SAME-DAY, LAGGED, AND 7-DAY HISTORY RESULTS
# ============================================================

results_dir = Path("results/wellbeing_analysis")

same_day_file = (
    results_dir /
    "final_daily_deviation_wellbeing_results.csv"
)

lagged_file = (
    results_dir /
    "final_lagged_deviation_wellbeing_results.csv"
)

seven_day_file = (
    results_dir /
    "seven_day_history_wellbeing_relationships.csv"
)

# ============================================================
# 1. LOAD RESULTS
# ============================================================

same_day = pd.read_csv(same_day_file)
lagged = pd.read_csv(lagged_file)
seven_day = pd.read_csv(seven_day_file)

# Keep only FDR-significant relationships
same_day_sig = same_day[
    same_day["Significant_FDR"] == True
].copy()

lagged_sig = lagged[
    lagged["Significant_FDR"] == True
].copy()

seven_day_sig = seven_day[
    seven_day["Significant_FDR"] == True
].copy()


# ============================================================
# 2. DEFINE RELATIONSHIP ID
# ============================================================

key_columns = [
    "Participant",
    "Behavioral_Variable",
    "Wellness_Variable"
]

same_day_sig["Relationship_ID"] = (
    same_day_sig[key_columns]
    .astype(str)
    .agg(" | ".join, axis=1)
)

lagged_sig["Relationship_ID"] = (
    lagged_sig[key_columns]
    .astype(str)
    .agg(" | ".join, axis=1)
)

seven_day_sig["Relationship_ID"] = (
    seven_day_sig[key_columns]
    .astype(str)
    .agg(" | ".join, axis=1)
)


# ============================================================
# 3. CREATE SETS
# ============================================================

same_set = set(same_day_sig["Relationship_ID"])
lagged_set = set(lagged_sig["Relationship_ID"])
seven_set = set(seven_day_sig["Relationship_ID"])


# ============================================================
# 4. OVERLAPS
# ============================================================

all_three = same_set & lagged_set & seven_set

same_seven = same_set & seven_set
same_lagged = same_set & lagged_set
seven_lagged = seven_set & lagged_set

any_two_or_more = (
    (same_set & lagged_set)
    | (same_set & seven_set)
    | (lagged_set & seven_set)
)


# ============================================================
# 5. CREATE COMPARISON TABLE
# ============================================================

all_relationships = sorted(
    same_set | lagged_set | seven_set
)

comparison = pd.DataFrame({
    "Relationship_ID": all_relationships
})

comparison[
    [
        "Participant",
        "Behavioral_Variable",
        "Wellness_Variable"
    ]
] = comparison["Relationship_ID"].str.split(
    " | ",
    expand=True
)


# Presence in each analysis
comparison["Same_day"] = comparison["Relationship_ID"].isin(
    same_set
)

comparison["One_day_lagged"] = comparison["Relationship_ID"].isin(
    lagged_set
)

comparison["Previous_7_days"] = comparison["Relationship_ID"].isin(
    seven_set
)


# Number of analyses in which the relationship was significant
comparison["Number_of_Analyses"] = (
    comparison[
        [
            "Same_day",
            "One_day_lagged",
            "Previous_7_days"
        ]
    ]
    .sum(axis=1)
)


# ============================================================
# 6. ADD r AND FDR q FOR EACH ANALYSIS
# ============================================================

same_stats = same_day_sig[
    key_columns + ["r", "p_FDR"]
].rename(columns={
    "r": "Same_day_r",
    "p_FDR": "Same_day_FDR_q"
})

lagged_stats = lagged_sig[
    key_columns + ["r", "p_FDR"]
].rename(columns={
    "r": "Lagged_r",
    "p_FDR": "Lagged_FDR_q"
})

seven_stats = seven_day_sig[
    key_columns + ["r", "p_FDR"]
].rename(columns={
    "r": "Seven_day_r",
    "p_FDR": "Seven_day_FDR_q"
})


comparison = comparison.merge(
    same_stats,
    on=key_columns,
    how="left"
)

comparison = comparison.merge(
    lagged_stats,
    on=key_columns,
    how="left"
)

comparison = comparison.merge(
    seven_stats,
    on=key_columns,
    how="left"
)


# ============================================================
# 7. SORT
# ============================================================

comparison = comparison.sort_values(
    [
        "Number_of_Analyses",
        "Participant",
        "Behavioral_Variable",
        "Wellness_Variable"
    ],
    ascending=[False, True, True, True]
)


# ============================================================
# 8. SAVE FULL COMPARISON
# ============================================================

comparison_file = (
    results_dir /
    "comparison_same_day_lagged_7day.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 9. SAVE REPEATED RELATIONSHIPS
# ============================================================

repeated = comparison[
    comparison["Number_of_Analyses"] >= 2
].copy()

repeated_file = (
    results_dir /
    "repeated_relationships_across_analyses.csv"
)

repeated.to_csv(
    repeated_file,
    index=False
)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print()
print("==============================================")
print("COMPARISON OF THREE ANALYSES")
print("==============================================")

print()
print("FDR-significant relationships:")
print(f"Same-day:        {len(same_set)}")
print(f"One-day lagged:  {len(lagged_set)}")
print(f"Previous 7 days: {len(seven_set)}")

print()
print("OVERLAPPING RELATIONSHIPS")
print("----------------------------------------------")
print(f"Same-day + 7-day:       {len(same_seven)}")
print(f"Same-day + lagged:      {len(same_lagged)}")
print(f"7-day + lagged:         {len(seven_lagged)}")
print(f"All three analyses:     {len(all_three)}")
print(f"Significant in >=2:     {len(any_two_or_more)}")

print()
print("Files saved:")
print(comparison_file)
print(repeated_file)

print("==============================================")
