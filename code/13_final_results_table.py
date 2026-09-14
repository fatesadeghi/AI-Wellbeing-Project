import pandas as pd
from pathlib import Path

# Paths
results_dir = Path("results/wellbeing_analysis")

same_day_file = results_dir / "final_daily_deviation_wellbeing_results.csv"
lagged_file = results_dir / "final_lagged_deviation_wellbeing_results.csv"

# Load results
same_day = pd.read_csv(same_day_file)
lagged = pd.read_csv(lagged_file)

# Keep only FDR-significant relationships
same_day_sig = same_day[same_day["Significant_FDR"] == True].copy()
lagged_sig = lagged[lagged["Significant_FDR"] == True].copy()

# Add analysis type
same_day_sig["Analysis"] = "Same-day"
lagged_sig["Analysis"] = "One-day lagged"

# Combine
final_table = pd.concat(
    [same_day_sig, lagged_sig],
    ignore_index=True
)

# Select and order columns
final_table = final_table[
    [
        "Analysis",
        "Participant",
        "Behavioral_Variable",
        "Wellness_Variable",
        "r",
        "p",
        "N",
        "p_FDR",
        "Significant_FDR"
    ]
]

# Sort
final_table = final_table.sort_values(
    ["Analysis", "Participant"]
)

# Save Excel
output_file = results_dir / "FINAL_RESULTS_TABLE.xlsx"
final_table.to_excel(output_file, index=False)

# Print summary
print("FINAL RESULTS TABLE")
print("-------------------")
print(f"Same-day FDR significant: {len(same_day_sig)}")
print(f"One-day lagged FDR significant: {len(lagged_sig)}")
print(f"Total: {len(final_table)}")
print(f"\nSaved to: {output_file}")
