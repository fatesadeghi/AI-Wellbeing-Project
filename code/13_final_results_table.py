import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# FINAL RESULTS TABLES AND CHART
# ============================================================

# Paths
results_dir = Path("results/wellbeing_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

same_day_file = results_dir / "final_daily_deviation_wellbeing_results.csv"
lagged_file = results_dir / "final_lagged_deviation_wellbeing_results.csv"


# ============================================================
# 1. LOAD RESULTS
# ============================================================

same_day = pd.read_csv(same_day_file)
lagged = pd.read_csv(lagged_file)


# ============================================================
# 2. KEEP ONLY FDR-SIGNIFICANT RELATIONSHIPS
# ============================================================

same_day_sig = same_day[
    same_day["Significant_FDR"] == True
].copy()

lagged_sig = lagged[
    lagged["Significant_FDR"] == True
].copy()


# ============================================================
# 3. CREATE REPORT-READY TABLES
# ============================================================

same_day_table = same_day_sig[
    [
        "Participant",
        "Behavioral_Variable",
        "Wellness_Variable",
        "r",
        "p",
        "N",
        "p_FDR"
    ]
].copy()

lagged_table = lagged_sig[
    [
        "Participant",
        "Behavioral_Variable",
        "Wellness_Variable",
        "r",
        "p",
        "N",
        "p_FDR"
    ]
].copy()


# Rename columns for the report

same_day_table = same_day_table.rename(columns={
    "Participant": "Participant",
    "Behavioral_Variable": "Behavioral Variable",
    "Wellness_Variable": "Well-being Variable",
    "r": "Pearson r",
    "p": "p-value",
    "N": "N",
    "p_FDR": "FDR q-value"
})

lagged_table = lagged_table.rename(columns={
    "Participant": "Participant",
    "Behavioral_Variable": "Previous-day Behavior",
    "Wellness_Variable": "Next-day Well-being",
    "r": "Pearson r",
    "p": "p-value",
    "N": "N",
    "p_FDR": "FDR q-value"
})


# Sort tables
same_day_table = same_day_table.sort_values(
    ["Participant", "Behavioral Variable", "Well-being Variable"]
)

lagged_table = lagged_table.sort_values(
    ["Participant", "Previous-day Behavior", "Next-day Well-being"]
)


# Round statistical values
same_day_table["Pearson r"] = same_day_table["Pearson r"].round(3)
same_day_table["p-value"] = same_day_table["p-value"].round(6)
same_day_table["FDR q-value"] = same_day_table["FDR q-value"].round(6)

lagged_table["Pearson r"] = lagged_table["Pearson r"].round(3)
lagged_table["p-value"] = lagged_table["p-value"].round(6)
lagged_table["FDR q-value"] = lagged_table["FDR q-value"].round(6)


# ============================================================
# 4. SAVE CSV TABLES
# ============================================================

same_day_csv = results_dir / "TABLE_1_same_day_FDR_significant.csv"
lagged_csv = results_dir / "TABLE_2_one_day_lagged_FDR_significant.csv"

same_day_table.to_csv(same_day_csv, index=False)
lagged_table.to_csv(lagged_csv, index=False)


# ============================================================
# 5. CREATE COMBINED SUMMARY FOR THE CHART
# ============================================================

same_day_counts = (
    same_day_sig["Behavioral_Variable"]
    .value_counts()
    .rename("Same-day")
)

lagged_counts = (
    lagged_sig["Behavioral_Variable"]
    .value_counts()
    .rename("One-day lagged")
)

chart_data = pd.concat(
    [same_day_counts, lagged_counts],
    axis=1
).fillna(0)

chart_data = chart_data.sort_values(
    "Same-day",
    ascending=True
)


# ============================================================
# 6. CREATE CHART
# ============================================================

ax = chart_data.plot(
    kind="barh",
    figsize=(10, 7)
)

ax.set_title(
    "FDR-Significant Relationships by Behavioral Variable",
    fontsize=14
)

ax.set_xlabel("Number of FDR-significant relationships")
ax.set_ylabel("Behavioral Variable")

ax.legend(
    title="Analysis"
)

plt.tight_layout()

chart_file = results_dir / "FIGURE_FDR_significant_relationships.png"

plt.savefig(
    chart_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. TRY TO CREATE EXCEL FILE
# ============================================================

excel_file = results_dir / "FINAL_RESULTS_TABLE.xlsx"

try:

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl"
    ) as writer:

        same_day_table.to_excel(
            writer,
            sheet_name="Same-day",
            index=False
        )

        lagged_table.to_excel(
            writer,
            sheet_name="One-day lagged",
            index=False
        )

        chart_data.reset_index().rename(
            columns={"index": "Behavioral Variable"}
        ).to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

    excel_status = f"Excel saved: {excel_file}"

except Exception as e:

    excel_status = (
        "Excel file was not created because openpyxl is not available. "
        "CSV tables and PNG chart were created successfully."
    )


# ============================================================
# 8. PRINT FINAL SUMMARY
# ============================================================

print()
print("==============================================")
print("FINAL RESULTS TABLES AND CHART")
print("==============================================")

print(f"Same-day FDR-significant relationships: {len(same_day_table)}")
print(f"One-day lagged FDR-significant relationships: {len(lagged_table)}")
print(f"Total FDR-significant relationships: "
      f"{len(same_day_table) + len(lagged_table)}")

print()
print("Files created:")
print(f"- {same_day_csv}")
print(f"- {lagged_csv}")
print(f"- {chart_file}")
print(f"- {excel_status}")

print("==============================================")
