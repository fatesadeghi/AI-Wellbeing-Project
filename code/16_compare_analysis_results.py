import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "wellbeing_analysis"

SAME_DAY_FILE = RESULTS_DIR / "final_daily_deviation_wellbeing_results.csv"
LAGGED_FILE = RESULTS_DIR / "final_lagged_deviation_wellbeing_results.csv"
SEVEN_DAY_FILE = RESULTS_DIR / "seven_day_history_wellbeing_relationships.csv"

OUTPUT_COMPARISON = RESULTS_DIR / "comparison_same_day_lagged_7day.csv"
OUTPUT_REPEATED = RESULTS_DIR / "repeated_relationships_across_analyses.csv"


# --------------------------------------------------
# Load significant results
# --------------------------------------------------

def load_significant(file_path, analysis_name):
    df = pd.read_csv(file_path)

    # Keep only FDR-significant relationships
    if "Significant_FDR" in df.columns:
        df = df[df["Significant_FDR"] == True].copy()
    elif "p_FDR" in df.columns:
        df = df[df["p_FDR"] < 0.05].copy()
    else:
        raise ValueError(f"No FDR significance column found in {file_path}")

    df["Analysis"] = analysis_name

    return df


same_day = load_significant(
    SAME_DAY_FILE,
    "Same-day"
)

lagged = load_significant(
    LAGGED_FILE,
    "One-day lagged"
)

seven_day = load_significant(
    SEVEN_DAY_FILE,
    "Previous 7 days"
)


# --------------------------------------------------
# Standardize column names
# --------------------------------------------------

def standardize_columns(df):

    rename_map = {}

    if "Participant" not in df.columns:
        for col in ["participant", "Participant_ID", "ParticipantID"]:
            if col in df.columns:
                rename_map[col] = "Participant"
                break

    if "Behavioral_Variable" not in df.columns:
        for col in [
            "Objective_Variable",
            "Behavior_Variable",
            "BehavioralVariable"
        ]:
            if col in df.columns:
                rename_map[col] = "Behavioral_Variable"
                break

    if "Wellness_Variable" not in df.columns:
        for col in [
            "Wellbeing_Variable",
            "WellnessVariable",
            "WellbeingVariable"
        ]:
            if col in df.columns:
                rename_map[col] = "Wellness_Variable"
                break

    df = df.rename(columns=rename_map)

    required = [
        "Participant",
        "Behavioral_Variable",
        "Wellness_Variable"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


same_day = standardize_columns(same_day)
lagged = standardize_columns(lagged)
seven_day = standardize_columns(seven_day)


# --------------------------------------------------
# Create relationship IDs
# --------------------------------------------------

def add_relationship_id(df):

    df = df.copy()

    df["Relationship_ID"] = (
        df["Participant"].astype(str)
        + " | "
        + df["Behavioral_Variable"].astype(str)
        + " | "
        + df["Wellness_Variable"].astype(str)
    )

    return df


same_day = add_relationship_id(same_day)
lagged = add_relationship_id(lagged)
seven_day = add_relationship_id(seven_day)


# --------------------------------------------------
# Compare significant relationships
# --------------------------------------------------

same_ids = set(same_day["Relationship_ID"])
lagged_ids = set(lagged["Relationship_ID"])
seven_ids = set(seven_day["Relationship_ID"])

all_ids = sorted(
    same_ids | lagged_ids | seven_ids
)

comparison = pd.DataFrame({
    "Relationship_ID": all_ids
})


comparison["Same_Day"] = comparison["Relationship_ID"].isin(
    same_ids
)

comparison["One_Day_Lagged"] = comparison["Relationship_ID"].isin(
    lagged_ids
)

comparison["Previous_7_Days"] = comparison["Relationship_ID"].isin(
    seven_ids
)


comparison["Number_of_Analyses"] = (
    comparison[
        [
            "Same_Day",
            "One_Day_Lagged",
            "Previous_7_Days"
        ]
    ].sum(axis=1)
)


# --------------------------------------------------
# Safely extract relationship components
# --------------------------------------------------

parts = comparison["Relationship_ID"].str.split(
    " | ",
    n=2,
    expand=True
)

comparison["Participant"] = parts[0]
comparison["Behavioral_Variable"] = parts[1]
comparison["Wellness_Variable"] = parts[2]


comparison = comparison[
    [
        "Participant",
        "Behavioral_Variable",
        "Wellness_Variable",
        "Same_Day",
        "One_Day_Lagged",
        "Previous_7_Days",
        "Number_of_Analyses"
    ]
]


# --------------------------------------------------
# Save comparison table
# --------------------------------------------------

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


# --------------------------------------------------
# Repeated relationships
# --------------------------------------------------

repeated = comparison[
    comparison["Number_of_Analyses"] >= 2
].copy()

repeated = repeated.sort_values(
    "Number_of_Analyses",
    ascending=False
)

repeated.to_csv(
    OUTPUT_REPEATED,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nComparison of FDR-significant relationships")
print("-------------------------------------------")

print(
    f"Same-day: {len(same_ids)}"
)

print(
    f"One-day lagged: {len(lagged_ids)}"
)

print(
    f"Previous 7 days: {len(seven_ids)}"
)

print(
    f"Same-day + One-day lagged overlap: "
    f"{len(same_ids & lagged_ids)}"
)

print(
    f"Same-day + Previous 7 days overlap: "
    f"{len(same_ids & seven_ids)}"
)

print(
    f"One-day lagged + Previous 7 days overlap: "
    f"{len(lagged_ids & seven_ids)}"
)

print(
    f"All three analyses: "
    f"{len(same_ids & lagged_ids & seven_ids)}"
)

print(
    f"Significant in at least two analyses: "
    f"{len(repeated)}"
)

print("\nOutput files:")
print(OUTPUT_COMPARISON)
print(OUTPUT_REPEATED)
