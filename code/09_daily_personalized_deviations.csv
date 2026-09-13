import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/pmdata")
OUTPUT_DIR = Path("results/baseline")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVE_VARIABLES = [
    "Steps",
    "Exercise_Count",
    "Exercise_Duration",
    "Exercise_Distance",
    "Exercise_Calories",
    "Exercise_Avg_HR",
    "Sleep_Hours",
    "Sleep_Duration_Score",
    "Deep_Sleep_Minutes",
    "Sleep_Restlessness",
    "Sleep_Composition",
    "Sleep_Revitalization",
    "Sleep_Score"
]

BASELINE_FRACTION = 0.5

baselines = []
deviations = []

files = sorted(DATA_DIR.glob("p*_daily_merged.csv"))

print("=" * 60)
print("PERSONALIZED BASELINE")
print("=" * 60)

for file in files:

    participant = file.name.split("_")[0]

    df = pd.read_csv(file)

    if "Date" not in df.columns:
        continue

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    total_days = len(df)

    baseline_days = int(total_days * BASELINE_FRACTION)

    baseline = df.iloc[:baseline_days]
    analysis = df.iloc[baseline_days:]

    print(
        participant,
        "baseline =", len(baseline),
        "analysis =", len(analysis)
    )

    baseline_row = {
        "Participant": participant
    }

    baseline_stats = {}

    for variable in OBJECTIVE_VARIABLES:

        if variable not in baseline.columns:

            baseline_stats[variable] = {
                "mean": np.nan,
                "median": np.nan,
                "sd": np.nan
            }

            baseline_row[variable + "_Mean"] = np.nan
            baseline_row[variable + "_Median"] = np.nan
            baseline_row[variable + "_SD"] = np.nan

            continue

        values = pd.to_numeric(
            baseline[variable],
            errors="coerce"
        ).dropna()

        if len(values) == 0:

            mean_value = np.nan
            median_value = np.nan
            sd_value = np.nan

        else:

            mean_value = values.mean()
            median_value = values.median()
            sd_value = values.std()

        baseline_stats[variable] = {
            "mean": mean_value,
            "median": median_value,
            "sd": sd_value
        }

        baseline_row[variable + "_Mean"] = mean_value
        baseline_row[variable + "_Median"] = median_value
        baseline_row[variable + "_SD"] = sd_value

    baselines.append(baseline_row)

    for _, row in analysis.iterrows():

        deviation_row = {
            "Participant": participant,
            "Date": row["Date"]
        }

        for variable in OBJECTIVE_VARIABLES:

            if variable not in analysis.columns:

                deviation_row[variable + "_Deviation"] = np.nan
                deviation_row[variable + "_Z"] = np.nan
                deviation_row[variable + "_Abs_Z"] = np.nan

                continue

            value = pd.to_numeric(
                pd.Series([row[variable]]),
                errors="coerce"
            ).iloc[0]

            mean_value = baseline_stats[variable]["mean"]
            median_value = baseline_stats[variable]["median"]
            sd_value = baseline_stats[variable]["sd"]

            if pd.isna(value) or pd.isna(median_value):

                deviation = np.nan

            else:

                deviation = value - median_value

            if (
                pd.isna(value)
                or pd.isna(mean_value)
                or pd.isna(sd_value)
                or sd_value == 0
            ):

                z_score = np.nan

            else:

                z_score = (
                    (value - mean_value)
                    / sd_value
                )

            if pd.isna(z_score):

                absolute_z = np.nan

            else:

                absolute_z = abs(z_score)

            deviation_row[
                variable + "_Deviation"
            ] = deviation

            deviation_row[
                variable + "_Z"
            ] = z_score

            deviation_row[
                variable + "_Abs_Z"
            ] = absolute_z

        deviations.append(deviation_row)


baseline_df = pd.DataFrame(baselines)

deviation_df = pd.DataFrame(deviations)


baseline_df.to_csv(
    OUTPUT_DIR / "personalized_baselines.csv",
    index=False
)

deviation_df.to_csv(
    OUTPUT_DIR / "daily_personalized_deviations.csv",
    index=False
)


print()
print("=" * 60)
print("COMPLETE")
print("=" * 60)

print("Participants:", len(baseline_df))

print(
    "Daily deviation records:",
    len(deviation_df)
)

print()
print("Files created:")
print(
    OUTPUT_DIR / "personalized_baselines.csv"
)

print(
    OUTPUT_DIR / "daily_personalized_deviations.csv"
)
