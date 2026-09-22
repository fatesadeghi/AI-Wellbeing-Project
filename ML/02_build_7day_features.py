from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "pmdata"
RESULTS_DIR = PROJECT_ROOT / "ML" / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


BEHAVIOR_VARIABLES = [
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
    "Sleep_Score",
]


DATE_COLUMNS = [
    "date",
    "Date",
    "datetime",
    "Datetime",
    "timestamp",
    "Timestamp",
]


def find_date_column(df):
    for col in DATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def get_long_missing_streak(series):
    values = series.notna()

    max_streak = 0
    current_streak = 0

    for value in values:
        if not value:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def calculate_feature(window, column):
    values = window[column].dropna()

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return np.nan, np.nan

    first_value = values.iloc[0]
    last_value = values.iloc[-1]

    change = last_value - first_value

    dates = window.loc[values.index, "date"]
    day_numbers = (dates - dates.iloc[0]).dt.days.astype(float)

    if len(values) >= 2 and day_numbers.nunique() >= 2:
        slope = np.polyfit(day_numbers, values.astype(float), 1)[0]
    else:
        slope = np.nan

    return slope, change


all_results = []
availability_log = []

participant_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("BUILDING 7-DAY BEHAVIOR FEATURES")
print("=" * 80)
print(f"Participant files found: {len(participant_files)}")
print()


for file_path in participant_files:

    participant = file_path.stem.replace("_daily_merged", "")

    print("-" * 80)
    print(f"Processing: {file_path.stem}")

    try:
        df = pd.read_csv(file_path)

        date_column = find_date_column(df)

        if date_column is None:
            print("  ERROR: No date column found")
            availability_log.append({
                "participant": participant,
                "variable": "",
                "status": "Missing_Date_Column",
                "message": "No date column found"
            })
            continue

        df["date"] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=["date"]).copy()

        if df.empty:
            print("  ERROR: No valid dates")
            continue

        df = (
            df.sort_values("date")
            .drop_duplicates(subset=["date"])
            .reset_index(drop=True)
        )

        first_date = df["date"].min()
        first_7_days_end = first_date + pd.Timedelta(days=6)

        active_variables = []

        for variable in BEHAVIOR_VARIABLES:

            if variable not in df.columns:
                first_week = df[
                    (df["date"] >= first_date)
                    & (df["date"] <= first_7_days_end)
                ]

                has_first_week_data = False

                if not first_week.empty:
                    has_first_week_data = False

                if not has_first_week_data:
                    print(
                        f"  {variable}: removed "
                        f"(no data in first 7 days)"
                    )

                    availability_log.append({
                        "participant": participant,
                        "variable": variable,
                        "status": "Removed_First_7_Days",
                        "message": "No data available in first 7 days"
                    })

                    continue

            first_week = df[
                (df["date"] >= first_date)
                & (df["date"] <= first_7_days_end)
            ]

            if variable in df.columns:
                first_week_values = first_week[variable].notna().sum()
            else:
                first_week_values = 0

            if first_week_values == 0:
                print(
                    f"  {variable}: removed "
                    f"(no data in first 7 days)"
                )

                availability_log.append({
                    "participant": participant,
                    "variable": variable,
                    "status": "Removed_First_7_Days",
                    "message": "No data available in first 7 days"
                })

                continue

            active_variables.append(variable)

        if not active_variables:
            print("  ERROR: No behavioral variables available")
            continue

        print(f"  Active variables: {len(active_variables)}")

        target_dates = pd.date_range(
            start=first_date + pd.Timedelta(days=7),
            end=df["date"].max(),
            freq="D"
        )

        participant_feature_rows = 0

        for target_date in target_dates:

            history_start = target_date - pd.Timedelta(days=7)
            history_end = target_date - pd.Timedelta(days=1)

            window = df[
                (df["date"] >= history_start)
                & (df["date"] <= history_end)
            ].copy()

            if window.empty:
                continue

            row = {
                "participant": participant,
                "date": target_date
            }

            for variable in active_variables:

                slope, change = calculate_feature(
                    window,
                    variable
                )

                row[f"{variable}_7d_slope"] = slope
                row[f"{variable}_7d_change"] = change

                missing_streak = get_long_missing_streak(
                    window[variable]
                )

                if missing_streak > 7:
                    status = "Warning_Long_Missing"
                else:
                    status = "Available"

                availability_log.append({
                    "participant": participant,
                    "variable": variable,
                    "date": target_date,
                    "status": status,
                    "missing_streak_days": missing_streak
                })

            all_results.append(row)
            participant_feature_rows += 1

        print(f"  Feature rows created: {participant_feature_rows}")

    except Exception as e:
        print(f"  ERROR: {e}")

        availability_log.append({
            "participant": participant,
            "variable": "",
            "status": "Processing_Error",
            "message": str(e)
        })


features_df = pd.DataFrame(all_results)
availability_df = pd.DataFrame(availability_log)

features_output = RESULTS_DIR / "behavior_7day_features.csv"
availability_output = RESULTS_DIR / "behavior_availability_log.csv"

features_df.to_csv(
    features_output,
    index=False
)

availability_df.to_csv(
    availability_output,
    index=False
)


print()
print("=" * 80)
print("7-DAY FEATURE BUILD COMPLETE")
print("=" * 80)
print(f"Feature rows created: {len(features_df)}")
print(f"Feature columns: {len(features_df.columns)}")
print(f"Feature output: {features_output}")
print(f"Availability log: {availability_output}")
