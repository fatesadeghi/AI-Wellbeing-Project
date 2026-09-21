import os
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "pmdata"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "final_ml"
)

COMPLETE_DIR = os.path.join(
    OUTPUT_DIR,
    "complete_data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPLETE_DIR, exist_ok=True)


# ============================================================
# 2. VARIABLES
# ============================================================

BEHAVIOR_VARS = [
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

WELLBEING_VARS = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress"
]

RANDOM_STATE = 42
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 3
MAX_SPLITS = 3


# ============================================================
# 3. FIND PARTICIPANT FILES
# ============================================================

participant_files = sorted(
    [
        f
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
        and f.startswith("p")
    ]
)

if not participant_files:
    raise FileNotFoundError(
        f"No participant files found in:\n{DATA_DIR}"
    )


print("=" * 80)
print("FINAL WELLBEING MACHINE LEARNING")
print("=" * 80)

print(
    f"Participant files found: {len(participant_files)}"
)


# ============================================================
# 4. FEATURE CREATION
# ============================================================

def create_features(df):
    """
    Create prediction features for each calendar date.

    Features use ONLY information available before
    the prediction date:

    - Previous day values (t-1)
    - Previous 7 calendar-day values / averages
    - 7-day change

    Target:
        Change in wellbeing on the next day (t+1)
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Create continuous calendar
    # --------------------------------------------------------

    calendar = pd.DataFrame(
        index=pd.date_range(
            start=df["Date"].min(),
            end=df["Date"].max(),
            freq="D"
        )
    )

    calendar.index.name = "Date"

    original = df.set_index("Date")

    for column in BEHAVIOR_VARS + WELLBEING_VARS:

        if column in original.columns:

            calendar[column] = pd.to_numeric(
                original[column],
                errors="coerce"
            )

        else:

            calendar[column] = np.nan

    # --------------------------------------------------------
    # Behavioral features
    # --------------------------------------------------------

    feature_columns = []

    for variable in BEHAVIOR_VARS:

        series = calendar[variable]

        # ----------------------------------------------------
        # Previous day
        # ----------------------------------------------------

        name = f"{variable}__lag1"

        calendar[name] = series.shift(1)

        feature_columns.append(name)

        # ----------------------------------------------------
        # 7-day mean
        # ----------------------------------------------------

        name = f"{variable}__mean7"

        calendar[name] = (
            series
            .shift(1)
            .rolling(
                window=7,
                min_periods=7
            )
            .mean()
        )

        feature_columns.append(name)

        # ----------------------------------------------------
        # 7-day standard deviation
        # ----------------------------------------------------

        name = f"{variable}__std7"

        calendar[name] = (
            series
            .shift(1)
            .rolling(
                window=7,
                min_periods=7
            )
            .std()
        )

        feature_columns.append(name)

        # ----------------------------------------------------
        # Difference between yesterday and 7-day mean
        # ----------------------------------------------------

        name = f"{variable}__deviation7"

        calendar[name] = (
            calendar[f"{variable}__lag1"]
            - calendar[f"{variable}__mean7"]
        )

        feature_columns.append(name)

        # ----------------------------------------------------
        # Change from 7 days ago to yesterday
        # ----------------------------------------------------

        name = f"{variable}__change7"

        calendar[name] = (
            series.shift(1)
            - series.shift(8)
        )

        feature_columns.append(name)

    # --------------------------------------------------------
    # Wellbeing target
    # --------------------------------------------------------

    target_columns = []

    for wellbeing in WELLBEING_VARS:

        target = f"NextDayChange_{wellbeing}"

        calendar[target] = (
            calendar[wellbeing].shift(-1)
            - calendar[wellbeing]
        )

        target_columns.append(target)

    # --------------------------------------------------------
    # Return only prediction rows
    # --------------------------------------------------------

    result = calendar.reset_index()

    result["Participant"] = df["Participant"].iloc[0]

    return result[
        [
            "Participant",
            "Date"
        ]
        + feature_columns
        + target_columns
    ]


# ============================================================
# 5. MODEL
# ============================================================

def train_model(
    data,
    feature_columns,
    target
):
    """
    Train and evaluate a Random Forest using
    time-series cross-validation.
    """

    data = data.sort_values("Date").reset_index(drop=True)

    X = data[feature_columns].copy()

    y = pd.to_numeric(
        data[target],
        errors="coerce"
    )

    # Target must exist
    valid_target = y.notna()

    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    if len(y) < 10:
        return None

    # --------------------------------------------------------
    # Complete rows for this specific prediction
    # --------------------------------------------------------

    complete_mask = X.notna().all(axis=1) & y.notna()

    X = X.loc[complete_mask].reset_index(drop=True)
    y = y.loc[complete_mask].reset_index(drop=True)

    if len(y) < 10:
        return None

    n_splits = min(
        MAX_SPLITS,
        len(y) - 1
    )

    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(
        n_splits=n_splits
    )

    actual = []
    predicted = []

    for train_idx, test_idx in tscv.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        actual.extend(
            y_test.tolist()
        )

        predicted.extend(
            prediction.tolist()
        )

    if len(actual) == 0:
        return None

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    # --------------------------------------------------------
    # Final model for feature importance
    # --------------------------------------------------------

    final_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    final_model.fit(
        X,
        y
    )

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": final_model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    return {
        "n_observations": len(y),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "feature_importance": importance
    }


# ============================================================
# 6. LOAD PARTICIPANTS
# ============================================================

all_prediction_data = []
complete_participants = []

participant_summary = []


for filename in participant_files:

    participant = os.path.splitext(
        filename
    )[0]

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    print("\n" + "-" * 80)
    print(f"Processing {participant}")

    df = pd.read_csv(
        filepath
    )

    if "Date" not in df.columns:
        print("  Date column missing.")
        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    df = df.sort_values(
        "Date"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = (
        BEHAVIOR_VARS
        + WELLBEING_VARS
    )

    missing_columns = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        print(
            "  Missing columns:",
            missing_columns
        )

        participant_summary.append({
            "Participant": participant,
            "Complete_Participant": False,
            "Reason": "Missing required columns"
        })

        continue

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["Participant"] = participant

    # --------------------------------------------------------
    # Participant-level completeness
    # --------------------------------------------------------

    complete_participant = (
        df[
            required_columns
        ]
        .notna()
        .all()
        .all()
    )

    if complete_participant:

        complete_participants.append(
            participant
        )

        df_complete = df.copy()

        df_complete.to_csv(
            os.path.join(
                COMPLETE_DIR,
                f"{participant}_complete.csv"
            ),
            index=False
        )

        print(
            "  COMPLETE participant"
        )

    else:

        print(
            "  Missing data detected"
        )

    # --------------------------------------------------------
    # Create ML data
    # --------------------------------------------------------

    prediction_data = create_features(
        df
    )

    if not prediction_data.empty:

        all_prediction_data.append(
            prediction_data
        )

    participant_summary.append({
        "Participant": participant,
        "Complete_Participant": complete_participant,
        "Reason": (
            "Complete"
            if complete_participant
            else "Missing values"
        )
    })


# ============================================================
# 7. SAVE COMPLETE PARTICIPANT INFORMATION
# ============================================================

summary_df = pd.DataFrame(
    participant_summary
)

summary_df.to_csv(
    os.path.join(
        COMPLETE_DIR,
        "complete_participant_summary.csv"
    ),
    index=False
)


# ============================================================
# 8. COMBINE ML DATA
# ============================================================

if not all_prediction_data:

    raise RuntimeError(
        "No prediction data could be created."
    )

ml_data = pd.concat(
    all_prediction_data,
    ignore_index=True
)

ml_data = ml_data.sort_values(
    ["Participant", "Date"]
).reset_index(drop=True)


ml_data.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_prediction_data.csv"
    ),
    index=False
)


# ============================================================
# 9. FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in ml_data.columns
    if "__lag1" in column
    or "__mean7" in column
    or "__std7" in column
    or "__deviation7" in column
    or "__change7" in column
]


# ============================================================
# 10. ALL AVAILABLE DATA ML
# ============================================================

print("\n")
print("=" * 80)
print("ML RESULTS — ALL AVAILABLE DATA")
print("=" * 80)

all_results = []
all_importance = []


for participant in sorted(
    ml_data["Participant"].unique()
):

    participant_data = ml_data[
        ml_data["Participant"] == participant
    ].copy()

    for wellbeing in WELLBEING_VARS:

        target = (
            f"NextDayChange_{wellbeing}"
        )

        result = train_model(
            participant_data,
            feature_columns,
            target
        )

        if result is None:
            continue

        all_results.append({
            "Participant": participant,
            "Wellbeing": wellbeing,
            "N_Observations": result["n_observations"],
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        importance = result[
            "feature_importance"
        ].copy()

        importance["Participant"] = participant
        importance["Wellbeing"] = wellbeing

        all_importance.append(
            importance
        )


all_results_df = pd.DataFrame(
    all_results
)

all_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_results_all_available.csv"
    ),
    index=False
)


if all_importance:

    all_importance_df = pd.concat(
        all_importance,
        ignore_index=True
    )

    all_importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance_all_available.csv"
        ),
        index=False
    )


# ============================================================
# 11. COMPLETE-DATA ML
# ============================================================

print("\n")
print("=" * 80)
print("ML RESULTS — COMPLETE DATA")
print("=" * 80)

complete_data = ml_data[
    ml_data["Participant"].isin(
        complete_participants
    )
].copy()


complete_results = []
complete_importance = []


for participant in complete_participants:

    participant_data = complete_data[
        complete_data["Participant"] == participant
    ].copy()

    for wellbeing in WELLBEING_VARS:

        target = (
            f"NextDayChange_{wellbeing}"
        )

        result = train_model(
            participant_data,
            feature_columns,
            target
        )

        if result is None:
            continue

        complete_results.append({
            "Participant": participant,
            "Wellbeing": wellbeing,
            "N_Observations": result["n_observations"],
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        importance = result[
            "feature_importance"
        ].copy()

        importance["Participant"] = participant
        importance["Wellbeing"] = wellbeing

        complete_importance.append(
            importance
        )


complete_results_df = pd.DataFrame(
    complete_results
)

complete_results_df.to_csv(
    os.path.join(
        COMPLETE_DIR,
        "ml_results_complete.csv"
    ),
    index=False
)


if complete_importance:

    complete_importance_df = pd.concat(
        complete_importance,
        ignore_index=True
    )

    complete_importance_df.to_csv(
        os.path.join(
            COMPLETE_DIR,
            "feature_importance_complete.csv"
        ),
        index=False
    )


# ============================================================
# 12. SUMMARY COMPARISON
# ============================================================

def summarize_results(
    results,
    dataset_name
):

    if results.empty:

        return {
            "Dataset": dataset_name,
            "Participant_Target_Models": 0,
            "Mean_MAE": np.nan,
            "Mean_RMSE": np.nan,
            "Mean_R2": np.nan
        }

    return {
        "Dataset": dataset_name,
        "Participant_Target_Models": len(results),
        "Mean_MAE": results["MAE"].mean(),
        "Mean_RMSE": results["RMSE"].mean(),
        "Mean_R2": results["R2"].mean()
    }


comparison = pd.DataFrame([
    summarize_results(
        all_results_df,
        "All Available"
    ),
    summarize_results(
        complete_results_df,
        "Complete Data"
    )
])


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ml_comparison.csv"
    ),
    index=False
)


# ============================================================
# 13. FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(
    f"Total participants: "
    f"{len(participant_files)}"
)

print(
    f"Complete participants: "
    f"{len(complete_participants)}"
)

print(
    f"All-data models: "
    f"{len(all_results_df)}"
)

print(
    f"Complete-data models: "
    f"{len(complete_results_df)}"
)

print("\nComparison:")

print(
    comparison.to_string(
        index=False
    )
)

print("\n")
print("Results saved to:")
print(OUTPUT_DIR)

print("\nComplete participant data saved to:")
print(COMPLETE_DIR)

print("=" * 80)
print("DONE")
print("=" * 80)
