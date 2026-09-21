import os
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

# Current file:
# AI-Wellbeing-Project/ML/Code/10_personalized_ml.py

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

ML_DIR = os.path.join(
    RESULTS_DIR,
    "ml"
)

INPUT_FILE = os.path.join(
    ML_DIR,
    "ml_prepared_data.csv"
)

FEATURE_STATUS_FILE = os.path.join(
    ML_DIR,
    "ml_feature_status.csv"
)

OUTPUT_DIR = ML_DIR

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
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

TARGET_PREFIX = "NextDayChange_"

RANDOM_STATE = 42

N_ESTIMATORS = 100

MIN_SAMPLES_LEAF = 3

MAX_SPLITS = 3


# ============================================================
# 3. FEATURE REPRESENTATIONS
# ============================================================

# Seven representations are created for each behavioral
# variable.
#
# 13 behavioral variables × 7 representations = 91
# candidate ML feature columns.

FEATURE_REPRESENTATIONS = [
    "raw",
    "change",
    "rolling_3",
    "rolling_7",
    "zscore",
    "abs_zscore",
    "deviation"
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def make_model():
    """
    Personalized Random Forest pipeline.

    Median imputation is performed inside the model pipeline
    so that imputation is learned only from training data.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )


def get_time_splits(n_rows):
    """
    Return an appropriate TimeSeriesSplit object.
    """

    if n_rows < 4:
        return None

    n_splits = min(
        MAX_SPLITS,
        n_rows - 1
    )

    if n_splits < 2:
        return None

    return TimeSeriesSplit(
        n_splits=n_splits
    )


def create_calendar_features(
    df,
    feature_reference_end
):
    """
    Create seven candidate representations for each
    behavioral variable using calendar-day alignment.

    Missing calendar dates are represented explicitly.

    - raw: value on the current calendar day
    - change: current day minus previous calendar day
    - rolling_3: mean of the previous 3 calendar days
    - rolling_7: mean of the previous 7 calendar days
    - zscore: historical z-score
    - abs_zscore: absolute historical z-score
    - deviation: historical median deviation

    Historical normalization statistics are calculated only
    from observations available up to feature_reference_end.
    """

    working = df.copy()

    working["Date"] = pd.to_datetime(
        working["Date"],
        errors="coerce"
    )

    working = (
        working
        .sort_values("Date")
        .drop_duplicates(
            subset="Date",
            keep="first"
        )
    )

    if working.empty:
        return pd.DataFrame(
            index=df.index
        )

    original_index = working.index

    calendar_index = pd.date_range(
        start=working["Date"].min(),
        end=working["Date"].max(),
        freq="D"
    )

    calendar_df = (
        working
        .set_index("Date")
        .reindex(calendar_index)
    )

    calendar_df.index.name = "Date"

    # --------------------------------------------------------
    # Reference data
    # --------------------------------------------------------

    reference_dates = pd.to_datetime(
        working.loc[
            working.index <= feature_reference_end,
            "Date"
        ],
        errors="coerce"
    )

    reference_dates = reference_dates.dropna()

    if len(reference_dates) > 0:

        reference_end_date = (
            reference_dates.max()
        )

        reference_df = calendar_df.loc[
            calendar_df.index <= reference_end_date
        ]

    else:

        reference_df = calendar_df.iloc[0:0]

    features_calendar = pd.DataFrame(
        index=calendar_df.index
    )

    for variable in BEHAVIOR_VARS:

        if variable not in calendar_df.columns:
            continue

        series = pd.to_numeric(
            calendar_df[variable],
            errors="coerce"
        )

        reference_series = pd.to_numeric(
            reference_df[variable],
            errors="coerce"
        )

        # ----------------------------------------------------
        # 1. Raw value
        # ----------------------------------------------------

        features_calendar[
            f"{variable}__raw"
        ] = series

        # ----------------------------------------------------
        # 2. Calendar-day change
        # ----------------------------------------------------

        features_calendar[
            f"{variable}__change"
        ] = (
            series
            - series.shift(1)
        )

        # ----------------------------------------------------
        # 3. Previous 3 calendar-day rolling mean
        # ----------------------------------------------------

        features_calendar[
            f"{variable}__rolling_3"
        ] = (
            series
            .shift(1)
            .rolling(
                window=3,
                min_periods=3
            )
            .mean()
        )

        # ----------------------------------------------------
        # 4. Previous 7 calendar-day rolling mean
        # ----------------------------------------------------

        features_calendar[
            f"{variable}__rolling_7"
        ] = (
            series
            .shift(1)
            .rolling(
                window=7,
                min_periods=7
            )
            .mean()
        )

        # ----------------------------------------------------
        # 5. Historical z-score
        # ----------------------------------------------------

        reference_mean = (
            reference_series.mean()
        )

        reference_std = (
            reference_series.std()
        )

        if (
            pd.isna(reference_std)
            or reference_std == 0
        ):

            z = pd.Series(
                0.0,
                index=series.index
            )

        else:

            z = (
                series - reference_mean
            ) / reference_std

        features_calendar[
            f"{variable}__zscore"
        ] = z

        # ----------------------------------------------------
        # 6. Absolute historical z-score
        # ----------------------------------------------------

        features_calendar[
            f"{variable}__abs_zscore"
        ] = z.abs()

        # ----------------------------------------------------
        # 7. Historical median deviation
        # ----------------------------------------------------

        reference_median = (
            reference_series.median()
        )

        features_calendar[
            f"{variable}__deviation"
        ] = (
            series - reference_median
        )

    # --------------------------------------------------------
    # Map calendar-day features back to original rows
    # --------------------------------------------------------

    date_map = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    result = features_calendar.reindex(
        date_map
    )

    result.index = df.index

    return result


# ============================================================
# 5. LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Prepared ML data not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = (
    df
    .dropna(
        subset=["Date"]
    )
    .sort_values(
        ["Participant", "Date"]
    )
    .reset_index(
        drop=True
    )
)


print("=" * 70)
print("PERSONALIZED MACHINE LEARNING")
print("=" * 70)

print(
    f"Rows loaded: {len(df)}"
)

print(
    f"Participants: "
    f"{df['Participant'].nunique()}"
)


# ============================================================
# 6. LOAD FEATURE AVAILABILITY INFORMATION
# ============================================================

excluded_features = set()

if os.path.exists(
    FEATURE_STATUS_FILE
):

    feature_status = pd.read_csv(
        FEATURE_STATUS_FILE
    )

    excluded_rows = feature_status[
        feature_status["Status"]
        ==
        "Excluded_First_10_Days_Missing"
    ]

    for _, row in excluded_rows.iterrows():

        excluded_features.add(
            (
                row["Participant"],
                row["Variable"]
            )
        )


# ============================================================
# 7. MODEL TRAINING
# ============================================================

performance_rows = []

importance_rows = []

final_feature_status_rows = []

participants = sorted(
    df["Participant"].unique()
)


for participant in participants:

    participant_df = (
        df[
            df["Participant"]
            ==
            participant
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        "\n" + "-" * 70
    )

    print(
        f"Participant: {participant}"
    )

    # --------------------------------------------------------
    # Participant-specific active behavior variables
    # --------------------------------------------------------

    active_behaviors = []

    for variable in BEHAVIOR_VARS:

        if (
            participant,
            variable
        ) in excluded_features:

            final_feature_status_rows.append({
                "Participant": participant,
                "Variable": variable,
                "Status": "Excluded"
            })

            continue

        active_behaviors.append(
            variable
        )

        final_feature_status_rows.append({
            "Participant": participant,
            "Variable": variable,
            "Status": "Active"
        })

    print(
        f"Active behavioral variables: "
        f"{len(active_behaviors)}/{len(BEHAVIOR_VARS)}"
    )

    # --------------------------------------------------------
    # Participant-specific feature columns
    # --------------------------------------------------------

    participant_feature_columns = []

    for variable in active_behaviors:

        for representation in FEATURE_REPRESENTATIONS:

            participant_feature_columns.append(
                f"{variable}__{representation}"
            )

    # --------------------------------------------------------
    # Train one model per wellbeing target
    # --------------------------------------------------------

    for wellbeing in WELLBEING_VARS:

        target = (
            f"{TARGET_PREFIX}{wellbeing}"
        )

        if target not in participant_df.columns:

            print(
                f"  {wellbeing}: target column not found"
            )

            continue

        target_values = pd.to_numeric(
            participant_df[target],
            errors="coerce"
        )

        valid_target_mask = (
            target_values.notna()
        )

        n_valid = int(
            valid_target_mask.sum()
        )

        if n_valid < 10:

            print(
                f"  {wellbeing}: "
                f"insufficient target observations "
                f"({n_valid})"
            )

            continue

        # ----------------------------------------------------
        # Determine valid target rows
        # ----------------------------------------------------

        valid_indices = np.where(
            valid_target_mask.values
        )[0]

        valid_df = participant_df.loc[
            valid_target_mask
        ].copy()

        valid_df = (
            valid_df
            .reset_index(drop=True)
        )

        y = pd.to_numeric(
            valid_df[target],
            errors="coerce"
        ).reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Time-series cross-validation
        # ----------------------------------------------------

        tscv = get_time_splits(
            len(y)
        )

        if tscv is None:

            print(
                f"  {wellbeing}: "
                "insufficient rows for TimeSeriesSplit"
            )

            continue

        fold_predictions = []

        fold_actuals = []

        fold_test_dates = []

        for train_index, test_index in tscv.split(y):

            # ------------------------------------------------
            # Actual participant rows belonging to this fold
            # ------------------------------------------------

            train_original_indices = (
                valid_indices[train_index]
            )

            test_original_indices = (
                valid_indices[test_index]
            )

            # ------------------------------------------------
            # Build calendar-aware features using ONLY the
            # training boundary.
            # ------------------------------------------------

            training_end_index = (
                train_original_indices.max()
            )

            fold_features = (
                create_calendar_features(
                    participant_df,
                    feature_reference_end=training_end_index
                )
            )

            fold_features = fold_features[
                [
                    column
                    for column in participant_feature_columns
                    if column in fold_features.columns
                ]
            ]

            X_train_fold = (
                fold_features.iloc[
                    train_original_indices
                ]
                .copy()
            )

            X_test_fold = (
                fold_features.iloc[
                    test_original_indices
                ]
                .copy()
            )

            y_train = y.iloc[
                train_index
            ].copy()

            y_test = y.iloc[
                test_index
            ].copy()

            # ------------------------------------------------
            # Train model
            # ------------------------------------------------

            model = make_model()

            model.fit(
                X_train_fold,
                y_train
            )

            predictions = model.predict(
                X_test_fold
            )

            fold_predictions.extend(
                predictions
            )

            fold_actuals.extend(
                y_test.values
            )

            fold_test_dates.extend(
                valid_df[
                    "Date"
                ].iloc[
                    test_index
                ].values
            )

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        if len(fold_actuals) == 0:
            continue

        actual = np.asarray(
            fold_actuals,
            dtype=float
        )

        predicted = np.asarray(
            fold_predictions,
            dtype=float
        )

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

        try:

            r2 = r2_score(
                actual,
                predicted
            )

        except Exception:

            r2 = np.nan

        performance_rows.append({
            "Participant": participant,
            "Wellbeing": wellbeing,
            "Target": target,
            "N_Observations": len(y),
            "N_Test_Predictions": len(actual),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "N_Features": len(
                participant_feature_columns
            )
        })

        print(
            f"  {wellbeing}: "
            f"R2={r2:.4f}, "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}"
        )

        # ----------------------------------------------------
        # Fit final model on all available observations
        #
        # Final calendar-aware features use the participant's
        # complete available history. No future observation
        # beyond the final recorded date exists here.
        # ----------------------------------------------------

        final_features = (
            create_calendar_features(
                participant_df,
                feature_reference_end=(
                    len(participant_df) - 1
                )
            )
        )

        final_features = (
            final_features
            .loc[
                :,
                participant_feature_columns
            ]
        )

        final_X = final_features.loc[
            valid_target_mask.values
        ].reset_index(
            drop=True
        )

        final_y = y.reset_index(
            drop=True
        )

        final_model = make_model()

        final_model.fit(
            final_X,
            final_y
        )

        rf_model = (
            final_model.named_steps[
                "model"
            ]
        )

        importances = (
            rf_model.feature_importances_
        )

        # ----------------------------------------------------
        # Save feature importance
        # ----------------------------------------------------

        for feature, importance in zip(
            participant_feature_columns,
            importances
        ):

            importance_rows.append({
                "Participant": participant,
                "Wellbeing": wellbeing,
                "Feature": feature,
                "Importance": importance
            })


# ============================================================
# 8. SAVE RESULTS
# ============================================================

performance_df = pd.DataFrame(
    performance_rows
)

importance_df = pd.DataFrame(
    importance_rows
)

feature_status_df = pd.DataFrame(
    final_feature_status_rows
)


PERFORMANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_model_performance.csv"
)

IMPORTANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_feature_importance.csv"
)

FINAL_STATUS_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_final_feature_status.csv"
)


performance_df.to_csv(
    PERFORMANCE_FILE,
    index=False
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)

feature_status_df.to_csv(
    FINAL_STATUS_FILE,
    index=False
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PERSONALIZED ML COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Participants: {len(participants)}"
)

print(
    "Candidate feature columns: "
    f"{len(BEHAVIOR_VARS) * len(FEATURE_REPRESENTATIONS)}"
)

print(
    f"Models trained: "
    f"{len(performance_df)}"
)

print(
    f"Feature importance rows: "
    f"{len(importance_df)}"
)

if not performance_df.empty:

    positive_r2 = (
        performance_df["R2"] > 0
    ).sum()

    zero_r2 = (
        performance_df["R2"] == 0
    ).sum()

    negative_r2 = (
        performance_df["R2"] < 0
    ).sum()

    print(
        f"Positive R2 models: "
        f"{positive_r2}"
    )

    print(
        f"Zero R2 models: "
        f"{zero_r2}"
    )

    print(
        f"Negative R2 models: "
        f"{negative_r2}"
    )

    print(
        f"Mean R2: "
        f"{performance_df['R2'].mean():.4f}"
    )

    print(
        f"Median R2: "
        f"{performance_df['R2'].median():.4f}"
    )

    print(
        f"Mean MAE: "
        f"{performance_df['MAE'].mean():.4f}"
    )

    print(
        f"Mean RMSE: "
        f"{performance_df['RMSE'].mean():.4f}"
    )


print(
    "\nOutput files:"
)

print(
    f"  {PERFORMANCE_FILE}"
)

print(
    f"  {IMPORTANCE_FILE}"
)

print(
    f"  {FINAL_STATUS_FILE}"
)

print(
    "\n" + "=" * 70
)
