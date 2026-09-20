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
# variable. With 13 behavioral variables this gives:
#
# 13 × 7 = 91 candidate ML feature columns.

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

def safe_zscore(series):
    """
    Calculate a standard z-score.
    Returns zeros when the standard deviation is zero.
    """

    mean = series.mean()
    std = series.std()

    if pd.isna(std) or std == 0:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        series - mean
    ) / std


def create_behavior_features(df):
    """
    Create the seven candidate representations for each
    behavioral variable.
    """

    features = pd.DataFrame(
        index=df.index
    )

    for variable in BEHAVIOR_VARS:

        if variable not in df.columns:
            continue

        series = pd.to_numeric(
            df[variable],
            errors="coerce"
        )

        # ----------------------------------------------------
        # 1. Raw value
        # ----------------------------------------------------

        features[
            f"{variable}__raw"
        ] = series

        # ----------------------------------------------------
        # 2. Day-to-day change
        # ----------------------------------------------------

        features[
            f"{variable}__change"
        ] = series.diff()

        # ----------------------------------------------------
        # 3. Previous 3-day rolling mean
        # ----------------------------------------------------

        features[
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
        # 4. Previous 7-day rolling mean
        # ----------------------------------------------------

        features[
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
        # 5. Z-score
        # ----------------------------------------------------

        z = safe_zscore(
            series
        )

        features[
            f"{variable}__zscore"
        ] = z

        # ----------------------------------------------------
        # 6. Absolute z-score
        # ----------------------------------------------------

        features[
            f"{variable}__abs_zscore"
        ] = z.abs()

        # ----------------------------------------------------
        # 7. Deviation from participant median
        # ----------------------------------------------------

        median = series.median()

        features[
            f"{variable}__deviation"
        ] = series - median

    return features


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
    df.dropna(subset=["Date"])
    .sort_values(
        ["Participant", "Date"]
    )
    .reset_index(drop=True)
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
# 6. CREATE CANDIDATE FEATURES
# ============================================================

all_feature_blocks = []

for participant, participant_df in df.groupby(
    "Participant",
    sort=False
):

    participant_df = (
        participant_df
        .sort_values("Date")
        .copy()
    )

    feature_block = create_behavior_features(
        participant_df
    )

    feature_block.insert(
        0,
        "Participant",
        participant
    )

    feature_block.insert(
        1,
        "Date",
        participant_df["Date"].values
    )

    all_feature_blocks.append(
        feature_block
    )


feature_df = pd.concat(
    all_feature_blocks,
    ignore_index=True
)


FEATURE_COLUMNS = [
    column
    for column in feature_df.columns
    if column not in [
        "Participant",
        "Date"
    ]
]

print(
    f"Candidate ML feature columns: "
    f"{len(FEATURE_COLUMNS)}"
)


# ============================================================
# 7. COMBINE FEATURES WITH TARGETS
# ============================================================

target_columns = [
    f"{TARGET_PREFIX}{variable}"
    for variable in WELLBEING_VARS
    if f"{TARGET_PREFIX}{variable}" in df.columns
]

target_df = df[
    [
        "Participant",
        "Date"
    ] + target_columns
].copy()

model_df = feature_df.merge(
    target_df,
    on=[
        "Participant",
        "Date"
    ],
    how="left"
)


# ============================================================
# 8. LOAD FEATURE AVAILABILITY INFORMATION
# ============================================================

excluded_features = set()

if os.path.exists(FEATURE_STATUS_FILE):

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
# 9. MODEL TRAINING
# ============================================================

performance_rows = []

importance_rows = []

final_feature_status_rows = []


participants = sorted(
    model_df["Participant"].unique()
)


for participant in participants:

    participant_df = (
        model_df[
            model_df["Participant"]
            ==
            participant
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print("\n" + "-" * 70)
    print(f"Participant: {participant}")

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

    participant_feature_columns = [
        column
        for column in FEATURE_COLUMNS
        if any(
            column.startswith(
                f"{variable}__"
            )
            for variable in active_behaviors
        )
    ]

    # --------------------------------------------------------
    # Train one model per wellbeing target
    # --------------------------------------------------------

    for wellbeing in WELLBEING_VARS:

        target = (
            f"{TARGET_PREFIX}{wellbeing}"
        )

        if target not in participant_df.columns:
            continue

        target_values = pd.to_numeric(
            participant_df[target],
            errors="coerce"
        )

        valid_target_mask = (
            target_values.notna()
        )

        if valid_target_mask.sum() < 10:

            print(
                f"  {wellbeing}: "
                f"insufficient target observations "
                f"({valid_target_mask.sum()})"
            )

            continue

        X = participant_df[
            participant_feature_columns
        ].copy()

        y = target_values.copy()

        # ----------------------------------------------------
        # Keep chronological order
        # ----------------------------------------------------

        X = X.loc[
            valid_target_mask
        ].reset_index(drop=True)

        y = y.loc[
            valid_target_mask
        ].reset_index(drop=True)

        dates = participant_df.loc[
            valid_target_mask,
            "Date"
        ].reset_index(drop=True)

        # ----------------------------------------------------
        # Time-series cross-validation
        # ----------------------------------------------------

        tscv = get_time_splits(
            len(X)
        )

        if tscv is None:

            print(
                f"  {wellbeing}: "
                "insufficient rows for TimeSeriesSplit"
            )

            continue

        fold_predictions = []

        fold_actuals = []

        fold_indices = []

        for train_index, test_index in tscv.split(X):

            X_train = X.iloc[
                train_index
            ]

            X_test = X.iloc[
                test_index
            ]

            y_train = y.iloc[
                train_index
            ]

            y_test = y.iloc[
                test_index
            ]

            model = make_model()

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            fold_predictions.extend(
                predictions
            )

            fold_actuals.extend(
                y_test.values
            )

            fold_indices.extend(
                test_index
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
        # ----------------------------------------------------

        final_model = make_model()

        final_model.fit(
            X,
            y
        )

        rf_model = final_model.named_steps[
            "model"
        ]

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
# 10. SAVE PERFORMANCE
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
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERSONALIZED ML COMPLETE")
print("=" * 70)

print(
    f"Participants: {len(participants)}"
)

print(
    f"Candidate feature columns: "
    f"{len(FEATURE_COLUMNS)}"
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
        f"Positive R2 models: {positive_r2}"
    )

    print(
        f"Zero R2 models: {zero_r2}"
    )

    print(
        f"Negative R2 models: {negative_r2}"
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

print("\nOutput files:")
print(
    f"  {PERFORMANCE_FILE}"
)
print(
    f"  {IMPORTANCE_FILE}"
)
print(
    f"  {FINAL_STATUS_FILE}"
)

print("\n" + "=" * 70)
