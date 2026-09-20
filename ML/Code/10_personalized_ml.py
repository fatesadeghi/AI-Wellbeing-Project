import os
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline


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

ML_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ml"
)

DATA_PATH = os.path.join(
    ML_DIR,
    "ml_ready_dataset.csv"
)

AVAILABILITY_PATH = os.path.join(
    ML_DIR,
    "ml_feature_availability.csv"
)

OUTPUT_DIR = ML_DIR

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. VARIABLES
# ============================================================

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

ACTIVITY_VARIABLES = [
    "Steps",
    "Exercise_Count",
    "Exercise_Duration",
    "Exercise_Distance",
    "Exercise_Calories",
    "Exercise_Avg_HR",
]

SLEEP_VARIABLES = [
    "Sleep_Hours",
    "Sleep_Duration_Score",
    "Deep_Sleep_Minutes",
    "Sleep_Restlessness",
    "Sleep_Composition",
    "Sleep_Revitalization",
    "Sleep_Score",
]

WELLBEING_TARGETS = [
    "fatigue_next_day_change",
    "mood_next_day_change",
    "readiness_next_day_change",
    "sleep_quality_next_day_change",
    "stress_next_day_change",
]

FEATURE_SUFFIXES = [
    "",
    "_deviation",
    "_z",
    "_abs_z",
    "_7d_mean",
    "_7d_std",
    "_1d_change",
]

MIN_ROWS = 30
N_SPLITS = 3
RANDOM_STATE = 42


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("PERSONALIZED MACHINE LEARNING")
print("=" * 70)

print("\nLoading ML dataset...")

df = pd.read_csv(
    DATA_PATH
)

availability = pd.read_csv(
    AVAILABILITY_PATH
)

print(f"Rows loaded: {len(df)}")

if "Participant" not in df.columns:
    raise ValueError(
        "Participant column is missing from ml_ready_dataset.csv"
    )

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

df = df.sort_values(
    ["Participant", "Date"]
).reset_index(
    drop=True
)

participants = sorted(
    df["Participant"].dropna().unique()
)

print(f"Participants: {len(participants)}")


# ============================================================
# 4. BUILD PARTICIPANT-SPECIFIC FEATURE STATUS
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANT-SPECIFIC FEATURE SELECTION")
print("=" * 70)

status_rows = []

for participant in participants:

    participant_availability = availability[
        availability["Participant"] == participant
    ].copy()

    active_variables = []
    excluded_variables = []

    for variable in BEHAVIOR_VARIABLES:

        row = participant_availability[
            participant_availability["Variable"] == variable
        ]

        exclude = False

        if (
            variable in ACTIVITY_VARIABLES
            and not row.empty
        ):
            first_10_missing = row.iloc[0].get(
                "First_10_Days_All_Missing",
                False
            )

            if pd.isna(first_10_missing):
                first_10_missing = False

            exclude = bool(first_10_missing)

        if exclude:
            excluded_variables.append(
                variable
            )
        else:
            active_variables.append(
                variable
            )

        status_rows.append(
            {
                "Participant": participant,
                "Variable": variable,
                "Variable_Type": (
                    "Activity"
                    if variable in ACTIVITY_VARIABLES
                    else "Sleep"
                ),
                "Active": not exclude,
                "Excluded_First_10_Days": exclude,
            }
        )

    print(
        f"{participant}: "
        f"{len(active_variables)}/{len(BEHAVIOR_VARIABLES)} "
        f"behavioral variables active"
    )

    if excluded_variables:
        print(
            "    Excluded: "
            + ", ".join(excluded_variables)
        )


feature_status = pd.DataFrame(
    status_rows
)

feature_status_path = os.path.join(
    OUTPUT_DIR,
    "ml_final_feature_status.csv"
)

feature_status.to_csv(
    feature_status_path,
    index=False
)


# ============================================================
# 5. BUILD CANDIDATE ML FEATURES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MODEL FEATURES")
print("=" * 70)

candidate_feature_columns = []

for variable in BEHAVIOR_VARIABLES:

    for suffix in FEATURE_SUFFIXES:

        column = (
            variable
            if suffix == ""
            else variable + suffix
        )

        if column in df.columns:
            candidate_feature_columns.append(
                column
            )

print(
    f"Candidate ML feature columns: "
    f"{len(candidate_feature_columns)}"
)

if len(candidate_feature_columns) != 91:
    warnings.warn(
        "Expected 91 candidate features "
        f"but found {len(candidate_feature_columns)}."
    )


# ============================================================
# 6. RESULTS CONTAINERS
# ============================================================

performance_rows = []
importance_rows = []


# ============================================================
# 7. PERSONALIZED MODELS
# ============================================================

for participant in participants:

    participant_df = df[
        df["Participant"] == participant
    ].copy()

    participant_df = participant_df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    active_variables = feature_status[
        (
            feature_status["Participant"]
            == participant
        )
        & (
            feature_status["Active"]
            == True
        )
    ]["Variable"].tolist()

    participant_feature_columns = []

    for variable in active_variables:

        for suffix in FEATURE_SUFFIXES:

            column = (
                variable
                if suffix == ""
                else variable + suffix
            )

            if column in participant_df.columns:
                participant_feature_columns.append(
                    column
                )

    print("\n" + "-" * 70)
    print(
        f"PARTICIPANT: {participant}"
    )
    print(
        f"Active behavioral variables: "
        f"{len(active_variables)}"
    )
    print(
        f"Available model features: "
        f"{len(participant_feature_columns)}"
    )

    if len(participant_feature_columns) == 0:
        print(
            "No usable behavioral features. Skipping."
        )
        continue


    # ========================================================
    # 8. MODEL FOR EACH WELLBEING INDICATOR
    # ========================================================

    for target in WELLBEING_TARGETS:

        print(
            f"\n  Target: {target}"
        )

        if target not in participant_df.columns:
            print(
                "    Target column missing. Skipping."
            )
            continue

        target_mask = np.isfinite(
            participant_df[target].to_numpy(
                dtype=float
            )
        )

        model_df = participant_df.loc[
            target_mask
        ].copy()

        if len(model_df) < MIN_ROWS:
            print(
                f"    Only {len(model_df)} valid rows. "
                f"Minimum is {MIN_ROWS}. Skipping."
            )
            continue

        X = model_df[
            participant_feature_columns
        ].copy()

        y = model_df[
            target
        ].astype(float)

        # ----------------------------------------------------
        # Remove columns that contain no usable value
        # anywhere for this participant/target.
        #
        # This is different from the participant-level
        # exclusion rule. We do NOT permanently remove
        # variables from the participant. We only avoid
        # impossible model columns for this target.
        # ----------------------------------------------------

        usable_columns = [
            column
            for column in X.columns
            if X[column].notna().any()
        ]

        X = X[
            usable_columns
        ]

        if X.shape[1] == 0:
            print(
                "    No usable model features. Skipping."
            )
            continue

        feature_cols = list(
            X.columns
        )

        if len(model_df) < 4:
            print(
                "    Not enough rows for time-series CV."
            )
            continue

        n_splits = min(
            N_SPLITS,
            len(model_df) - 1
        )

        if n_splits < 2:
            print(
                "    Not enough rows for TimeSeriesSplit."
            )
            continue

        tscv = TimeSeriesSplit(
            n_splits=n_splits
        )

        # ----------------------------------------------------
        # Model pipeline
        # ----------------------------------------------------

        pipeline = Pipeline(
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
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        min_samples_leaf=3,
                        n_jobs=-1
                    )
                ),
            ]
        )

        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        fold_importances = []

        successful_folds = 0

        # ====================================================
        # 9. TIME-SERIES CROSS-VALIDATION
        # ====================================================

        for fold_number, (
            train_idx,
            test_idx
        ) in enumerate(
            tscv.split(X),
            start=1
        ):

            X_train = X.iloc[
                train_idx
            ]

            X_test = X.iloc[
                test_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            y_test = y.iloc[
                test_idx
            ]

            if len(y_train) < 2:
                continue

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            if len(y_test) >= 2:
                r2 = r2_score(
                    y_test,
                    predictions
                )
            else:
                r2 = np.nan

            fold_mae.append(
                mae
            )

            fold_rmse.append(
                rmse
            )

            fold_r2.append(
                r2
            )

            successful_folds += 1

            # =================================================
            # 10. ROBUST FEATURE IMPORTANCE
            # =================================================

            model = pipeline.named_steps[
                "model"
            ]

            model_importances = (
                model.feature_importances_
            )

            # Because keep_empty_features=True is used,
            # the imputer preserves the complete feature
            # structure. Therefore the Random Forest should
            # return one importance value per input feature.

            if len(model_importances) != len(
                feature_cols
            ):
                raise ValueError(
                    "Feature importance length mismatch: "
                    f"model={len(model_importances)}, "
                    f"features={len(feature_cols)}"
                )

            fold_importance_full = np.asarray(
                model_importances,
                dtype=float
            )

            fold_importances.append(
                fold_importance_full
            )

        # ====================================================
        # 11. CHECK CV RESULT
        # ====================================================

        if successful_folds == 0:
            print(
                "    No successful CV folds."
            )
            continue

        mean_mae = float(
            np.mean(
                fold_mae
            )
        )

        mean_rmse = float(
            np.mean(
                fold_rmse
            )
        )

        valid_r2 = [
            value
            for value in fold_r2
            if np.isfinite(value)
        ]

        mean_r2 = (
            float(np.mean(valid_r2))
            if valid_r2
            else np.nan
        )

        performance_rows.append(
            {
                "Participant": participant,
                "Target": target,
                "N_Rows": len(model_df),
                "N_Features": len(feature_cols),
                "CV_Folds": successful_folds,
                "MAE": mean_mae,
                "RMSE": mean_rmse,
                "R2": mean_r2,
            }
        )

        print(
            f"    Rows: {len(model_df)}"
        )

        print(
            f"    Features: {len(feature_cols)}"
        )

        print(
            f"    Folds: {successful_folds}"
        )

        print(
            f"    MAE: {mean_mae:.4f}"
        )

        print(
            f"    RMSE: {mean_rmse:.4f}"
        )

        if np.isfinite(mean_r2):
            print(
                f"    R2: {mean_r2:.4f}"
            )
        else:
            print(
                "    R2: NaN"
            )


        # ====================================================
        # 12. MEAN FEATURE IMPORTANCE
        # ====================================================

        if fold_importances:

            mean_importances = np.mean(
                np.vstack(
                    fold_importances
                ),
                axis=0
            )

            for feature, importance in zip(
                feature_cols,
                mean_importances
            ):

                importance_rows.append(
                    {
                        "Participant": participant,
                        "Target": target,
                        "Feature": feature,
                        "Importance": float(
                            importance
                        ),
                    }
                )


# ============================================================
# 13. SAVE MODEL PERFORMANCE
# ============================================================

performance_df = pd.DataFrame(
    performance_rows
)

performance_path = os.path.join(
    OUTPUT_DIR,
    "ml_model_performance.csv"
)

performance_df.to_csv(
    performance_path,
    index=False
)


# ============================================================
# 14. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    importance_rows
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "ml_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MACHINE LEARNING COMPLETED")
print("=" * 70)

print(
    f"Participants: {len(participants)}"
)

print(
    f"Model results: {len(performance_df)}"
)

print(
    f"Feature importance rows: "
    f"{len(importance_df)}"
)

print(
    "\nSaved:"
)

print(
    f"  {performance_path}"
)

print(
    f"  {importance_path}"
)

print(
    f"  {feature_status_path}"
)

print("\nDone.")
