from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "ML" / "Results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FEATURES_FILE = (
    RESULTS_DIR
    / "behavior_7day_features.csv"
)

WELLBEING_FILE = (
    RESULTS_DIR
    / "wellbeing_index.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "personal_model_predictions.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "personal_model_summary.csv"
)


RANDOM_STATE = 42
N_ESTIMATORS = 200
MIN_TRAINING_SAMPLES = 10


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


FEATURE_COLUMNS = [
    f"{variable}_{feature_type}"
    for variable in BEHAVIOR_VARIABLES
    for feature_type in [
        "7d_slope",
        "7d_change",
    ]
]


def build_model():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def main():

    print()
    print("=" * 80)
    print("TRAIN PERSONALIZED RANDOM FOREST MODELS")
    print("=" * 80)
    print()

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURES_FILE}"
        )

    if not WELLBEING_FILE.exists():
        raise FileNotFoundError(
            f"Wellbeing file not found:\n{WELLBEING_FILE}"
        )

    features_df = pd.read_csv(
        FEATURES_FILE
    )

    wellbeing_df = pd.read_csv(
        WELLBEING_FILE
    )

    features_df["target_date"] = pd.to_datetime(
        features_df["target_date"],
        errors="coerce",
    )

    wellbeing_df["date"] = pd.to_datetime(
        wellbeing_df["date"],
        errors="coerce",
    )

    features_df = features_df.dropna(
        subset=[
            "participant_id",
            "target_date",
        ]
    ).copy()

    wellbeing_df = wellbeing_df.dropna(
        subset=[
            "participant_id",
            "date",
        ]
    ).copy()

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns:\n"
            f"{missing_features}"
        )

    if "Wellbeing_Index" not in wellbeing_df.columns:
        raise ValueError(
            "Wellbeing_Index column not found."
        )

    merged_df = features_df.merge(
        wellbeing_df[
            [
                "participant_id",
                "date",
                "Wellbeing_Index",
            ]
        ],
        left_on=[
            "participant_id",
            "target_date",
        ],
        right_on=[
            "participant_id",
            "date",
        ],
        how="left",
    )

    merged_df = merged_df.drop(
        columns=["date"]
    )

    predictions = []
    summaries = []

    participant_ids = sorted(
        merged_df["participant_id"]
        .dropna()
        .unique()
    )

    print(
        f"Participants found: "
        f"{len(participant_ids)}"
    )

    print()

    for participant_id in participant_ids:

        participant_data = (
            merged_df[
                merged_df["participant_id"]
                == participant_id
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        participant_predictions = 0
        participant_insufficient = 0
        participant_missing = 0

        for _, current_row in participant_data.iterrows():

            target_date = current_row[
                "target_date"
            ]

            historical = participant_data[
                participant_data["target_date"]
                < target_date
            ].copy()

            historical = historical[
                historical["Wellbeing_Index"]
                .notna()
            ].copy()

            historical = historical[
                historical[
                    FEATURE_COLUMNS
                ].notna().any(axis=1)
            ].copy()

            if len(historical) < MIN_TRAINING_SAMPLES:

                predictions.append(
                    {
                        "participant_id": participant_id,
                        "target_date": target_date,
                        "actual_wellbeing": current_row[
                            "Wellbeing_Index"
                        ],
                        "training_samples": len(
                            historical
                        ),
                        "predicted_wellbeing": None,
                        "prediction_status":
                            "Insufficient_History",
                    }
                )

                participant_insufficient += 1

                continue

            target_features = current_row[
                FEATURE_COLUMNS
            ].to_frame().T

            if target_features.isna().all(
                axis=1
            ).iloc[0]:

                predictions.append(
                    {
                        "participant_id": participant_id,
                        "target_date": target_date,
                        "actual_wellbeing": current_row[
                            "Wellbeing_Index"
                        ],
                        "training_samples": len(
                            historical
                        ),
                        "predicted_wellbeing": None,
                        "prediction_status":
                            "Missing_Target_Features",
                    }
                )

                participant_missing += 1

                continue

            X_train = historical[
                FEATURE_COLUMNS
            ]

            y_train = historical[
                "Wellbeing_Index"
            ]

            X_target = target_features

            model = build_model()

            model.fit(
                X_train,
                y_train,
            )

            predicted_value = model.predict(
                X_target
            )[0]

            predictions.append(
                {
                    "participant_id": participant_id,
                    "target_date": target_date,
                    "actual_wellbeing": current_row[
                        "Wellbeing_Index"
                    ],
                    "training_samples": len(
                        historical
                    ),
                    "predicted_wellbeing":
                        predicted_value,
                    "prediction_status":
                        "Predicted",
                }
            )

            participant_predictions += 1

        summaries.append(
            {
                "participant_id": participant_id,
                "total_target_days":
                    len(participant_data),
                "predicted_days":
                    participant_predictions,
                "insufficient_history_days":
                    participant_insufficient,
                "missing_target_feature_days":
                    participant_missing,
            }
        )

        print(
            f"{participant_id}: "
            f"{participant_predictions} predicted, "
            f"{participant_insufficient} insufficient history, "
            f"{participant_missing} missing features"
        )

    predictions_df = pd.DataFrame(
        predictions
    )

    summaries_df = pd.DataFrame(
        summaries
    )

    predictions_df = predictions_df.sort_values(
        [
            "participant_id",
            "target_date",
        ]
    ).reset_index(drop=True)

    summaries_df = summaries_df.sort_values(
        "participant_id"
    ).reset_index(drop=True)

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    summaries_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("PERSONALIZED RANDOM FOREST COMPLETED")
    print("=" * 80)
    print()

    print(
        f"Prediction rows: "
        f"{len(predictions_df)}"
    )

    print(
        f"Predicted rows: "
        f"{(
            predictions_df['prediction_status']
            == 'Predicted'
        ).sum()}"
    )

    print()

    print(
        f"Saved:\n{PREDICTIONS_FILE}"
    )

    print(
        f"Saved:\n{SUMMARY_FILE}"
    )

    print()


if __name__ == "__main__":
    main()
