from pathlib import Path

import numpy as np
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


FEATURE_COLUMNS = []

for variable in BEHAVIOR_VARIABLES:

    FEATURE_COLUMNS.append(
        f"{variable}_7d_slope"
    )

    FEATURE_COLUMNS.append(
        f"{variable}_7d_change"
    )


def build_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def get_person_label(participant_id):

    digits = "".join(
        character
        for character in str(participant_id)
        if character.isdigit()
    )

    if digits:
        return f"Person {int(digits)}"

    return str(participant_id)


def main():

    print()
    print("=" * 80)
    print("TRAIN PERSONALIZED RANDOM FOREST MODELS")
    print("=" * 80)
    print()

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURES_FILE}"
        )

    if not WELLBEING_FILE.exists():
        raise FileNotFoundError(
            f"Wellbeing file not found:\n"
            f"{WELLBEING_FILE}"
        )

    features_df = pd.read_csv(
        FEATURES_FILE
    )

    wellbeing_df = pd.read_csv(
        WELLBEING_FILE
    )

    required_feature_columns = [
        "participant_id",
        "target_date",
    ]

    for column in required_feature_columns:

        if column not in features_df.columns:
            raise ValueError(
                f"Feature file must contain "
                f"'{column}'."
            )

    required_wellbeing_columns = [
        "participant_id",
        "date",
        "Wellbeing_Index",
    ]

    for column in required_wellbeing_columns:

        if column not in wellbeing_df.columns:
            raise ValueError(
                f"Wellbeing file must contain "
                f"'{column}'."
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

    wellbeing_df["Wellbeing_Index"] = pd.to_numeric(
        wellbeing_df["Wellbeing_Index"],
        errors="coerce",
    )

    for column in FEATURE_COLUMNS:

        if column not in features_df.columns:
            features_df[column] = np.nan

        features_df[column] = pd.to_numeric(
            features_df[column],
            errors="coerce",
        )

    features_df = (
        features_df.sort_values(
            [
                "participant_id",
                "target_date",
            ]
        )
        .reset_index(drop=True)
    )

    wellbeing_df = (
        wellbeing_df.sort_values(
            [
                "participant_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    participants = sorted(
        features_df[
            "participant_id"
        ].unique()
    )

    print(
        f"Participants found: "
        f"{len(participants)}"
    )

    print(
        f"Feature columns: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print()

    prediction_rows = []
    summary_rows = []

    for participant in participants:

        person_label = get_person_label(
            participant
        )

        print("-" * 80)
        print(
            f"Participant: "
            f"{person_label}"
        )

        participant_features = (
            features_df[
                features_df[
                    "participant_id"
                ] == participant
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        participant_wellbeing = (
            wellbeing_df[
                wellbeing_df[
                    "participant_id"
                ] == participant
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if participant_wellbeing.empty:

            print(
                "  No wellbeing data"
            )

            continue

        predicted_count = 0
        insufficient_count = 0
        no_feature_count = 0

        for _, target_row in (
            participant_features.iterrows()
        ):

            target_date = (
                target_row["target_date"]
            )

            target_features = (
                target_row[
                    FEATURE_COLUMNS
                ]
                .astype(float)
            )

            if (
                target_features.notna().sum()
                == 0
            ):

                no_feature_count += 1

                actual_rows = (
                    participant_wellbeing[
                        participant_wellbeing[
                            "date"
                        ] == target_date
                    ]
                )

                actual_value = np.nan

                if not actual_rows.empty:
                    actual_value = (
                        actual_rows[
                            "Wellbeing_Index"
                        ].iloc[0]
                    )

                prediction_rows.append(
                    {
                        "participant_id":
                            participant,
                        "person":
                            person_label,
                        "target_date":
                            target_date,
                        "actual_wellbeing":
                            actual_value,
                        "predicted_wellbeing":
                            np.nan,
                        "training_samples":
                            0,
                        "prediction_status":
                            "No_Features",
                    }
                )

                continue

            training_features = (
                participant_features[
                    participant_features[
                        "target_date"
                    ] < target_date
                ].copy()
            )

            training_features = (
                training_features.merge(
                    participant_wellbeing[
                        [
                            "date",
                            "Wellbeing_Index",
                        ]
                    ],
                    left_on="target_date",
                    right_on="date",
                    how="inner",
                )
            )

            training_features = (
                training_features[
                    training_features[
                        "Wellbeing_Index"
                    ].notna()
                ]
            )

            X_train = training_features[
                FEATURE_COLUMNS
            ]

            y_train = training_features[
                "Wellbeing_Index"
            ]

            valid_training_rows = (
                X_train.notna().any(axis=1)
                & y_train.notna()
            )

            X_train = X_train[
                valid_training_rows
            ]

            y_train = y_train[
                valid_training_rows
            ]

            if len(X_train) < MIN_TRAINING_SAMPLES:

                insufficient_count += 1

                actual_rows = (
                    participant_wellbeing[
                        participant_wellbeing[
                            "date"
                        ] == target_date
                    ]
                )

                actual_value = np.nan

                if not actual_rows.empty:
                    actual_value = (
                        actual_rows[
                            "Wellbeing_Index"
                        ].iloc[0]
                    )

                prediction_rows.append(
                    {
                        "participant_id":
                            participant,
                        "person":
                            person_label,
                        "target_date":
                            target_date,
                        "actual_wellbeing":
                            actual_value,
                        "predicted_wellbeing":
                            np.nan,
                        "training_samples":
                            len(X_train),
                        "prediction_status":
                            "Insufficient_Training_Data",
                    }
                )

                continue

            model = build_model()

            model.fit(
                X_train,
                y_train,
            )

            X_target = pd.DataFrame(
                [
                    target_features.values
                ],
                columns=FEATURE_COLUMNS,
            )

            predicted_value = (
                model.predict(
                    X_target
                )[0]
            )

            actual_rows = (
                participant_wellbeing[
                    participant_wellbeing[
                        "date"
                    ] == target_date
                ]
            )

            actual_value = np.nan

            if not actual_rows.empty:
                actual_value = (
                    actual_rows[
                        "Wellbeing_Index"
                    ].iloc[0]
                )

            prediction_rows.append(
                {
                    "participant_id":
                        participant,
                    "person":
                        person_label,
                    "target_date":
                        target_date,
                    "actual_wellbeing":
                        actual_value,
                    "predicted_wellbeing":
                        predicted_value,
                    "training_samples":
                        len(X_train),
                    "prediction_status":
                        "Predicted",
                }
            )

            predicted_count += 1

        summary_rows.append(
            {
                "participant_id":
                    participant,
                "person":
                    person_label,
                "total_prediction_dates":
                    len(participant_features),
                "predictions_created":
                    predicted_count,
                "insufficient_training_dates":
                    insufficient_count,
                "no_feature_dates":
                    no_feature_count,
            }
        )

        print(
            f"  Predictions: "
            f"{predicted_count}"
        )

        print(
            f"  Insufficient training: "
            f"{insufficient_count}"
        )

        print(
            f"  No features: "
            f"{no_feature_count}"
        )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if not predictions_df.empty:

        predictions_df = (
            predictions_df.sort_values(
                [
                    "participant_id",
                    "target_date",
                ]
            )
            .reset_index(drop=True)
        )

    if not summary_df.empty:

        summary_df = (
            summary_df.sort_values(
                "participant_id"
            )
            .reset_index(drop=True)
        )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("PERSONALIZED MODEL TRAINING COMPLETE")
    print("=" * 80)
    print()

    print(
        f"Total prediction rows: "
        f"{len(predictions_df)}"
    )

    if not predictions_df.empty:

        successful_predictions = (
            predictions_df[
                "prediction_status"
            ] == "Predicted"
        ).sum()

    else:

        successful_predictions = 0

    print(
        f"Successful predictions: "
        f"{successful_predictions}"
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
