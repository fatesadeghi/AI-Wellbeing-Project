import os
import glob
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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data", "pmdata")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "final_ml")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. IMPORTANT VARIABLES
# ============================================================

BEHAVIOR_VARS = [
    "Steps",
    "Exercise_Count",
    "Exercise_Duration",
    "Sleep_Hours",
    "Sleep_Score",
    "Deep_Sleep_Minutes",
    "Sleep_Restlessness",
]

WELLBEING_VARS = [
    "mood",
    "stress",
    "fatigue",
    "readiness",
    "sleep_quality",
]


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 3


# ============================================================
# 4. FEATURE CREATION
# ============================================================

def create_features(df):
    """
    Create predictive features using only information
    available before the prediction day.

    For prediction at day t+1:
        - lag1      = behavior at day t
        - mean7     = mean of previous 7 days
        - std7      = variability across previous 7 days
        - deviation7 = day t value - previous 7-day mean
        - change7   = day t value - value 7 days earlier
    """

    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    for var in BEHAVIOR_VARS:

        # Previous day
        df[f"{var}_lag1"] = df[var].shift(1)

        # Seven-day history
        df[f"{var}_mean7"] = (
            df[var]
            .shift(1)
            .rolling(window=7, min_periods=3)
            .mean()
        )

        df[f"{var}_std7"] = (
            df[var]
            .shift(1)
            .rolling(window=7, min_periods=3)
            .std()
        )

        # Yesterday compared with its recent 7-day baseline
        df[f"{var}_deviation7"] = (
            df[f"{var}_lag1"] - df[f"{var}_mean7"]
        )

        # Yesterday compared with 7 days earlier
        df[f"{var}_change7"] = (
            df[var].shift(1) - df[var].shift(8)
        )

    return df


# ============================================================
# 5. CREATE NEXT-DAY WELLBEING CHANGE
# ============================================================

def create_targets(df):

    df = df.copy()

    for var in WELLBEING_VARS:

        df[f"NextDayChange_{var}"] = (
            df[var].shift(-1) - df[var]
        )

    return df


# ============================================================
# 6. CHECK IMPORTANT COLUMNS
# ============================================================

def has_all_important_columns(df):

    required = BEHAVIOR_VARS + WELLBEING_VARS

    missing = [
        col for col in required
        if col not in df.columns
    ]

    return len(missing) == 0, missing


# ============================================================
# 7. MODEL
# ============================================================

def run_model(df, target):

    feature_cols = []

    for var in BEHAVIOR_VARS:
        feature_cols.extend([
            f"{var}_lag1",
            f"{var}_mean7",
            f"{var}_std7",
            f"{var}_deviation7",
            f"{var}_change7",
        ])

    data = df[feature_cols + [target]].copy()

    # Keep chronological order
    data = data.reset_index(drop=True)

    # Target must exist
    data = data[data[target].notna()].copy()

    if len(data) < 15:
        return None

    X = data[feature_cols]
    y = data[target]

    # --------------------------------------------------------
    # Time-series cross-validation
    # --------------------------------------------------------

    n_splits = min(3, len(data) - 1)

    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    predictions = []

    for fold, (train_idx, test_idx) in enumerate(
        tscv.split(X),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Imputation is fitted ONLY on training data
        model = Pipeline([
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
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        predictions.append(
            pd.DataFrame({
                "actual": y_test.values,
                "predicted": y_pred,
                "fold": fold
            })
        )

    if not fold_results:
        return None

    metrics = pd.DataFrame(fold_results)

    result = {
        "MAE": metrics["MAE"].mean(),
        "RMSE": metrics["RMSE"].mean(),
        "R2": metrics["R2"].mean(),
        "n_observations": len(data),
        "n_folds": len(metrics)
    }

    return result


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINAL WELLBEING ML — PARTICIPANTS WITH ALL IMPORTANT DATA")
    print("=" * 80)

    files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "p*_daily_merged.csv"
            )
        )
    )

    print(f"Participant files found: {len(files)}")

    eligible_participants = []
    excluded_participants = []

    # --------------------------------------------------------
    # First pass:
    # identify participants with all important columns
    # --------------------------------------------------------

    for file in files:

        participant = os.path.basename(file).replace(
            "_daily_merged.csv",
            ""
        )

        print("\n" + "-" * 80)
        print(f"Checking {participant}")

        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"  ERROR reading file: {e}")
            excluded_participants.append({
                "participant": participant,
                "reason": "Read_Error"
            })
            continue

        has_all, missing = has_all_important_columns(df)

        if not has_all:

            print(
                "  EXCLUDED — missing important columns:"
            )

            for col in missing:
                print(f"    - {col}")

            excluded_participants.append({
                "participant": participant,
                "reason": "Missing_Important_Column",
                "missing_columns": ", ".join(missing)
            })

            continue

        print("  ELIGIBLE — all important columns are present")

        eligible_participants.append(
            (participant, df)
        )

    print("\n" + "=" * 80)
    print("ELIGIBILITY SUMMARY")
    print("=" * 80)

    print(
        f"Eligible participants: "
        f"{len(eligible_participants)}"
    )

    print(
        f"Excluded participants: "
        f"{len(excluded_participants)}"
    )

    # --------------------------------------------------------
    # Save participant eligibility
    # --------------------------------------------------------

    eligibility_rows = []

    for participant, _ in eligible_participants:

        eligibility_rows.append({
            "participant": participant,
            "status": "Eligible",
            "reason": "All important columns present"
        })

    for row in excluded_participants:
        eligibility_rows.append(row)

    pd.DataFrame(
        eligibility_rows
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "participant_eligibility.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    all_results = []

    for participant, original_df in eligible_participants:

        print("\n" + "=" * 80)
        print(f"ML FOR {participant}")
        print("=" * 80)

        df = original_df.copy()

        # Date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            df = df.sort_values("date")

        # Convert required variables to numeric
        for col in BEHAVIOR_VARS + WELLBEING_VARS:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # Create lag/history features
        df = create_features(df)

        # Create next-day wellbeing changes
        df = create_targets(df)

        # ----------------------------------------------------
        # Run one model for each wellbeing target
        # ----------------------------------------------------

        for wellbeing in WELLBEING_VARS:

            target = f"NextDayChange_{wellbeing}"

            result = run_model(
                df,
                target
            )

            if result is None:

                print(
                    f"  {wellbeing}: "
                    f"not enough usable observations"
                )

                continue

            print(
                f"  {wellbeing}: "
                f"n={result['n_observations']} | "
                f"MAE={result['MAE']:.4f} | "
                f"RMSE={result['RMSE']:.4f} | "
                f"R2={result['R2']:.4f}"
            )

            all_results.append({
                "participant": participant,
                "target": wellbeing,
                **result
            })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    results_path = os.path.join(
        OUTPUT_DIR,
        "personalized_ml_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary by wellbeing target
    # --------------------------------------------------------

    if not results_df.empty:

        summary = (
            results_df
            .groupby("target")
            .agg(
                participants=("participant", "nunique"),
                observations=("n_observations", "sum"),
                MAE_mean=("MAE", "mean"),
                RMSE_mean=("RMSE", "mean"),
                R2_mean=("R2", "mean")
            )
            .reset_index()
        )

        summary_path = os.path.join(
            OUTPUT_DIR,
            "ml_summary_by_wellbeing.csv"
        )

        summary.to_csv(
            summary_path,
            index=False
        )

        print("\n" + "=" * 80)
        print("FINAL ML SUMMARY")
        print("=" * 80)

        print(summary.to_string(index=False))

    else:

        print("\nNo valid ML results were produced.")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
