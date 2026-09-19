import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "pmdata"
)

BASELINE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "baseline",
    "personalized_baselines.csv"
)

DEVIATION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "baseline",
    "daily_personalized_deviations.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "same_day"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "same_day_behavior_wellbeing_relationships.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

MIN_N = 10
BASELINE_FRACTION = 0.50
MIN_BASELINE_N = 7

BEHAVIOR_VARS = [
    "Steps",
    "Sleep_Duration_Score",
    "Sleep_Score",
    "Sleep_Deep_Minutes",
    "Sleep_Restlessness",
    "Exercise_Distance",
    "Exercise_Calories",
    "Exercise_Duration",
]

WELLBEING_VARS = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_quality",
    "stress",
]


# ============================================================
# 3. SAFE PEARSON CORRELATION
# ============================================================

def safe_pearson(x, y, min_n=MIN_N):
    """
    Calculate Pearson correlation safely.

    Returns:
        r, p, n
    """

    temp = pd.DataFrame({
        "x": x,
        "y": y
    }).replace([np.inf, -np.inf], np.nan).dropna()

    n = len(temp)

    if n < min_n:
        return np.nan, np.nan, n

    x_values = temp["x"].to_numpy(dtype=float)
    y_values = temp["y"].to_numpy(dtype=float)

    # Prevent constant-input Pearson warnings
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        return np.nan, np.nan, n

    if np.std(x_values, ddof=1) == 0:
        return np.nan, np.nan, n

    if np.std(y_values, ddof=1) == 0:
        return np.nan, np.nan, n

    try:
        r, p = pearsonr(x_values, y_values)
        return r, p, n

    except Exception:
        return np.nan, np.nan, n


# ============================================================
# 4. LOAD BASELINE / DEVIATION DATA
# ============================================================

baselines = pd.read_csv(BASELINE_FILE)

deviations = pd.read_csv(
    DEVIATION_FILE,
    parse_dates=["Date"]
)


# ============================================================
# 5. PARTICIPANT FILES
# ============================================================

participant_files = sorted(
    [
        f
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
    ]
)

print(
    f"Found {len(participant_files)} participant files."
)


# ============================================================
# 6. ANALYSIS
# ============================================================

results = []

for filename in participant_files:

    participant = os.path.splitext(filename)[0]

    file_path = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        print(
            f"Skipping {participant}: Date column missing."
        )
        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = (
        df.dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    n_total = len(df)

    if n_total == 0:
        continue

    baseline_n = int(
        np.floor(
            n_total * BASELINE_FRACTION
        )
    )

    if baseline_n < MIN_BASELINE_N:
        print(
            f"Skipping {participant}: "
            f"baseline N={baseline_n} < {MIN_BASELINE_N}"
        )
        continue

    # --------------------------------------------------------
    # Chronological 50/50 split
    # --------------------------------------------------------

    analysis_df = df.iloc[baseline_n:].copy()

    if analysis_df.empty:
        continue

    # --------------------------------------------------------
    # Keep only wellbeing variables that exist
    # --------------------------------------------------------

    available_wellbeing = [
        v
        for v in WELLBEING_VARS
        if v in analysis_df.columns
    ]

    if not available_wellbeing:
        print(
            f"Skipping {participant}: "
            "no wellbeing variables found."
        )
        continue

    wellbeing_df = analysis_df[
        ["Date"] + available_wellbeing
    ].copy()

    # --------------------------------------------------------
    # Participant deviations
    # --------------------------------------------------------

    participant_deviations = deviations[
        deviations["Participant"].astype(str) == participant
    ].copy()

    if participant_deviations.empty:
        print(
            f"Skipping {participant}: "
            "no deviation data found."
        )
        continue

    participant_deviations = (
        participant_deviations[
            ["Date"]
            + [
                f"{v}_Z"
                for v in BEHAVIOR_VARS
                if f"{v}_Z" in participant_deviations.columns
            ]
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Merge same-day behavior deviations + wellbeing
    # --------------------------------------------------------

    merged = pd.merge(
        participant_deviations,
        wellbeing_df,
        on="Date",
        how="inner"
    )

    if merged.empty:
        continue

    # --------------------------------------------------------
    # Participant-level correlations
    # --------------------------------------------------------

    for behavior in BEHAVIOR_VARS:

        behavior_col = f"{behavior}_Z"

        if behavior_col not in merged.columns:
            continue

        for wellbeing in available_wellbeing:

            r, p, n = safe_pearson(
                merged[behavior_col],
                merged[wellbeing]
            )

            results.append({
                "Participant": participant,
                "Behavioral_Variable": behavior,
                "Wellbeing_Variable": wellbeing,
                "r": r,
                "p": p,
                "N": n,
                "Included": "Yes" if n >= MIN_N else "No"
            })


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    print("No results generated.")
else:

    results_df = results_df.sort_values(
        [
            "Participant",
            "Behavioral_Variable",
            "Wellbeing_Variable"
        ]
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("SAME-DAY ANALYSIS COMPLETE")
    print("=" * 60)
    print(
        f"Total relationships: {len(results_df)}"
    )
    print(
        f"Relationships with N >= {MIN_N}: "
        f"{(results_df['N'] >= MIN_N).sum()}"
    )
    print(
        f"Output saved to:\n{OUTPUT_FILE}"
    )
    print("=" * 60)
