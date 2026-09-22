#run_pipeline.py

Purpose
-------
Run the complete personalized wellbeing ML pipeline
in the correct chronological order.

Pipeline
--------
01_build_wellbeing_index.py
        ↓
02_build_7day_features.py
        ↓
03_train_personal_model.py
        ↓
04_predict_and_report.py

Usage
-----
From the project root:

    python ML/run_pipeline.py

The pipeline stops immediately if any stage fails.
This prevents incomplete or invalid downstream results.
"""

from pathlib import Path
import subprocess
import sys
import time


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DIR = PROJECT_ROOT / "ML"


# ============================================================
# 2. PIPELINE CONFIGURATION
# ============================================================

PIPELINE_STAGES = [
    (
        "01",
        "Build Wellbeing Index",
        ML_DIR / "01_build_wellbeing_index.py",
    ),
    (
        "02",
        "Build 7-Day Behavioral Features",
        ML_DIR / "02_build_7day_features.py",
    ),
    (
        "03",
        "Train Personalized Random Forest",
        ML_DIR / "03_train_personal_model.py",
    ),
    (
        "04",
        "Predict and Generate Report",
        ML_DIR / "04_predict_and_report.py",
    ),
]


# ============================================================
# 3. DISPLAY HELPERS
# ============================================================

def print_header() -> None:
    """Print the pipeline header."""

    print()
    print("=" * 80)
    print("AI WELLBEING PROJECT")
    print("PERSONALIZED ML PIPELINE")
    print("=" * 80)
    print()

    print(
        f"Project root:\n"
        f"  {PROJECT_ROOT}"
    )

    print()

    print(
        "Pipeline stages:"
    )

    for number, name, _ in PIPELINE_STAGES:
        print(
            f"  {number}. {name}"
        )

    print()


def print_stage_start(
    stage_number: str,
    stage_name: str,
) -> None:
    """Print stage start information."""

    print()
    print("=" * 80)
    print(
        f"STAGE {stage_number}: {stage_name}"
    )
    print("=" * 80)
    print()


def print_stage_success(
    stage_number: str,
    stage_name: str,
    elapsed_seconds: float,
) -> None:
    """Print successful stage completion."""

    print()
    print("-" * 80)
    print(
        f"STAGE {stage_number} COMPLETED: "
        f"{stage_name}"
    )
    print(
        f"Time: {elapsed_seconds:.2f} seconds"
    )
    print("-" * 80)


def print_stage_failure(
    stage_number: str,
    stage_name: str,
    elapsed_seconds: float,
) -> None:
    """Print failed stage information."""

    print()
    print("-" * 80)
    print(
        f"STAGE {stage_number} FAILED: "
        f"{stage_name}"
    )
    print(
        f"Time: {elapsed_seconds:.2f} seconds"
    )
    print("-" * 80)


# ============================================================
# 4. VALIDATE PIPELINE
# ============================================================

def validate_pipeline() -> None:
    """
    Make sure all pipeline scripts exist before execution.
    """

    if not ML_DIR.exists():

        raise FileNotFoundError(
            f"ML directory not found:\n{ML_DIR}"
        )

    missing_scripts = []

    for _, _, script_path in PIPELINE_STAGES:

        if not script_path.exists():

            missing_scripts.append(
                str(script_path)
            )

    if missing_scripts:

        raise FileNotFoundError(
            "The following pipeline scripts are missing:\n"
            + "\n".join(missing_scripts)
        )


# ============================================================
# 5. RUN ONE STAGE
# ============================================================

def run_stage(
    stage_number: str,
    stage_name: str,
    script_path: Path,
) -> float:
    """
    Execute one pipeline stage.

    Returns
    -------
    elapsed execution time in seconds.

    Raises
    ------
    subprocess.CalledProcessError
        If the stage exits with a non-zero status.
    """

    print_stage_start(
        stage_number,
        stage_name,
    )

    start_time = time.perf_counter()

    command = [
        sys.executable,
        str(script_path),
    ]

    print(
        "Running:"
    )

    print(
        "  "
        + " ".join(
            f'"{part}"'
            if " " in part
            else part
            for part in command
        )
    )

    print()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if result.returncode != 0:

        print_stage_failure(
            stage_number,
            stage_name,
            elapsed_seconds,
        )

        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=command,
        )

    print_stage_success(
        stage_number,
        stage_name,
        elapsed_seconds,
    )

    return elapsed_seconds


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main() -> None:

    pipeline_start = time.perf_counter()

    print_header()

    # --------------------------------------------------------
    # Validate all scripts before starting.
    # --------------------------------------------------------

    try:

        validate_pipeline()

    except Exception as error:

        print()
        print("=" * 80)
        print("PIPELINE VALIDATION FAILED")
        print("=" * 80)
        print()
        print(error)
        print()

        sys.exit(1)

    # --------------------------------------------------------
    # Run stages sequentially.
    # --------------------------------------------------------

    completed_stages = []

    try:

        for (
            stage_number,
            stage_name,
            script_path,
        ) in PIPELINE_STAGES:

            elapsed = run_stage(
                stage_number,
                stage_name,
                script_path,
            )

            completed_stages.append(
                {
                    "number": stage_number,
                    "name": stage_name,
                    "time": elapsed,
                }
            )

    except subprocess.CalledProcessError as error:

        total_time = (
            time.perf_counter()
            - pipeline_start
        )

        print()
        print("=" * 80)
        print("PIPELINE STOPPED")
        print("=" * 80)
        print()

        print(
            "A pipeline stage failed."
        )

        print(
            "No later stages were executed."
        )

        print()

        print(
            f"Return code: "
            f"{error.returncode}"
        )

        print(
            f"Total elapsed time: "
            f"{total_time:.2f} seconds"
        )

        print()

        if completed_stages:

            print(
                "Successfully completed stages:"
            )

            for stage in completed_stages:

                print(
                    f"  {stage['number']}. "
                    f"{stage['name']} "
                    f"({stage['time']:.2f}s)"
                )

        print()

        sys.exit(
            error.returncode
        )

    except Exception as error:

        total_time = (
            time.perf_counter()
            - pipeline_start
        )

        print()
        print("=" * 80)
        print("PIPELINE ERROR")
        print("=" * 80)
        print()

        print(error)

        print()

        print(
            f"Total elapsed time: "
            f"{total_time:.2f} seconds"
        )

        sys.exit(1)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.perf_counter()
        - pipeline_start
    )

    print()
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()

    print(
        f"Stages completed: "
        f"{len(completed_stages)}/{len(PIPELINE_STAGES)}"
    )

    print(
        f"Total execution time: "
        f"{total_time:.2f} seconds"
    )

    print()

    print(
        "Completed stages:"
    )

    for stage in completed_stages:

        print(
            f"  ✓ {stage['number']}. "
            f"{stage['name']} "
            f"({stage['time']:.2f}s)"
        )

    print()

    print(
        "Final results are available in:"
    )

    print(
        f"  {RESULTS_DIRECTORY}"
    )


# ============================================================
# 7. RESULTS DIRECTORY
# ============================================================

RESULTS_DIRECTORY = ML_DIR / "Results"


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":
    main()
