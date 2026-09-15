"""Prefect orchestration flow for the Module 3 telecom delinquency ETL pipeline.

The flow reuses the existing command-line scripts rather than duplicating business
logic. It orchestrates the sequence:

raw validation -> cleaning -> interim validation -> model-ready transformation -> processed validation

The raw source is treated as immutable. Any failed subprocess stops the flow.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = Path("data/raw/sample_data_intw.csv")
DEFAULT_INTERIM = Path("data/interim/sample_data_intw_cleaned.csv")
DEFAULT_PROCESSED = Path("data/processed/telecom_delinquency_model_ready.csv")


def _run_python_script(*args: str) -> None:
    """Run one repository script with the active Python interpreter."""
    command = [sys.executable, *args]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


@task(name="validate raw data", retries=0)
def validate_raw(raw_path: str) -> str:
    logger = get_run_logger()
    logger.info("Running raw-data validation on %s", raw_path)
    _run_python_script(
        "src/data_quality/gx_raw_validation.py",
        "--data",
        raw_path,
    )
    return raw_path


@task(name="clean raw data", retries=0)
def clean_raw(raw_path: str, interim_path: str) -> str:
    logger = get_run_logger()
    logger.info("Creating cleaned interim dataset %s", interim_path)
    _run_python_script(
        "src/data/clean_interim.py",
        "--data",
        raw_path,
        "--output",
        interim_path,
    )
    return interim_path


@task(name="validate interim data", retries=0)
def validate_interim(interim_path: str) -> str:
    logger = get_run_logger()
    logger.info("Running interim-data validation on %s", interim_path)
    _run_python_script(
        "src/data_quality/gx_interim_validation.py",
        "--data",
        interim_path,
    )
    return interim_path


@task(name="build model-ready dataset", retries=0)
def build_model_ready(interim_path: str, processed_path: str) -> str:
    logger = get_run_logger()
    logger.info("Building model-ready dataset %s", processed_path)
    _run_python_script(
        "src/features/build_model_dataset.py",
        "--data",
        interim_path,
        "--output",
        processed_path,
    )
    return processed_path


@task(name="validate processed data", retries=0)
def validate_processed(processed_path: str) -> str:
    logger = get_run_logger()
    logger.info("Running processed-data validation on %s", processed_path)
    _run_python_script(
        "src/data_quality/gx_processed_validation.py",
        "--data",
        processed_path,
    )
    return processed_path


@flow(name="telecom-delinquency-etl", log_prints=True)
def telecom_delinquency_etl(
    raw_path: str = str(DEFAULT_RAW),
    interim_path: str = str(DEFAULT_INTERIM),
    processed_path: str = str(DEFAULT_PROCESSED),
) -> str:
    """Run the leakage-conscious, validated ETL path from raw to model-ready data."""
    raw_checked = validate_raw(raw_path)
    interim = clean_raw(raw_checked, interim_path)
    interim_checked = validate_interim(interim)
    processed = build_model_ready(interim_checked, processed_path)
    processed_checked = validate_processed(processed)
    return processed_checked


if __name__ == "__main__":
    telecom_delinquency_etl()
