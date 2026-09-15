"""Prefect orchestration flow for the Module 3 telecom delinquency ETL pipeline.

The flow reuses the existing command-line scripts rather than duplicating business
logic. It orchestrates the sequence:

raw validation -> cleaning -> interim validation -> model-ready transformation -> processed validation

The raw source is treated as immutable. Raw validation is diagnostic: known quality
failures are expected before cleaning and therefore do not stop the flow when a
validation report is successfully produced. Interim and processed validation are hard
gates and do stop the flow on failure.

A privacy-safe JSONL audit log records pipeline access, transformations, validation
outcomes, identifier minimisation, and overall run status without storing raw personal
identifiers or row-level personal data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, get_run_logger, task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.privacy.privacy_audit import append_privacy_event


DEFAULT_RAW = Path("data/raw/sample_data_intw.csv")
DEFAULT_INTERIM = Path("data/interim/sample_data_intw_cleaned.csv")
DEFAULT_PROCESSED = Path("data/processed/telecom_delinquency_model_ready.csv")
RAW_VALIDATION_REPORT = PROJECT_ROOT / "reports/tables/gx_raw_validation.json"


def _run_python_script(*args: str, allow_nonzero: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one repository script with the active Python interpreter."""
    command = [sys.executable, *args]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0 and not allow_nonzero:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Pipeline command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{message[-4000:]}"
        )

    return completed


@task(name="validate raw data", retries=0)
def validate_raw(raw_path: str, run_id: str) -> str:
    logger = get_run_logger()
    logger.info("Running raw-data validation on %s", raw_path)
    append_privacy_event(
        run_id=run_id,
        event_type="data_access",
        stage="raw_validation",
        status="started",
        details={"dataset_path": raw_path, "purpose": "raw data-quality validation"},
    )

    completed = _run_python_script(
        "src/data_quality/gx_raw_validation.py",
        "--data",
        raw_path,
        allow_nonzero=True,
    )

    if not RAW_VALIDATION_REPORT.exists():
        raise RuntimeError(
            "Raw-data validation did not produce reports/tables/gx_raw_validation.json."
        )

    try:
        with RAW_VALIDATION_REPORT.open("r", encoding="utf-8") as handle:
            validation_result = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Raw-data validation report could not be read.") from exc

    success = bool(validation_result.get("success", False))
    if success:
        logger.info("Raw validation passed without quality exceptions.")
    else:
        logger.warning(
            "Raw validation found expected pre-cleaning quality issues. "
            "The report was created successfully, so the pipeline will continue to cleaning."
        )

    if completed.returncode != 0:
        logger.info(
            "Raw validator returned exit code %s because one or more expectations failed; "
            "this is diagnostic at the raw stage, not a pipeline execution failure.",
            completed.returncode,
        )

    append_privacy_event(
        run_id=run_id,
        event_type="validation",
        stage="raw_validation",
        status="passed" if success else "diagnostic_findings",
        details={
            "report": "reports/tables/gx_raw_validation.json",
            "blocking": False,
        },
    )
    return raw_path


@task(name="clean raw data", retries=0)
def clean_raw(raw_path: str, interim_path: str, run_id: str) -> str:
    logger = get_run_logger()
    logger.info("Creating cleaned interim dataset %s", interim_path)
    _run_python_script(
        "src/data/clean_interim.py",
        "--data",
        raw_path,
        "--output",
        interim_path,
    )
    append_privacy_event(
        run_id=run_id,
        event_type="transformation",
        stage="cleaning",
        status="completed",
        details={
            "input_path": raw_path,
            "output_path": interim_path,
            "cleaning_audit": "data/interim/cleaning_audit_log.csv",
        },
    )
    logger.info("Cleaning completed.")
    return interim_path


@task(name="validate interim data", retries=0)
def validate_interim(interim_path: str, run_id: str) -> str:
    logger = get_run_logger()
    logger.info("Running interim-data validation on %s", interim_path)
    _run_python_script(
        "src/data_quality/gx_interim_validation.py",
        "--data",
        interim_path,
    )
    append_privacy_event(
        run_id=run_id,
        event_type="validation",
        stage="interim_validation",
        status="passed",
        details={
            "dataset_path": interim_path,
            "report": "reports/tables/gx_interim_validation.json",
            "blocking": True,
        },
    )
    logger.info("Interim validation passed.")
    return interim_path


@task(name="build model-ready dataset", retries=0)
def build_model_ready(interim_path: str, processed_path: str, run_id: str) -> str:
    logger = get_run_logger()
    logger.info("Building model-ready dataset %s", processed_path)
    _run_python_script(
        "src/features/build_model_dataset.py",
        "--data",
        interim_path,
        "--output",
        processed_path,
    )
    append_privacy_event(
        run_id=run_id,
        event_type="identifier_minimisation",
        stage="model_ready_transformation",
        status="completed",
        details={
            "input_path": interim_path,
            "output_path": processed_path,
            "identifier_removed": "msisdn",
            "source_label_removed": True,
            "identifier_retained_in_output": False,
        },
    )
    logger.info("Model-ready dataset created.")
    return processed_path


@task(name="validate processed data", retries=0)
def validate_processed(processed_path: str, run_id: str) -> str:
    logger = get_run_logger()
    logger.info("Running processed-data validation on %s", processed_path)
    _run_python_script(
        "src/data_quality/gx_processed_validation.py",
        "--data",
        processed_path,
    )
    append_privacy_event(
        run_id=run_id,
        event_type="validation",
        stage="processed_validation",
        status="passed",
        details={
            "dataset_path": processed_path,
            "report": "reports/tables/gx_processed_validation.json",
            "blocking": True,
        },
    )
    logger.info("Processed-data validation passed.")
    return processed_path


@flow(name="telecom-delinquency-etl", log_prints=True)
def telecom_delinquency_etl(
    raw_path: str = str(DEFAULT_RAW),
    interim_path: str = str(DEFAULT_INTERIM),
    processed_path: str = str(DEFAULT_PROCESSED),
) -> str:
    """Run the leakage-conscious, validated ETL path from raw to model-ready data."""
    run_id = datetime.now(timezone.utc).strftime("pipeline_%Y%m%dT%H%M%S_%fZ")
    append_privacy_event(
        run_id=run_id,
        event_type="pipeline_run",
        stage="pipeline",
        status="started",
        details={"orchestrator": "Prefect"},
    )

    try:
        raw_checked = validate_raw(raw_path, run_id)
        interim = clean_raw(raw_checked, interim_path, run_id)
        interim_checked = validate_interim(interim, run_id)
        processed = build_model_ready(interim_checked, processed_path, run_id)
        processed_checked = validate_processed(processed, run_id)
        append_privacy_event(
            run_id=run_id,
            event_type="pipeline_run",
            stage="pipeline",
            status="completed",
            details={"processed_output": processed_checked},
        )
        return processed_checked
    except Exception as exc:
        append_privacy_event(
            run_id=run_id,
            event_type="pipeline_run",
            stage="pipeline",
            status="failed",
            details={"error_type": type(exc).__name__},
        )
        raise


if __name__ == "__main__":
    telecom_delinquency_etl()
