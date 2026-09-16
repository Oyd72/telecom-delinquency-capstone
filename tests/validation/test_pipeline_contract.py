"""Repository-level validation tests for the implemented ETL contract.

These tests deliberately avoid loading the customer-level dataset. They verify that the
orchestrated pipeline points to real stage scripts and that its default asset paths keep
the raw/interim/processed separation documented by the project.
"""

from pathlib import Path

from src.pipeline import prefect_etl


def test_prefect_pipeline_stage_scripts_exist() -> None:
    expected_scripts = [
        "src/data_quality/gx_raw_validation.py",
        "src/data/clean_interim.py",
        "src/data_quality/gx_interim_validation.py",
        "src/features/build_model_dataset.py",
        "src/data_quality/gx_processed_validation.py",
        "src/monitoring/representation_bias_checks.py",
    ]

    missing = [
        path
        for path in expected_scripts
        if not (prefect_etl.PROJECT_ROOT / path).is_file()
    ]

    assert missing == []


def test_default_pipeline_paths_preserve_data_lifecycle() -> None:
    assert prefect_etl.DEFAULT_RAW == Path("data/raw/sample_data_intw.csv")
    assert prefect_etl.DEFAULT_INTERIM == Path("data/interim/sample_data_intw_cleaned.csv")
    assert prefect_etl.DEFAULT_PROCESSED == Path(
        "data/processed/telecom_delinquency_model_ready.csv"
    )


def test_raw_validation_report_is_kept_outside_customer_data_folders() -> None:
    relative = prefect_etl.RAW_VALIDATION_REPORT.relative_to(prefect_etl.PROJECT_ROOT)
    assert relative == Path("reports/tables/gx_raw_validation.json")
