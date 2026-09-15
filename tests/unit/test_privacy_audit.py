import json
from pathlib import Path

import pytest

from src.privacy.privacy_audit import append_privacy_event


def test_append_privacy_event_writes_structured_jsonl_without_identifier_values(tmp_path: Path):
    audit_path = tmp_path / "privacy_audit.jsonl"

    record = append_privacy_event(
        run_id="pipeline_test",
        event_type="identifier_minimisation",
        stage="model_ready_transformation",
        status="completed",
        details={
            "identifier_removed": "msisdn",
            "identifier_retained_in_output": False,
            "output_path": "data/processed/example.csv",
        },
        audit_path=audit_path,
    )

    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    saved = json.loads(lines[0])
    assert saved["run_id"] == "pipeline_test"
    assert saved["event_type"] == "identifier_minimisation"
    assert saved["details"]["identifier_removed"] == "msisdn"
    assert saved["details"]["identifier_retained_in_output"] is False
    assert "timestamp_utc" in saved
    assert record == saved


def test_append_privacy_event_rejects_forbidden_personal_data_keys(tmp_path: Path):
    audit_path = tmp_path / "privacy_audit.jsonl"

    with pytest.raises(ValueError, match="forbidden keys"):
        append_privacy_event(
            run_id="pipeline_test",
            event_type="data_access",
            stage="raw_validation",
            status="started",
            details={"msisdn": "9999999999"},
            audit_path=audit_path,
        )

    assert not audit_path.exists()
