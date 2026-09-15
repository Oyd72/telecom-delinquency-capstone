"""Privacy-preserving pipeline audit logging utilities.

The audit log records pipeline-level access and transformation events without storing
raw personal identifiers or row-level personal data. Events are appended as JSON Lines
(JSONL) so each line is an independent structured record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_PATH = Path("reports/privacy/privacy_audit_log.jsonl")
FORBIDDEN_KEYS = {
    "msisdn",
    "original_value",
    "raw_identifier",
    "customer_id",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_no_forbidden_keys(payload: dict[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_KEYS.intersection(payload.keys()))
    if forbidden:
        raise ValueError(f"Privacy audit payload contains forbidden keys: {forbidden}")


def append_privacy_event(
    *,
    run_id: str,
    event_type: str,
    stage: str,
    status: str,
    details: dict[str, Any] | None = None,
    audit_path: Path = DEFAULT_AUDIT_PATH,
) -> dict[str, Any]:
    """Append one privacy-safe pipeline event and return the written record."""
    details = details or {}
    _assert_no_forbidden_keys(details)

    record: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "run_id": run_id,
        "event_type": event_type,
        "stage": stage,
        "status": status,
        "details": details,
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
