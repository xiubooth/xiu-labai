from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class EvidenceLedger:
    def __init__(self, project_root: Path, task_run_id: str) -> None:
        self.project_root = project_root.resolve()
        self.task_run_id = task_run_id
        self.path = self.project_root / ".labai" / "evidence" / f"{task_run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record_type: str, payload: Any) -> None:
        rendered = {
            "task_run_id": self.task_run_id,
            "record_type": record_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": _normalize_payload(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rendered, ensure_ascii=False) + "\n")


def _normalize_payload(payload: Any) -> Any:
    if hasattr(payload, "to_record"):
        return payload.to_record()
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return {key: _normalize_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_normalize_payload(item) for item in payload]
    return payload
