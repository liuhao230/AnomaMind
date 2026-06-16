from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from AnomaMind.core.utils import load_segment_data
from AnomaMind.core.workflowstate import WorkflowState


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"record is not a JSON object: {path}")
    return obj


def _validate_smoke_record(record: Dict[str, Any], record_path: Path) -> None:
    segment_dir = record.get("segment_dir")
    if not isinstance(segment_dir, str) or not segment_dir.strip():
        raise ValueError(f"missing segment_dir: {record_path}")
    if not isinstance(record.get("visual_localization"), list):
        raise ValueError(f"visual_localization must be list: {record_path}")
    if not isinstance(record.get("visual_localization_items"), list):
        raise ValueError(f"visual_localization_items must be list: {record_path}")


def restore_detector_training_sample(record_path: str | Path) -> Dict[str, Any]:
    record_path = Path(record_path).resolve()
    record = _read_json(record_path)
    _validate_smoke_record(record, record_path)

    segment_dir = Path(str(record["segment_dir"])).resolve()
    pack = load_segment_data(segment_dir)
    initial_state: WorkflowState = {
        "data": pack["segment_data"],
        "type": pack["type"],
        "path": pack["path"],
        "visual_description": str(record.get("visual_description", "") or ""),
        "visual_localization": list(record.get("visual_localization") or []),
        "visual_localization_items": list(record.get("visual_localization_items") or []),
        "actor_messages": [],
        "detector_messages": [],
        "has_error": False,
        "error_message": "",
        "final_output": {},
    }
    return {
        "record_path": str(record_path),
        "segment_dir": str(segment_dir),
        "ground_truth": list(pack.get("ground_truth") or []),
        "initial_state": initial_state,
        "record": record,
    }
