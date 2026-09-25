"""Record a human yes or no on one deduction."""

from __future__ import annotations

import json
import re
from pathlib import Path

VERDICTS = {"accepted": "valide", "rejected": "rejete"}


def parse_review(body: str, label: str) -> tuple[str, str] | None:
    match = re.search(r"^event_id:\s*(\S+)\s*$", body or "", re.MULTILINE)
    if not match:
        return None
    if label == "valide":
        return match.group(1), "accepted"
    if label == "rejete":
        return match.group(1), "rejected"
    return None


def apply_review(events: list[dict], learning: dict, event_id: str, verdict: str) -> bool:
    if verdict not in VERDICTS:
        return False
    target = next((event for event in events if event.get("id") == event_id), None)
    if target is None:
        return False
    previous = target.get("review")
    if previous == verdict:
        return False
    if previous in ("accepted", "rejected"):
        learning[previous] = max(0, int(learning.get(previous, 0)) - 1)
    target["review"] = verdict
    learning[verdict] = int(learning.get(verdict, 0)) + 1
    return True


def apply_files(root: Path, event_id: str, verdict: str) -> bool:
    events_path = root / "data" / "events.json"
    learning_path = root / "data" / "learning.json"
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    learning = json.loads(learning_path.read_text(encoding="utf-8")) if learning_path.is_file() else {}
    if not apply_review(payload.get("events") or [], learning, event_id, verdict):
        return False
    events_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    learning_path.write_text(json.dumps(learning, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True
