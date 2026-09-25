"""Record a human yes or no on one deduction."""

from __future__ import annotations

import json
import re
from pathlib import Path

VERDICTS = {"accepted": "valide", "rejected": "rejete"}
CLASSES = {"voiture": ("vehicle", "Voiture"), "bus": ("bus", "Bus")}


def parse_review(body: str, label: str) -> tuple[str, str, str] | None:
    match = re.search(r"^event_id:\s*(\S+)\s*$", body or "", re.MULTILINE)
    if not match:
        return None
    classe = ""
    found = re.search(r"^classe:\s*(\S+)\s*$", body or "", re.MULTILINE)
    if found:
        classe = found.group(1).strip().lower()
    if label == "valide":
        return match.group(1), "accepted", classe
    if label == "rejete":
        return match.group(1), "rejected", ""
    return None


def apply_review(events: list[dict], learning: dict, event_id: str, verdict: str, classe: str = "") -> bool:
    if verdict not in VERDICTS:
        return False
    target = next((event for event in events if event.get("id") == event_id), None)
    if target is None:
        return False
    changed = False
    if classe in CLASSES:
        kind, label = CLASSES[classe]
        detail = dict(target.get("detail") or {})
        if target.get("type") != kind or target.get("label") != label or detail.get("correction") != label:
            target["type"] = kind
            target["label"] = label
            detail["correction"] = label
            target["detail"] = detail
            changed = True
    previous = target.get("review")
    if previous != verdict:
        if previous in ("accepted", "rejected"):
            learning[previous] = max(0, int(learning.get(previous, 0)) - 1)
        target["review"] = verdict
        learning[verdict] = int(learning.get(verdict, 0)) + 1
        changed = True
    return changed


def apply_files(root: Path, event_id: str, verdict: str, classe: str = "") -> bool:
    events_path = root / "data" / "events.json"
    learning_path = root / "data" / "learning.json"
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    learning = json.loads(learning_path.read_text(encoding="utf-8")) if learning_path.is_file() else {}
    if not apply_review(payload.get("events") or [], learning, event_id, verdict, classe):
        return False
    events_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    learning_path.write_text(json.dumps(learning, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True
