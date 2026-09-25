"""Apply a validation issue to the event history. Used by GitHub Actions."""

import os
import sys
from pathlib import Path

from watcher.review import apply_files, parse_review

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parsed = parse_review(os.environ.get("ISSUE_BODY", ""), os.environ.get("LABEL", ""))
    if parsed is None:
        print("missing")
        return 0
    event_id, verdict = parsed
    if apply_files(ROOT, event_id, verdict):
        print(f"updated {event_id} {verdict}")
        return 0
    print("unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
