"""Push named events to GitHub at most every fifteen minutes."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("ventoux.publish")


def publish(repo: Path) -> bool:
    if not (repo / ".git").is_dir():
        log.info("Pas de dépôt git, publication ignorée")
        return False
    paths = ["data/events.json", "data/thumbs", "data/learning.json", "data/view.json", "data/view.jpg"]
    try:
        _git(repo, "pull", "--rebase", "origin", "main")
    except subprocess.CalledProcessError:
        log.warning("Historique distant non rapatrié")
    status = _git(repo, "status", "--porcelain", "--", *paths)
    if not status.strip():
        return False
    _git(repo, "add", "--", *paths)
    staged = _git(repo, "diff", "--cached", "--name-only")
    blocked = [line for line in staged.splitlines() if line.startswith("secrets/") or line.endswith(".json") and "drive" in line]
    if blocked:
        log.error("Refus de publier des secrets: %s", blocked)
        _git(repo, "reset", "HEAD")
        return False
    _git(repo, "commit", "-m", "Ajoute les événements nommés de la webcam")
    _git(repo, "push", "origin", "HEAD")
    log.info("Historique publié")
    return True


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout
