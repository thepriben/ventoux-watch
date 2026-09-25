"""Optional short clips on Google Drive. The key never enters the repository."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("ventoux.drive")


class DriveUploader:
    def __init__(self, credentials: str, folder_id: str):
        self.credentials = Path(credentials)
        self.folder_id = folder_id
        self._service = None

    @property
    def enabled(self) -> bool:
        return self.credentials.is_file() and bool(self.folder_id)

    def upload(self, path: Path, name: str) -> str:
        if not self.enabled:
            return ""
        service = self._client()
        if service is None:
            return ""
        from googleapiclient.http import MediaFileUpload

        body = {"name": name, "parents": [self.folder_id]}
        media = MediaFileUpload(str(path), mimetype="video/mp4", resumable=False)
        created = service.files().create(body=body, media_body=media, fields="id, webViewLink").execute()
        file_id = created["id"]
        service.permissions().create(fileId=file_id, body={"type": "anyone", "role": "reader"}).execute()
        return created.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"

    def _client(self):
        if self._service is not None:
            return self._service
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            log.warning("Paquet Google absent, extraits non envoyés")
            return None
        creds = service_account.Credentials.from_service_account_file(
            str(self.credentials),
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service
