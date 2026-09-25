"""Save one frame from the Mont Serein stream."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://visionenvironnement.quanteec.com/contents/encodings/live/78e0f372-db6f-420e-746c-7561-6665-64-b4d7-fc979b816efed/master.m3u8"
TARGET = ROOT / "data" / "reference.jpg"

def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", URL, "-frames:v", "1", "-q:v", "3", str(TARGET)],
        check=True,
    )
    print(TARGET)

if __name__ == "__main__":
    sys.exit(main())
