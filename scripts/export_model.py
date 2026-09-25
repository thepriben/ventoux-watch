"""Export YOLO11 nano to ONNX. Run this on the Mac, not on the Pi."""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    exported = YOLO("yolo11n.pt").export(format="onnx", imgsz=640, simplify=True)
    source = Path(exported)
    target = DEST / "yolo11n.onnx"
    target.write_bytes(source.read_bytes())
    print(target)


if __name__ == "__main__":
    main()
