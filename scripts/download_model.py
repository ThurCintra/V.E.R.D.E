"""Baixa pesos do YOLOv8 para models/ (não baixa de novo o que já existe).

    python scripts/download_model.py               # yolov8n.pt (detector padrão)
    python scripts/download_model.py yolov8s.pt    # detector maior, mais preciso
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DEFAULT_MODEL, MODELS_DIR  # noqa: E402
from src.weights import download as _download  # noqa: E402


def download(dest: Path = DEFAULT_MODEL) -> Path:
    return _download(dest)


if __name__ == "__main__":
    for name in sys.argv[1:] or [DEFAULT_MODEL.name]:
        print(download(MODELS_DIR / name))
