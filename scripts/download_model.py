"""Baixa o modelo base yolov8n.pt para models/ (não faz nada se já existir)."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DEFAULT_MODEL  # noqa: E402

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
MIN_SIZE = 5_000_000  # ~6,2 MB; menor que isso é download incompleto


def download(dest: Path = DEFAULT_MODEL, url: str = URL) -> Path:
    if dest.exists() and dest.stat().st_size >= MIN_SIZE:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < MIN_SIZE:
        tmp.unlink()
        raise RuntimeError(f"Download incompleto de {url}")
    tmp.replace(dest)
    return dest


if __name__ == "__main__":
    print(download())
