"""Download dos pesos oficiais do YOLOv8 (release da Ultralytics no GitHub) para models/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from src.paths import MODELS_DIR

RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
# Tamanho mínimo esperado de cada peso: menor que isso é download incompleto
KNOWN = {
    "yolov8n.pt": 5_000_000,
    "yolov8s.pt": 20_000_000,
    "yolov8m.pt": 45_000_000,
    "yolov8n-cls.pt": 4_000_000,
    "yolov8s-cls.pt": 10_000_000,
}


def download(dest: str | Path, url: str | None = None, min_size: int | None = None) -> Path:
    """Baixa ``dest`` (não baixa de novo se já existe inteiro). Devolve o caminho."""
    dest = Path(dest)
    url = url or RELEASE + dest.name
    min_size = min_size or KNOWN.get(dest.name, 1_000_000)
    if dest.exists() and dest.stat().st_size >= min_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < min_size:
        tmp.unlink()
        raise RuntimeError(f"Download incompleto de {url}")
    tmp.replace(dest)
    return dest


def resolve(name_or_path: str | Path) -> Path:
    """Um caminho, ou o nome de um peso em models/ (``yolov8s.pt``). Pesos oficiais são baixados se faltarem."""
    path = Path(name_or_path)
    if path.is_file():
        return path
    if path.parent == Path("."):
        path = MODELS_DIR / path.name
    if not path.is_file() and path.name in KNOWN:
        download(path)
    return path
