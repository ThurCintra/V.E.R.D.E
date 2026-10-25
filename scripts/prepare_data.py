"""Prepara as imagens de teste em data/.

- bus.jpg: foto real de exemplo que vem dentro do pacote ultralytics (sem download).
- synthetic.jpg: imagem gerada com formas geométricas, sem objetos reais.
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR  # noqa: E402


def copy_sample(dest_dir: Path = DATA_DIR) -> Path:
    import ultralytics

    src = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    dest = dest_dir / "bus.jpg"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(src, dest)
    return dest


def make_synthetic(dest_dir: Path = DATA_DIR, size: int = 640) -> Path:
    img = np.full((size, size, 3), 235, dtype=np.uint8)
    cv2.rectangle(img, (60, 80), (260, 300), (40, 90, 200), -1)
    cv2.circle(img, (450, 200), 110, (60, 170, 60), -1)
    cv2.line(img, (0, 500), (size, 560), (30, 30, 30), 12)
    dest = dest_dir / "synthetic.jpg"
    dest_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)
    return dest


if __name__ == "__main__":
    print(copy_sample())
    print(make_synthetic())
