"""Prepara as imagens de teste em data/.

- bus.jpg: foto que vem dentro do pacote ultralytics (pessoas e ônibus, nenhum resíduo).
- mesa_garrafas_banana.jpg e mesa_vinho.jpg: fotos do COCO (coco128, licença CC BY 4.0) com garrafas, copo,
  taça e bananas, baixadas do release oficial da Ultralytics no GitHub.
- synthetic.jpg: imagem gerada com formas geométricas, sem objetos reais.
- esteira.mp4: vídeo sintético de uma foto deslizando pela tela, para testar o rastreamento.
"""

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR  # noqa: E402

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_SAMPLES = {
    "mesa_garrafas_banana.jpg": "coco128/images/train2017/000000000142.jpg",
    "mesa_vinho.jpg": "coco128/images/train2017/000000000283.jpg",
}


def copy_sample(dest_dir: Path = DATA_DIR) -> Path:
    import ultralytics

    src = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    dest = dest_dir / "bus.jpg"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(src, dest)
    return dest


def fetch_coco_samples(dest_dir: Path = DATA_DIR) -> list[Path]:
    dests = [dest_dir / name for name in COCO_SAMPLES]
    if all(d.exists() for d in dests):
        return dests
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(COCO128_URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    for name, member in COCO_SAMPLES.items():
        (dest_dir / name).write_bytes(archive.read(member))
    return dests


def make_synthetic(dest_dir: Path = DATA_DIR, size: int = 640) -> Path:
    img = np.full((size, size, 3), 235, dtype=np.uint8)
    cv2.rectangle(img, (60, 80), (260, 300), (40, 90, 200), -1)
    cv2.circle(img, (450, 200), 110, (60, 170, 60), -1)
    cv2.line(img, (0, 500), (size, 560), (30, 30, 30), 12)
    dest = dest_dir / "synthetic.jpg"
    dest_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)
    return dest


def make_conveyor_video(photo: Path, dest_dir: Path = DATA_DIR, frames: int = 30) -> Path:
    """A foto atravessa a tela da esquerda para a direita, como numa esteira (mesmos objetos em vários frames)."""
    dest = dest_dir / "esteira.mp4"
    if dest.exists():
        return dest
    img = cv2.imread(str(photo))
    h, w = img.shape[:2]
    canvas_w = w + 200
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), 10, (canvas_w, h))
    for i in range(frames):
        frame = np.full((h, canvas_w, 3), 60, dtype=np.uint8)
        x = round(i * 200 / (frames - 1))
        frame[:, x : x + w] = img
        writer.write(frame)
    writer.release()
    return dest


if __name__ == "__main__":
    print(copy_sample())
    for path in fetch_coco_samples():
        print(path)
    print(make_synthetic())
    print(make_conveyor_video(DATA_DIR / "mesa_garrafas_banana.jpg"))
