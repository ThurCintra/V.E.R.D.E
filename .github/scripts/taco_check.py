"""Confere se as fotos do TACO baixadas batem com as anotações (orientação, tamanho), na foto original e na
miniatura de 640 px. Resultado em notices do GitHub Actions.

    python .github/scripts/taco_check.py [quantas fotos]
"""

import json
import random
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluate import iou  # noqa: E402
from src.train import TACO_ANNOTATIONS_URL, fetch_url  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
out = Path("taco_check")
out.mkdir(exist_ok=True)
coco = json.loads(urllib.request.urlopen(TACO_ANNOTATIONS_URL).read())
images = random.Random(0).sample(coco["images"], N)
boxes = {}
for ann in coco["annotations"]:
    boxes.setdefault(ann["image_id"], []).append(ann["bbox"])


def fetch(img):
    got = {}
    for kind, key in (("full", "flickr_url"), ("thumb", "flickr_640_url")):
        dest = out / f"{img['id']}_{kind}.jpg"
        if not img.get(key):
            got[kind] = "sem endereço"
        elif dest.is_file() or fetch_url(img[key], dest, tries=4, wait_s=3):
            got[kind] = dest
        else:
            got[kind] = "recusado mesmo esperando"
    return img, got


start = time.time()
with ThreadPoolExecutor(2) as pool:
    fetched = list(pool.map(fetch, images))
print(f"::notice title=Download::{N} fotos em {time.time() - start:.0f} s, 2 de cada vez")

from ultralytics import YOLO  # noqa: E402

model = YOLO(str(ROOT / "models" / "yolov8s.pt") if (ROOT / "models" / "yolov8s.pt").is_file() else "yolov8s.pt")


def aligned(pixels, img) -> float | None:
    """Fração das caixas grandes (> 2% da foto) que alguma detecção do COCO cobre com IoU >= 0,5."""
    h, w = pixels.shape[:2]
    sx, sy = w / img["width"], h / img["height"]
    big = [(x * sx, y * sy, (x + bw) * sx, (y + bh) * sy) for x, y, bw, bh in boxes.get(img["id"], [])
           if bw * bh > 0.02 * img["width"] * img["height"]]
    if not big:
        return None
    dets = model.predict(pixels, conf=0.05, verbose=False)[0].boxes.xyxy.tolist()
    return float(np.mean([max([iou(b, d) for d in dets] or [0]) >= 0.5 for b in big]))


stats = {k: Counter() for k in ("full", "thumb")}
align = {k: [] for k in ("full", "thumb", "full_rot180")}
for img, got in fetched:
    for kind in ("full", "thumb"):
        path = got[kind]
        if isinstance(path, str):
            stats[kind]["erro no download"] += 1
            stats[kind]["  " + path[:40]] += 1
            continue
        with Image.open(path) as pil:
            orientation = pil.getexif().get(0x0112, 1)
            raw = pil.size  # (w, h) sem aplicar o EXIF
        pixels = cv2.imread(str(path))
        if pixels is None:
            stats[kind]["ilegível"] += 1
            continue
        stats[kind][f"EXIF orientação {orientation}"] += 1
        h, w = pixels.shape[:2]
        ann_ratio, ratio = img["width"] / img["height"], w / h
        if abs(ann_ratio - ratio) < 0.02:
            stats[kind]["proporção igual à anotação"] += 1
            a = aligned(pixels, img)
            if a is not None:
                align[kind].append(a)
                if kind == "full":
                    a180 = aligned(cv2.rotate(pixels, cv2.ROTATE_180), img)
                    align["full_rot180"].append(a180)
        elif abs(ann_ratio - 1 / ratio) < 0.02:
            stats[kind]["proporção trocada (girada 90°)"] += 1
        else:
            stats[kind]["proporção diferente"] += 1
        if kind == "full" and (w, h) != (img["width"], img["height"]):
            stats[kind][f"tamanho diferente (arquivo {raw}, lido {w}x{h}, anotação {img['width']}x{img['height']})"[:90]] += 1

for kind, counter in stats.items():
    print(f"::notice title=TACO {kind}::" + "; ".join(f"{k}: {v}" for k, v in counter.most_common(12)))
for kind, values in align.items():
    if values:
        print(f"::notice title=Alinhamento {kind}::{np.mean(values):.1%} das caixas grandes achadas pelo COCO "
              f"(IoU >= 0,5), em {len(values)} fotos")
