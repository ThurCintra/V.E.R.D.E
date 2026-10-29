"""Treina o detector de resíduos de uma classe só ("residuo") nas cenas de treino do TACO e compara o sistema com
ele e com o COCO nas cenas de validação e teste (as mesmas variações do scenes_grid.py).

    python .github/scripts/train_detector.py resultado.md

O peso fica em ``models/residuos_taco.pt``; o sistema só passa a usá-lo se ele for copiado para
``models/waste_detector.pt`` (ou apontado em ``WASTE_MODEL``).
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import scenes_grid  # noqa: E402
from src.paths import MODELS_DIR  # noqa: E402
from src.train import prepare_taco_detection, prepare_taco_scenes, train  # noqa: E402

BASE = os.environ.get("DET_BASE", "yolo11n.pt")  # na CPU do GitHub, ~5 min por época com o yolo11n
EPOCHS = int(os.environ.get("DET_EPOCHS", "40"))


def main(out: str) -> None:
    start = time.time()
    index = Path(os.environ["CENAS_JSON"]) if os.environ.get("CENAS_JSON") else prepare_taco_scenes()
    data = prepare_taco_detection(index)
    train(data, BASE, epochs=EPOCHS, imgsz=640, batch=16, device="cpu", output=MODELS_DIR / "residuos_taco.pt",
          workers=3, patience=15, plots=False)
    print(f"Treino: {(time.time() - start) / 60:.0f} min", flush=True)
    scenes_grid.main(out, ["taco", "coco"])
    text = Path(out).read_text(encoding="utf-8")
    Path(out).write_text(f"Detector: {BASE}, até {EPOCHS} épocas a 640 px ({(time.time() - start) / 60:.0f} min "
                         f"no total)\n\n{text}", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
