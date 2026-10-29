"""Guarda as imagens em que o sistema ficou em dúvida, já com rótulos sugeridos, para revisão humana.

Cada caso vira três arquivos em ``storage/review/`` (ou ``WASTE_REVIEW_DIR``):

- ``<id>.jpg``: a imagem original;
- ``<id>.txt``: as caixas no formato YOLO, com as 5 categorias (``classes.txt``), prontas para corrigir no
  Roboflow, CVAT ou Label Studio (importe a pasta como "YOLO");
- ``<id>.json``: o motivo e o que o sistema detectou.

Corrigidos, esses arquivos viram dados de treino (``python -m src.train yaml``): é o ciclo que mais melhora o
modelo, porque ensina justamente o que ele erra, com a câmera e a luz de verdade.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src import paths
from src.paths import REVIEW_DIR
from src.recycling import CATEGORIES

CATEGORY_NAMES = list(CATEGORIES)


def review_reasons(detections, low_conf: float = 0.5, single_item: bool = False) -> list[str]:
    """Por que estas detecções merecem revisão (lista vazia: nenhum motivo)."""
    waste = [d for d in detections if d.category]
    reasons = []
    if single_item and not waste:
        reasons.append("nenhum resíduo reconhecido")
    for d in waste:
        if d.confidence < low_conf:
            reasons.append(f"baixa confiança: {d.label} {d.confidence:.0%}")
        if d.material and d.material_confidence is not None and d.material_confidence < 0.6:
            reasons.append(f"material incerto: {d.material} {d.material_confidence:.0%}")
    if len({d.category for d in waste}) > 1 and single_item:
        reasons.append("mais de uma categoria num item só")
    return reasons


class ReviewCollector:
    def __init__(
        self,
        directory: str | Path = REVIEW_DIR,
        low_conf: float = 0.5,
        max_items: int = 1000,
        min_interval_s: float = 2.0,
        clock=time.monotonic,
    ):
        """``max_items`` limita o disco; ``min_interval_s`` evita gravar todos os frames de um vídeo."""
        self.directory = Path(directory)
        self.low_conf = low_conf
        self.max_items = max_items
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, directory: str | Path | None = None) -> "ReviewCollector | None":
        """``WASTE_REVIEW=0`` desliga a coleta; ``WASTE_REVIEW_CONF`` é a confiança abaixo da qual guardar.

        No site público, vem desligada: seriam fotos de visitantes guardadas no servidor (e baixáveis por outros)."""
        if os.environ.get("WASTE_REVIEW", "0" if paths.PUBLIC else "1") == "0":
            return None
        return cls(directory or REVIEW_DIR, float(os.environ.get("WASTE_REVIEW_CONF", "0.5")))

    def count(self) -> int:
        return len(list(self.directory.glob("*.jpg"))) if self.directory.is_dir() else 0

    def maybe_save(self, image: np.ndarray, detections, source: str = "api", single_item: bool = False) -> Path | None:
        """Grava o caso se houver motivo. Devolve o caminho da imagem gravada, ou None."""
        reasons = review_reasons(detections, self.low_conf, single_item)
        if not reasons:
            return None
        with self._lock:
            now = self._clock()
            if self._last is not None and now - self._last < self.min_interval_s:
                return None
            if self.count() >= self.max_items:
                return None
            self._last = now
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "classes.txt").write_text("\n".join(CATEGORY_NAMES) + "\n", encoding="utf-8")

            stamp = datetime.now(timezone.utc)
            case_id = f"{stamp:%Y%m%d-%H%M%S-%f}"
            h, w = image.shape[:2]
            cv2.imwrite(str(self.directory / f"{case_id}.jpg"), image)
            lines = []
            for d in detections:
                if not d.category:
                    continue
                x1, y1, x2, y2 = d.box
                lines.append(
                    f"{CATEGORY_NAMES.index(d.category)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f}"
                    f" {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                )
            (self.directory / f"{case_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            (self.directory / f"{case_id}.json").write_text(
                json.dumps(
                    {
                        "ts": stamp.isoformat(timespec="seconds"),
                        "source": source,
                        "reasons": reasons,
                        "detections": [d.to_dict() for d in detections],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return self.directory / f"{case_id}.jpg"

    def export_zip(self) -> bytes:
        """Todos os casos num zip (imagens, rótulos YOLO e classes.txt) para subir no Roboflow."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            if self.directory.is_dir():
                for path in sorted(self.directory.iterdir()):
                    if path.suffix in (".jpg", ".txt", ".json"):
                        archive.write(path, path.name)
        return buf.getvalue()
