"""Detector de objetos com YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.paths import DEFAULT_MODEL


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels


class Detector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, conf: float = 0.25, device: str = "cpu"):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado em {model_path}. Rode: python scripts/download_model.py"
            )
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.device = device

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    def detect(self, image: str | Path | np.ndarray) -> list[Detection]:
        """Recebe caminho de imagem ou array BGR (como o OpenCV lê) e devolve as detecções."""
        if isinstance(image, (str, Path)):
            if not Path(image).is_file():
                raise FileNotFoundError(f"Imagem não encontrada: {image}")
            image = str(image)
        elif not (isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3):
            raise ValueError("A imagem deve ser um caminho ou um array HxWx3")

        result = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
        names = result.names
        return [
            Detection(names[int(c)], float(p), tuple(float(v) for v in xyxy))
            for xyxy, p, c in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
            )
        ]
