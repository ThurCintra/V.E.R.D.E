"""Classificador de material: a segunda etapa do detector.

O detector acha *onde* está cada objeto; este modelo olha o recorte e decide *de que material* ele é. Treinado
no TrashNet (``python -m src.train trashnet``), separa o que o COCO não separa: garrafa PET de garrafa de vidro,
copo de papel de copo plástico, caixa de papelão de pote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.recycling import category_from_name

# Objetos do COCO cujo material varia e que o classificador pode reclassificar. Comida e eletrônicos ficam de fora:
# o TrashNet não tem essas classes e o mapeamento do COCO já acerta.
REFINABLE_LABELS = {
    "bottle", "cup", "bowl", "wine glass", "vase", "fork", "knife", "spoon", "scissors", "book",
    "toothbrush", "umbrella", "frisbee", "sports ball",
}


@dataclass(frozen=True)
class MaterialPrediction:
    label: str  # classe do modelo (plastic, glass, cardboard…)
    category: str  # categoria da coleta seletiva
    confidence: float


class MaterialClassifier:
    def __init__(self, model_path: str | Path, device: str = "cpu", imgsz: int = 224):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Classificador não encontrado em {self.model_path}. Treine com: python -m src.train trashnet"
            )
        from ultralytics import YOLO  # pesado (torch): só quando o classificador é usado

        self.model = YOLO(str(self.model_path), task="classify")
        self.device = device
        self.imgsz = imgsz

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    def classify(self, images: list[np.ndarray]) -> list[MaterialPrediction]:
        """Classifica recortes BGR (em lote)."""
        images = [img for img in images if img is not None and img.size]
        if not images:
            return []
        results = self.model.predict(images, imgsz=self.imgsz, device=self.device, verbose=False)
        predictions = []
        for r in results:
            label = r.names[int(r.probs.top1)]
            predictions.append(MaterialPrediction(label, category_from_name(label).name, float(r.probs.top1conf)))
        return predictions


def crop(image: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.1) -> np.ndarray:
    """Recorte da caixa com uma margem (o classificador acerta mais vendo a borda do objeto)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
    return image[y1:y2, x1:x2]
