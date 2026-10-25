"""Estágio 2: especialista em textura. Olha o recorte de cada objeto e dá a probabilidade de cada material.

Dois especialistas possíveis, escolhidos por ``load_texture_expert``:

- **Hugging Face** (``WASTE_HF_MODEL``, ex.: um ViT ajustado no TrashNet): carregado com
  ``AutoModelForImageClassification``. Os rótulos do modelo (plastic, glass, cardboard…) viram categorias.
- **Local** (``models/material_cls.pt``, ou o re-treinado com as suas correções): o YOLOv8-cls treinado no
  TrashNet que vem no repositório. É o fallback automático se o Hugging Face estiver fora do ar, sem internet,
  ou se o modelo pedido não existir.

``ensemble`` junta o voto de forma (detector) com o de textura: 40% forma, 60% textura.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from src import paths
from src.recycling import CATEGORIES, category_from_name

log = logging.getLogger(__name__)

CATEGORY_ORDER = list(CATEGORIES)
# Sugestão de modelo no Hugging Face. Não foi possível verificar se ele existe (o Hugging Face é inacessível no
# ambiente onde isto foi escrito): se não existir, o especialista local assume sozinho.
DEFAULT_HF_MODEL = "watersplash/waste-classification"
SHAPE_WEIGHT, TEXTURE_WEIGHT = 0.4, 0.6


def to_categories(class_probs: dict[str, float]) -> dict[str, float]:
    """Probabilidades por classe do modelo (plastic, cardboard, paper…) → por categoria da coleta (somadas)."""
    out = {c: 0.0 for c in CATEGORY_ORDER}
    for label, p in class_probs.items():
        out[category_from_name(label).name] += float(p)
    total = sum(out.values()) or 1.0
    return {c: v / total for c, v in out.items()}


def shape_vote(category: str | None, confidence: float) -> dict[str, float]:
    """O voto do detector: a confiança vai para a categoria dele, o resto se divide entre as outras."""
    if category is None:
        return {c: 1 / len(CATEGORY_ORDER) for c in CATEGORY_ORDER}
    rest = (1 - confidence) / (len(CATEGORY_ORDER) - 1)
    return {c: confidence if c == category else rest for c in CATEGORY_ORDER}


def ensemble(shape: dict[str, float], texture: dict[str, float] | None, w_shape: float = SHAPE_WEIGHT,
             w_texture: float = TEXTURE_WEIGHT) -> dict[str, float]:
    if texture is None:
        return dict(shape)
    mixed = {c: w_shape * shape.get(c, 0) + w_texture * texture.get(c, 0) for c in CATEGORY_ORDER}
    total = sum(mixed.values()) or 1.0
    return {c: v / total for c, v in mixed.items()}


def top(probs: dict[str, float]) -> tuple[str, float]:
    category = max(probs, key=probs.get)
    return category, probs[category]


@dataclass
class TexturePrediction:
    label: str  # classe mais provável do modelo (plastic, glass…)
    confidence: float
    probs: dict[str, float]  # por categoria da coleta


class LocalTextureExpert:
    """YOLOv8-cls treinado no TrashNet (models/material_cls.pt)."""

    name = "local"

    def __init__(self, model_path=None, device: str = "cpu", imgsz: int = 224):
        from ultralytics import YOLO

        self.model_path = model_path or paths.active_material_model()
        self.model = YOLO(str(self.model_path), task="classify")
        self.device, self.imgsz = device, imgsz

    def predict(self, crops: list[np.ndarray]) -> list[TexturePrediction]:
        crops = [c for c in crops if c is not None and c.size]
        if not crops:
            return []
        out = []
        for r in self.model.predict(crops, imgsz=self.imgsz, device=self.device, verbose=False):
            probs = r.probs.data.tolist()
            by_class = {r.names[i]: p for i, p in enumerate(probs)}
            label = r.names[int(r.probs.top1)]
            out.append(TexturePrediction(label, float(r.probs.top1conf), to_categories(by_class)))
        return out


class HFTextureExpert:
    """Classificador de imagens do Hugging Face (ViT, ConvNeXt…) ajustado em resíduos."""

    name = "huggingface"

    def __init__(self, model_id: str, device: str = "cpu", model=None, processor=None):
        import torch  # noqa: F401  (o transformers precisa dele)
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.model_id = model_id
        self.processor = processor or AutoImageProcessor.from_pretrained(model_id)
        self.model = (model or AutoModelForImageClassification.from_pretrained(model_id)).eval().to(device)
        self.device = device
        labels = list(self.model.config.id2label.values())
        if not any(category_from_name(label).name != "Orgânico/Rejeito" for label in labels):
            raise ValueError(f"Os rótulos de {model_id} não parecem materiais: {labels[:6]}")

    def predict(self, crops: list[np.ndarray]) -> list[TexturePrediction]:
        import torch

        crops = [c for c in crops if c is not None and c.size]
        if not crops:
            return []
        rgb = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops]  # o OpenCV dá BGR
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = self.model(**inputs).logits.softmax(-1).cpu().numpy()
        id2label = self.model.config.id2label
        out = []
        for row in probs:
            by_class = {id2label[i]: float(p) for i, p in enumerate(row)}
            best = int(row.argmax())
            out.append(TexturePrediction(id2label[best], float(row[best]), to_categories(by_class)))
        return out


def load_texture_expert(device: str = "cpu"):
    """Hugging Face se configurado e acessível; senão o modelo local. Devolve None se nenhum existir."""
    model_id = os.environ.get("WASTE_HF_MODEL", DEFAULT_HF_MODEL).strip()
    if model_id and model_id.lower() not in ("off", "0", "none"):
        try:
            return HFTextureExpert(model_id, device)
        except Exception as err:  # sem internet, modelo inexistente, rótulos estranhos…
            log.warning("Especialista do Hugging Face (%s) indisponível, usando o local: %s", model_id, err)
    if paths.active_material_model().is_file():
        return LocalTextureExpert(device=device)
    return None
