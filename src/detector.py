"""Detector de resíduos com YOLOv8.

Dois modos, escolhidos automaticamente pelo modelo carregado:

- **Modelo customizado** (``models/waste_detector.pt``, gerado por ``src/train.py``): as classes já são
  materiais/resíduos e viram categoria pelo nome.
- **Heurístico COCO** (``models/yolov8s.pt``, enquanto não há modelo treinado): as classes do COCO que
  aparecem no lixo (garrafa, copo, talheres, frutas, livro…) são mapeadas para as 5 categorias da coleta
  seletiva.

No modo COCO, se existir ``models/material_cls.pt`` (classificador de material treinado no TrashNet), cada
objeto de material ambíguo (garrafa, copo, pote…) é recortado e reclassificado por ele: é o que separa garrafa
PET de vidro. Com ``single_item=True`` (lixeira que fotografa um item por vez), uma foto em que o detector não
achou nada é classificada inteira.

Configuração por ambiente (``Detector.from_env``): ``WASTE_MODEL`` (ex.: ``yolov8s.pt``, mais preciso),
``WASTE_IMGSZ`` (ex.: 960 para itens pequenos), ``DETECTOR_CONF``, ``WASTE_TTA=1`` (inferência com aumento de
dados, mais lenta), ``WASTE_MATERIAL_MODEL`` (``off`` desliga a 2ª etapa) e ``WASTE_MATERIAL_CONF``.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.material import REFINABLE_LABELS, MaterialClassifier, crop
from src.paths import CUSTOM_MODEL, DEFAULT_MODEL, MATERIAL_MODEL
from src.recycling import CATEGORIES, COCO_RULES, DEFAULT_WEIGHT_G, METAL, category_from_name
from src.weights import resolve


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels
    category: str | None = None  # None: não é resíduo (pessoa, carro…)
    weight_g: float = 0.0
    material: str | None = None  # o que o classificador de material viu no recorte (glass, plastic…)
    material_confidence: float | None = None

    @property
    def color_hex(self) -> str | None:
        return CATEGORIES[self.category].hex if self.category else None

    @property
    def command(self) -> str | None:
        return CATEGORIES[self.category].command if self.category else None

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "label": self.label,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "color": self.color_hex,
            "weight_g": self.weight_g,
            "command": self.command,
            "material": (
                {"label": self.material, "confidence": round(self.material_confidence, 4)} if self.material else None
            ),
        }


def default_model_path() -> Path:
    """O modelo treinado para resíduos, se existir; senão o YOLOv8n do COCO."""
    return CUSTOM_MODEL if CUSTOM_MODEL.is_file() else DEFAULT_MODEL


class Detector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.25,
        device: str = "cpu",
        imgsz: int = 640,
        augment: bool = False,
        material_model: str | Path | None = "auto",
        material_conf: float = 0.6,
    ):
        """``material_model``: caminho do classificador de material, ``"auto"`` (usa ``models/material_cls.pt`` se
        existir) ou ``None`` (desligado). ``material_conf``: confiança mínima para ele mudar a categoria."""
        model_path = resolve(model_path) if model_path else default_model_path()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado em {model_path}. Rode: python scripts/download_model.py"
            )
        self.model_path = model_path
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.augment = augment
        names = self.model.names
        self.is_coco = len(names) == 80 and names[0] == "person"

        if material_model == "auto":
            material_model = MATERIAL_MODEL if MATERIAL_MODEL.is_file() else None
        self.material = MaterialClassifier(material_model, device) if material_model else None
        self.material_conf = material_conf

    @classmethod
    def from_env(cls) -> "Detector":
        material = os.environ.get("WASTE_MATERIAL_MODEL", "auto")
        return cls(
            model_path=os.environ.get("WASTE_MODEL") or None,
            conf=float(os.environ.get("DETECTOR_CONF", "0.25")),
            imgsz=int(os.environ.get("WASTE_IMGSZ", "640")),
            augment=os.environ.get("WASTE_TTA", "0") == "1",
            material_model=None if material.lower() in ("off", "0", "none") else material,
            material_conf=float(os.environ.get("WASTE_MATERIAL_CONF", "0.6")),
        )

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    @property
    def mode(self) -> str:
        if not self.is_coco:
            return "customizado"
        return "coco+material" if self.material else "coco-heuristico"

    def _classify(self, label: str) -> tuple[str | None, float]:
        if not self.is_coco:
            category = category_from_name(label)
            return category.name, DEFAULT_WEIGHT_G[category.name]
        rule = COCO_RULES.get(label)
        if rule is None:
            return None, 0.0
        return rule.category.name, rule.weight_g

    def detect(
        self, image: str | Path | bytes | np.ndarray, waste_only: bool = False, single_item: bool = False
    ) -> list[Detection]:
        """Recebe caminho, bytes de arquivo ou array BGR (como o OpenCV lê) e devolve as detecções.

        ``single_item``: a foto tem um resíduo só (lixeira com câmera fixa). O classificador de material olha a
        foto inteira e, se tiver certeza, decide a categoria do item; o detector só dá a posição.
        """
        image = load_image(image)
        result = self.model.predict(
            image, conf=self.conf, imgsz=self.imgsz, augment=self.augment, device=self.device, verbose=False
        )[0]
        names = result.names
        detections = []
        for xyxy, p, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            label = names[int(c)]
            category, weight = self._classify(label)
            if waste_only and category is None:
                continue
            detections.append(Detection(label, float(p), tuple(float(v) for v in xyxy), category, weight))

        if self.material and self.is_coco:
            detections = self._refine(image, detections)
        if single_item and self.material:
            detections = self._single_item(image, detections)
        return detections

    def _refine(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Segunda etapa: reclassifica pelo recorte os objetos de material ambíguo."""
        todo = [i for i, d in enumerate(detections) if d.label in REFINABLE_LABELS]
        predictions = self.material.classify([crop(image, detections[i].box) for i in todo])
        for i, pred in zip(todo, predictions):
            det = detections[i]
            det = replace(det, material=pred.label, material_confidence=pred.confidence)
            if pred.confidence >= self.material_conf and pred.category != det.category:
                det = replace(det, category=pred.category, weight_g=DEFAULT_WEIGHT_G[pred.category])
            detections[i] = det
        return detections

    def _single_item(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Um item por foto: vira uma detecção só, com a categoria que o classificador vê na foto inteira.

        No teste do TrashNet, deixar o detector decidir errava 16% das fotos: o COCO "vê" papelão como celular e
        coisas alaranjadas como banana, e esses rótulos não passam pelo classificador. Se o classificador não
        tiver certeza, fica o que o detector achou.
        """
        (pred,) = self.material.classify([image])
        if pred.confidence < self.material_conf:
            return detections
        waste = [d for d in detections if d.category]
        others = [d for d in detections if not d.category]
        if waste:
            best = max(waste, key=lambda d: d.confidence)
            weight = best.weight_g if best.category == pred.category else DEFAULT_WEIGHT_G[pred.category]
            item = replace(
                best, category=pred.category, weight_g=weight, material=pred.label, material_confidence=pred.confidence
            )
        else:
            h, w = image.shape[:2]
            item = Detection(
                pred.label, pred.confidence, (0.0, 0.0, float(w), float(h)), pred.category,
                DEFAULT_WEIGHT_G[pred.category], pred.label, pred.confidence,
            )
        return [item] + others


def load_image(image: str | Path | bytes | np.ndarray) -> np.ndarray:
    """Caminho, bytes de um arquivo (JPEG/PNG…) ou array HxWx3 BGR → array BGR."""
    if isinstance(image, (str, Path)):
        if not Path(image).is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {image}")
        loaded = cv2.imread(str(image))
        if loaded is None:
            raise ValueError(f"Não foi possível ler a imagem: {image}")
        return loaded
    if isinstance(image, (bytes, bytearray)):
        loaded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if loaded is None:
            raise ValueError("Os bytes enviados não são uma imagem válida")
        return loaded
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError("A imagem deve ser um caminho, bytes de arquivo ou um array HxWx3")


def annotate(image: np.ndarray, detections, track_ids: list[int] | None = None) -> np.ndarray:
    """Desenha as caixas nas cores da CONAMA (cinza para o que não é resíduo). Não altera a imagem original."""
    out = image.copy()
    thickness = max(2, round(min(out.shape[:2]) / 300))
    for i, det in enumerate(detections):
        color = CATEGORIES[det.category].bgr if det.category else (150, 150, 150)
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{det.category or det.label} | {det.label} {det.confidence:.0%}"
        if track_ids is not None:
            text = f"#{track_ids[i]} {text}"
        text = _ascii(text)  # as fontes do OpenCV não têm acentos
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(th + 4, y1)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text), color, -1)
        ink = (0, 0, 0) if det.category == METAL.name else (255, 255, 255)  # preto no amarelo
        cv2.putText(out, text, (x1 + 2, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv2.LINE_AA)
    return out


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
