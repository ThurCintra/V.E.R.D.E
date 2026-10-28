"""Motor de inferência híbrido de 3 estágios para resíduos recicláveis.

1. **Detecção espacial** (rápida): YOLO-World com vocabulário de resíduos (``src/open_vocab.py``), com
   pré-processamento CLAHE + filtro bilateral (``src/preprocess.py``), filtro de tamanho (ignora o que ocupa mais
   de 70% da imagem, como paredes e o próprio corpo) e descarte de pessoas. Se o vocabulário não puder ser montado
   (sem internet na primeira vez), o detector cai para o YOLOE e, por fim, para o YOLOv8 do COCO.
2. **Especialista de textura** (``src/material_classifier.py``): classifica o recorte de cada objeto; o resultado
   é combinado com o detector (40% forma, 60% textura).
3. **Estabilização temporal** (vídeo, ``track``): o ByteTrack (pacote ``trackers``, sobre o ``supervision``) dá um ``tracker_id`` a cada objeto, a votação por
   EMA dos últimos 12 quadros impede a categoria de piscar e o Deep Scan (``src/deep_scan.py``) dá a palavra
   final em objetos duvidosos parados no centro (ou em todos, se pedido).

Modos de operação: ``rapido`` (só o estágio 1), ``alta`` (1 + 2) e ``ultra`` (1 + 2 + Deep Scan).

Configuração por ambiente (``Detector.from_env``): ``WASTE_MODE``, ``WASTE_DETECTOR`` (``yolo-world``, ``yoloe``
ou ``coco``), ``WASTE_MODEL`` (um peso específico), ``WASTE_IMGSZ``, ``DETECTOR_CONF``, ``WASTE_TTA=1``,
``WASTE_PREPROCESS=1`` (liga o CLAHE), ``WASTE_MATERIAL_MODEL`` (``off`` desliga o estágio 2), ``WASTE_HF_MODEL`` e
``WASTE_MATERIAL_CONF``.
"""

from __future__ import annotations

import logging
import os
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.material import REFINABLE_LABELS, crop
from src.material_classifier import (
    SHAPE_WEIGHT,
    TEXTURE_WEIGHT,
    LocalTextureExpert,
    ensemble,
    load_texture_expert,
    shape_vote,
    top,
)
from src.open_vocab import IGNORED_VOCABULARY, VOCAB_CATEGORY, is_open_vocab, load_open_vocab
from src import paths
from src.paths import CUSTOM_MODEL, DEFAULT_MODEL
from src.preprocess import SceneCalibration, enhance
from src.recycling import CATEGORIES, COCO_RULES, DEFAULT_WEIGHT_G, METAL, category_from_name
from src.temporal import TemporalVoter
from src.weights import resolve

log = logging.getLogger(__name__)

MODES = {"rapido": "Rápido", "alta": "Alta precisão", "ultra": "Ultra precisão"}
# Detector de uma classe só (treinado no TACO: acha o lixo, e a textura decide o material). Antes da textura, o voto
# do detector é a frequência de cada categoria entre os resíduos do TACO.
AGNOSTIC_LABEL = "residuo"
AGNOSTIC_PRIOR = {"Plástico": 0.528, "Papel": 0.107, "Metal": 0.129, "Vidro": 0.06, "Orgânico/Rejeito": 0.177}
DEEP_SCAN_RANGE = (0.35, 0.60)  # confiança "duvidosa" que dispara o Deep Scan automático
BACKGROUND_MIN_CHANGE = 0.15  # com o fundo calibrado, caixas com menos de 15% de mudança são o próprio fundo


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels
    category: str | None = None  # None: não é resíduo (pessoa, carro…)
    weight_g: float = 0.0
    material: str | None = None  # o que o especialista de textura viu no recorte (glass, plastic…)
    material_confidence: float | None = None
    probs: dict | None = None  # probabilidade de cada categoria (após o ensemble/EMA)
    tracker_id: int | None = None
    scan: object | None = None  # ScanResult do Deep Scan, se houve

    @property
    def color_hex(self) -> str | None:
        return CATEGORIES[self.category].hex if self.category else None

    def top_probs(self, n: int = 3) -> list[tuple[str, float]]:
        if not self.probs:
            return [(self.category, 1.0)] if self.category else []
        return sorted(self.probs.items(), key=lambda kv: -kv[1])[:n]

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "label": self.label,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "color": self.color_hex,
            "weight_g": self.weight_g,
            "material": (
                {"label": self.material, "confidence": round(self.material_confidence, 4)} if self.material else None
            ),
            "probs": {c: round(p, 4) for c, p in self.probs.items()} if self.probs else None,
            "tracker_id": self.tracker_id,
            "deep_scan": (
                {"category": self.scan.category, "material": self.scan.material,
                 "justification": self.scan.justification, "source": self.scan.source}
                if self.scan else None
            ),
        }


def _containment(a, b) -> float:
    """Quanto da menor das duas caixas está dentro da outra (1: uma está toda dentro da outra)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def merge_duplicates(detections: list[Detection], overlap: float) -> list[Detection]:
    """Uma caixa da mesma categoria quase toda dentro de outra é o mesmo resíduo contado duas vezes (a banana e o
    sanduíche dela, a garrafa e o rótulo): fica a mais confiante."""
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda d: -d.confidence):
        if d.category and any(k.category == d.category and _containment(d.box, k.box) >= overlap for k in kept):
            continue
        kept.append(d)
    return [d for d in detections if any(d is k for k in kept)]


def best_waste(detections) -> Detection | None:
    """O resíduo detectado com maior confiança: a lixeira que vale para a foto inteira."""
    waste = [d for d in detections if d.category]
    return max(waste, key=lambda d: d.confidence) if waste else None


def default_model_path() -> Path:
    """O detector treinado para resíduos, se existir; senão o YOLOv8 do COCO."""
    return CUSTOM_MODEL if CUSTOM_MODEL.is_file() else DEFAULT_MODEL


def _load_detector(model_path, kind: str) -> tuple[YOLO, Path, str | None]:
    """(modelo, caminho, aviso). Sem peso explícito: vocabulário aberto pedido → o outro → treinado/COCO."""
    if model_path:
        path = resolve(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Modelo não encontrado em {path}. Rode: python scripts/download_model.py")
        return YOLO(str(path)), path, None
    if CUSTOM_MODEL.is_file():
        return YOLO(str(CUSTOM_MODEL)), CUSTOM_MODEL, None
    errors = []
    candidates = [] if kind == "coco" else list(dict.fromkeys(k for k in (kind, "yolo-world", "yoloe")
                                                              if k in ("yolo-world", "yoloe")))
    for candidate in candidates:
        try:
            model, path = load_open_vocab(candidate)
            note = None if candidate == kind else f"{kind} indisponível ({errors[-1]}); usando {candidate}"
            return model, path, note
        except Exception as err:  # sem internet na primeira vez, codificador de texto bloqueado…
            errors.append(f"{candidate}: {type(err).__name__}")
            log.warning("Vocabulário aberto %s indisponível: %s", candidate, err)
    path = resolve(DEFAULT_MODEL)
    if not path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado em {path}. Rode: python scripts/download_model.py")
    note = None if kind == "coco" else f"vocabulário aberto indisponível ({'; '.join(errors)}); usando o COCO"
    return YOLO(str(path)), path, note


class Detector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.25,
        device: str = "cpu",
        imgsz: int = 640,
        augment: bool = False,
        material_model: str | Path | None = "auto",
        material_conf: float = 0.5,
        mode: str = "alta",
        detector_kind: str = "yolo-world",
        preprocess: bool = False,
        min_area: float = 0.0005,
        max_area: float = 0.70,
        deep_scanner="auto",
    ):
        """``material_model``: ``"auto"`` (Hugging Face se configurado, senão o classificador local), um caminho
        de classificador local, ou ``None`` (sem estágio 2). ``material_conf``: certeza mínima na categoria para,
        no modo de 1 item por foto (ou no plano B), a foto inteira decidir. ``min_area``/``max_area``: fração da imagem que uma caixa pode
        ocupar."""
        self.model, self.model_path, self.fallback_note = _load_detector(model_path, detector_kind)
        names = self.model.names
        self.is_coco = len(names) == 80 and names[0] == "person"
        self.is_open_vocab = is_open_vocab(names)
        self.conf, self.device, self.imgsz, self.augment = conf, device, imgsz, augment
        self.preprocess = preprocess
        self.min_area, self.max_area = min_area, max_area
        self.mode = mode if mode in MODES else "alta"
        self.weights = (SHAPE_WEIGHT, TEXTURE_WEIGHT)
        self.material_conf = material_conf
        self.calibration: SceneCalibration | None = None
        self.deep_scan_all = False  # Deep Scan em todo objeto (não só nos duvidosos parados no centro)
        self.ignore_labels: frozenset[str] = frozenset()  # classes do detector que nunca viram resíduo
        self.merge_overlap: float | None = None  # junta caixas da mesma categoria com essa sobreposição (None: não)

        if material_model == "auto":
            self.texture = load_texture_expert(device)
        else:
            self.texture = LocalTextureExpert(material_model, device) if material_model else None
        self._deep_scanner_arg = deep_scanner
        self._scanner = None
        self.reset_tracking()

    @classmethod
    def from_env(cls) -> "Detector":
        material = os.environ.get("WASTE_MATERIAL_MODEL", "auto")
        return cls(
            model_path=os.environ.get("WASTE_MODEL") or None,
            conf=float(os.environ.get("DETECTOR_CONF", "0.25")),
            imgsz=int(os.environ.get("WASTE_IMGSZ", "640")),
            augment=os.environ.get("WASTE_TTA", "0") == "1",
            material_model=None if material.lower() in ("off", "0", "none") else material,
            material_conf=float(os.environ.get("WASTE_MATERIAL_CONF", "0.5")),
            mode=os.environ.get("WASTE_MODE", "alta"),
            # No site público, o COCO: mediu quase igual ao YOLOE (~1 ponto a menos no modo normal) e não precisa
            # baixar nem carregar o CLIP (~600 MB a mais de memória), bem além do que os servidores gratuitos garantem
            detector_kind=os.environ.get("WASTE_DETECTOR", "coco" if paths.PUBLIC else "yolo-world"),
            preprocess=os.environ.get("WASTE_PREPROCESS", "0") == "1",
        )

    # ------------------------------------------------------------ informações
    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    @property
    def material(self):
        """O especialista de textura (nome antigo, mantido para quem já usava)."""
        return self.texture

    @property
    def backend(self) -> str:
        if self.is_open_vocab:
            return "yoloe" if "yoloe" in self.model_path.name else "yolo-world"
        return "coco" if self.is_coco else "customizado"

    @property
    def texture_name(self) -> str | None:
        if self.texture is None:
            return None
        return getattr(self.texture, "model_id", None) or Path(self.texture.model_path).name

    @property
    def description(self) -> str:
        stages = [self.backend]
        if self.texture and self.mode != "rapido":
            stages.append(f"textura ({self.texture.name})")
        if self.mode == "ultra":
            stages.append(f"deep scan ({self.scanner.backend})")
        return " + ".join(stages)

    @property
    def scanner(self):
        if self._scanner is None and self.texture is not None:
            from src.deep_scan import DeepScanner

            arg = self._deep_scanner_arg
            self._scanner = arg if not isinstance(arg, str) else DeepScanner(self.texture)
        return self._scanner

    # ------------------------------------------------------------ estágio 1
    def _shape_category(self, label: str) -> tuple[str | None, float]:
        if self.is_open_vocab:
            category = VOCAB_CATEGORY.get(label)
            return category, DEFAULT_WEIGHT_G[category] if category else 0.0
        if label == AGNOSTIC_LABEL:
            category = max(AGNOSTIC_PRIOR, key=AGNOSTIC_PRIOR.get)
            return category, DEFAULT_WEIGHT_G[category]
        if not self.is_coco:
            category = category_from_name(label)
            return category.name, DEFAULT_WEIGHT_G[category.name]
        rule = COCO_RULES.get(label)
        if rule is None:
            return None, 0.0
        return rule.category.name, rule.weight_g

    def calibrate(self, frame: np.ndarray | None) -> None:
        """Guarda a cena vazia como fundo e ajusta o realce à iluminação (None desfaz a calibração)."""
        self.calibration = SceneCalibration.from_frame(frame) if frame is not None else None

    def detect(
        self, image: str | Path | bytes | np.ndarray, waste_only: bool = False, single_item: bool = False,
        whole_image_fallback: bool = True,
    ) -> list[Detection]:
        """Recebe caminho, bytes de arquivo ou array BGR (como o OpenCV lê) e devolve as detecções.

        ``single_item``: a foto tem um resíduo só (lixeira com câmera fixa). O especialista de textura olha a foto
        inteira e, se tiver certeza, decide a categoria do item; o detector só dá a posição.
        ``whole_image_fallback``: sem ``single_item``, se nenhum resíduo for achado, tenta a foto inteira (fotos de
        perto). O vídeo desliga isso: num quadro vazio da câmera, não há nada a achar.
        """
        image = load_image(image)
        h, w = image.shape[:2]
        clip = self.calibration.clip_limit if self.calibration else 2.0
        source = enhance(image, clip) if self.preprocess else image
        result = self.model.predict(
            source, conf=self.conf, imgsz=self.imgsz, augment=self.augment, device=self.device, verbose=False
        )[0]
        names = result.names
        detections = []
        for xyxy, p, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            label = names[int(c)]
            if self.is_open_vocab and label in IGNORED_VOCABULARY:
                continue  # estão no vocabulário só para o detector não confundir pessoas e mãos com resíduo
            if label in self.ignore_labels:
                continue
            category, weight = self._shape_category(label)
            if waste_only and category is None:
                continue
            box = tuple(float(v) for v in xyxy)
            area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            if category and not single_item and not self.min_area <= area <= self.max_area:
                continue  # resíduo "do tamanho da imagem" é parede, mesa ou o próprio corpo; minúsculo é ruído
            if self.calibration and self.calibration.foreground_ratio(image, box) < BACKGROUND_MIN_CHANGE:
                continue  # é o fundo calibrado
            if label == AGNOSTIC_LABEL:
                probs = dict(AGNOSTIC_PRIOR)
            else:
                probs = shape_vote(category, float(p)) if category else None
            detections.append(Detection(label, float(p), box, category, weight, probs=probs))

        if self.texture and self.mode != "rapido":
            detections = self._ensemble(image, detections)
        if self.merge_overlap is not None and not single_item:
            detections = merge_duplicates(detections, self.merge_overlap)
        if self.texture and self.mode != "rapido":
            if single_item:
                detections = self._single_item(image, detections)
            elif whole_image_fallback and not any(d.category for d in detections):
                # Nada achado: quase sempre é uma foto de perto (o objeto ocupa a foto toda, o detector não o
                # isola ou o filtro de 70% o descarta). A foto inteira entra, se a textura tiver certeza.
                detections = self._single_item(image, detections)
        return detections

    # ------------------------------------------------------------ estágio 2
    def _ensemble(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Junta o voto do detector com o do especialista de textura (40% forma, 60% textura).

        No COCO, só os objetos de material ambíguo (garrafa, copo, pote…) passam pela textura: comida e
        eletrônicos já estão certos pelo nome, e o TrashNet não tem essas classes.
        """
        todo = [i for i, d in enumerate(detections) if d.category and (not self.is_coco or d.label in REFINABLE_LABELS)]
        predictions = self.texture.predict([crop(image, detections[i].box) for i in todo])
        for i, pred in zip(todo, predictions):
            det = detections[i]
            mixed = ensemble(det.probs, pred.probs, *self.weights)
            category, _ = top(mixed)
            weight = det.weight_g if category == det.category else DEFAULT_WEIGHT_G[category]
            detections[i] = replace(
                det, category=category, weight_g=weight, material=pred.label,
                material_confidence=pred.confidence, probs=mixed,
            )
        return detections

    def _single_item(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Um item por foto: vira uma detecção só, com a categoria que a textura vê na foto inteira.

        No teste do TrashNet, deixar o detector decidir errava 16% das fotos: o COCO "vê" papelão como celular e
        coisas alaranjadas como banana. Se o especialista não tiver certeza, fica o que o detector achou.

        A certeza que conta é a da categoria, não a da classe: papelão 45% + papel 40% é Papel com 85%.
        """
        (pred,) = self.texture.predict([image])
        category, certainty = top(pred.probs)
        if certainty < self.material_conf:
            return detections
        waste = [d for d in detections if d.category]
        others = [d for d in detections if not d.category]
        if waste:
            best = max(waste, key=lambda d: d.confidence)
            weight = best.weight_g if best.category == category else DEFAULT_WEIGHT_G[category]
            item = replace(
                best, category=category, weight_g=weight, material=pred.label,
                material_confidence=pred.confidence, probs=pred.probs,
            )
        else:
            h, w = image.shape[:2]
            item = Detection(
                pred.label, pred.confidence, (0.0, 0.0, float(w), float(h)), category,
                DEFAULT_WEIGHT_G[category], pred.label, pred.confidence, pred.probs,
            )
        return [item] + others

    def deep_scan(self, image: np.ndarray, det: Detection) -> Detection:
        """Deep Scan síncrono de um objeto de uma foto (o resultado dele decide a categoria)."""
        if self.scanner is None:
            return det
        result = self.scanner.scan_now(crop(image, det.box))
        weight = det.weight_g if result.category == det.category else DEFAULT_WEIGHT_G[result.category]
        return replace(det, category=result.category, weight_g=weight, scan=result, probs=result.probs or det.probs)

    # ------------------------------------------------------------ estágio 3 (vídeo)
    def reset_tracking(self) -> None:
        """Começa um vídeo novo: ids e históricos do anterior não valem mais."""
        from trackers import ByteTrackTracker  # o ByteTrack do Roboflow saiu do supervision para o pacote trackers

        # Detectores de vocabulário aberto dão confianças baixas (0,1 a 0,4): os limiares do ByteTrack acompanham
        self.tracker = ByteTrackTracker(
            lost_track_buffer=30, frame_rate=15, track_activation_threshold=self.conf,
            minimum_consecutive_frames=1, high_conf_det_threshold=self.conf,
        )
        self.voter = TemporalVoter()

    def track(self, frame: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        """Detecta e rastreia num quadro de vídeo: cada objeto ganha um ``tracker_id``, categoria estável
        (EMA de 12 quadros), caixa suavizada e, no modo ultra, o veredito do Deep Scan."""
        import supervision as sv

        t = time.monotonic() if timestamp is None else timestamp
        h, w = frame.shape[:2]
        detections = self.detect(frame, waste_only=True, whole_image_fallback=False)
        if detections:
            tracked = self.tracker.update(
                sv.Detections(
                    xyxy=np.array([d.box for d in detections], dtype=float),
                    confidence=np.array([d.confidence for d in detections], dtype=float),
                    class_id=np.zeros(len(detections), dtype=int),
                    data={"index": np.arange(len(detections))},
                )
            )
        else:
            tracked = self.tracker.update(sv.Detections.empty())

        out = []
        for xyxy, tid, index in zip(tracked.xyxy, tracked.tracker_id, tracked.data.get("index", [])):
            if tid < 0:
                continue  # ainda não virou trilha (aparece a partir do quadro seguinte)
            det, tid = detections[int(index)], int(tid)
            state = self.voter.update(tid, det.probs or shape_vote(det.category, det.confidence), xyxy, t, (w, h))
            category = state.shown
            weight = det.weight_g if category == det.category else DEFAULT_WEIGHT_G[category]
            det = replace(det, tracker_id=tid, category=category, weight_g=weight, probs=dict(state.ema),
                          box=self.voter.smoothed_box(state))
            if self.mode == "ultra" and self.scanner is not None:
                result = self.scanner.result(tid)
                if result:
                    weight = det.weight_g if result.category == det.category else DEFAULT_WEIGHT_G[result.category]
                    det = replace(det, category=result.category, weight_g=weight, scan=result)
                elif not state.scanned and (
                    self.deep_scan_all
                    or (self.voter.is_still_in_center(state)
                        and DEEP_SCAN_RANGE[0] <= state.confidence <= DEEP_SCAN_RANGE[1])
                ):
                    self.scanner.submit(tid, crop(frame, det.box))
                    state.scanned = True
            out.append(det)
        self.voter.end_frame()
        return out


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


def annotate(image: np.ndarray, detections, track_ids: list[int] | None = None, compact: bool = False) -> np.ndarray:
    """Desenha as caixas nas cores da CONAMA (cinza para o que não é resíduo). Não altera a imagem original.

    ``compact``: só o número e a categoria (``#3 Metal``), para quem não precisa dos detalhes técnicos.
    """
    out = image.copy()
    thickness = max(2, round(min(out.shape[:2]) / 300))
    for i, det in enumerate(detections):
        color = CATEGORIES[det.category].bgr if det.category else (150, 150, 150)
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if compact:  # "Orgânico/Rejeito" ocuparia meia caixa: só o primeiro nome
            text = (det.category or det.label).split("/")[0]
        else:
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
