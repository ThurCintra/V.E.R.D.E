"""Motor de inferência híbrido de 3 estágios para resíduos recicláveis.

1. **Detecção espacial** (rápida): YOLOE com vocabulário de resíduos (``src/open_vocab.py``; o modelo pronto vem
   no repositório) ou, se pedido, YOLO-World; pré-processamento CLAHE opcional (``src/preprocess.py``), filtro de
   tamanho (ignora o que ocupa mais de 70% da imagem, como paredes e o próprio corpo) e descarte de pessoas. Se o
   vocabulário não puder ser carregado, o detector cai para o outro e, por fim, para o YOLOv8 do COCO.
2. **Especialista de textura** (``src/material_classifier.py``): classifica o recorte de cada objeto; o resultado
   é combinado com o detector (40% forma, 60% textura).
3. **Estabilização temporal** (vídeo, ``track``): o ByteTrack (pacote ``trackers``, sobre o ``supervision``) dá
   um ``tracker_id`` a cada objeto, a votação por EMA dos últimos 12 quadros impede a categoria de piscar e o Deep Scan (``src/deep_scan.py``) dá a palavra
   final em objetos duvidosos parados no centro (ou em todos, se pedido).

Modos de operação: ``rapido`` (só o estágio 1), ``alta`` (1 + 2) e ``ultra`` (1 + 2 + Deep Scan).

Configuração por ambiente (``Detector.from_env``): ``WASTE_MODE``, ``WASTE_DETECTOR`` (``yoloe``, o padrão,
``yolo-world`` ou ``coco``), ``WASTE_MODEL`` (um peso específico), ``WASTE_IMGSZ``, ``DETECTOR_CONF``, ``WASTE_TTA=1``,
``WASTE_PREPROCESS=1`` (liga o CLAHE), ``WASTE_MATERIAL_MODEL`` (``off`` desliga o estágio 2), ``WASTE_HF_MODEL`` e
``WASTE_MATERIAL_CONF``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from src.material import REFINABLE_LABELS, crop
from src import accel
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
from src.paths import CUSTOM_MODEL, DEFAULT_MODEL
from src.preprocess import SceneCalibration, enhance
from src.recycling import CATEGORIES, COCO_RULES, DEFAULT_WEIGHT_G, METAL, category_from_name
from src.temporal import TemporalVoter
from src.weights import resolve

log = logging.getLogger(__name__)

MODES = {"rapido": "Rápido", "alta": "Alta precisão", "ultra": "Ultra precisão"}
DEEP_SCAN_RANGE = (0.35, 0.60)  # confiança "duvidosa" que dispara o Deep Scan automático
BACKGROUND_MIN_CHANGE = 0.15  # com o fundo calibrado, caixas com menos de 15% de mudança são o próprio fundo
TEXTURE_EVERY = 5  # no vídeo, o especialista de textura revê cada objeto a cada 5 quadros


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

    @property
    def command(self) -> str | None:
        return CATEGORIES[self.category].command if self.category else None

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
            "command": self.command,
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


def default_model_path() -> Path:
    """O detector treinado para resíduos, se existir; senão o YOLOv8 do COCO."""
    return CUSTOM_MODEL if CUSTOM_MODEL.is_file() else DEFAULT_MODEL


def _accelerate(model, path: Path):
    """O mesmo modelo rodando no OpenVINO, se ele estiver exportado ao lado do .pt; senão o próprio ``model``."""
    fast = accel.load_openvino(accel.openvino_dir(path), model.task, list(model.names.values()))
    return (fast, "openvino") if fast is not None else (model, "pytorch")


def _load_detector(model_path, kind: str):
    """(modelo, caminho, aviso). Sem peso explícito: vocabulário aberto pedido → o outro → treinado/COCO.

    O ``ultralytics`` (e o torch) só é importado aqui: importar este módulo fica leve, e as páginas do dashboard
    que não usam IA abrem sem esperar por ele."""
    from ultralytics import YOLO

    if model_path:
        path = resolve(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Modelo não encontrado em {path}. Rode: python scripts/download_model.py")
        return YOLO(str(path)), path, None
    if CUSTOM_MODEL.is_file():
        return YOLO(str(CUSTOM_MODEL)), CUSTOM_MODEL, None
    errors = []
    candidates = [] if kind == "coco" else list(dict.fromkeys(k for k in (kind, "yoloe", "yolo-world")
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
        material_conf: float = 0.6,
        mode: str = "alta",
        detector_kind: str = "yoloe",
        preprocess: bool = False,
        min_area: float = 0.0005,
        max_area: float = 0.70,
        deep_scanner="auto",
    ):
        """``material_model``: ``"auto"`` (Hugging Face se configurado, senão o classificador local), um caminho
        de classificador local, ou ``None`` (sem estágio 2). ``material_conf``: certeza mínima para, no modo de
        1 item por foto, a foto inteira decidir. ``min_area``/``max_area``: fração da imagem que uma caixa pode
        ocupar."""
        model, self.model_path, self.fallback_note = _load_detector(model_path, detector_kind)
        self.model, self.accelerator = _accelerate(model, self.model_path)
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

        if material_model == "auto":
            self.texture = load_texture_expert(device)
        else:
            self.texture = LocalTextureExpert(material_model, device) if material_model else None
        self._deep_scanner_arg = deep_scanner
        self._scanner = None
        # O modelo do YOLO não pode ser usado por duas threads ao mesmo tempo (duas abas do dashboard, ou a API
        # atendendo dois pedidos): as análises entram uma de cada vez.
        self._lock = threading.RLock()
        self.reset_tracking()

    @classmethod
    def from_env(cls) -> "Detector":
        from src.deep_scan import load_env

        load_env()  # WASTE_DETECTOR, WASTE_HF_MODEL etc. também valem quando escritos no .env
        material = os.environ.get("WASTE_MATERIAL_MODEL", "auto")
        return cls(
            model_path=os.environ.get("WASTE_MODEL") or None,
            conf=float(os.environ.get("DETECTOR_CONF", "0.25")),
            imgsz=int(os.environ.get("WASTE_IMGSZ", "640")),
            augment=os.environ.get("WASTE_TTA", "0") == "1",
            material_model=None if material.lower() in ("off", "0", "none") else material,
            material_conf=float(os.environ.get("WASTE_MATERIAL_CONF", "0.6")),
            mode=os.environ.get("WASTE_MODE", "alta"),
            detector_kind=os.environ.get("WASTE_DETECTOR", "yoloe"),
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
        stages = [self.backend + (" (OpenVINO)" if self.accelerator == "openvino" else "")]
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
        with self._lock:
            return self._detect(load_image(image), waste_only, single_item, whole_image_fallback)

    def _detect(self, image: np.ndarray, waste_only: bool, single_item: bool, whole_image_fallback: bool,
                texture: bool = True):
        h, w = image.shape[:2]
        clip = self.calibration.clip_limit if self.calibration else 2.0
        source = enhance(image, clip) if self.preprocess else image
        result = self.model.predict(
            source, conf=self.conf, imgsz=self.imgsz, augment=self.augment, device=self.device, verbose=False,
            # O YOLOE já faz isso no PyTorch (uma caixa, um rótulo); pelo OpenVINO precisa ser pedido
            agnostic_nms=self.is_open_vocab,
        )[0]
        names = result.names
        detections = []
        for xyxy, p, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            label = names[int(c)]
            if self.is_open_vocab and label in IGNORED_VOCABULARY:
                continue  # estão no vocabulário só para o detector não confundir pessoas e mãos com resíduo
            category, weight = self._shape_category(label)
            if waste_only and category is None:
                continue
            box = tuple(float(v) for v in xyxy)
            area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            if category and not single_item and not self.min_area <= area <= self.max_area:
                continue  # resíduo "do tamanho da imagem" é parede, mesa ou o próprio corpo; minúsculo é ruído
            if self.calibration and self.calibration.foreground_ratio(image, box) < BACKGROUND_MIN_CHANGE:
                continue  # é o fundo calibrado
            probs = shape_vote(category, float(p)) if category else None
            detections.append(Detection(label, float(p), box, category, weight, probs=probs))

        if not texture:
            return detections  # o vídeo aplica a textura por trilha (_track)
        if self.texture and self.mode != "rapido":
            detections = self._ensemble(image, detections)
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
        todo = [i for i, d in enumerate(detections) if self._refinable(d)]
        predictions = self.texture.predict([crop(image, detections[i].box) for i in todo])
        for i, pred in zip(todo, predictions):
            detections[i] = self._with_texture(detections[i], pred)
        return detections

    def _refinable(self, det: Detection) -> bool:
        return bool(det.category) and (not self.is_coco or det.label in REFINABLE_LABELS)

    def _with_texture(self, det: Detection, pred) -> Detection:
        mixed = ensemble(det.probs, pred.probs, *self.weights)
        category, _ = top(mixed)
        weight = det.weight_g if category == det.category else DEFAULT_WEIGHT_G[category]
        return replace(det, category=category, weight_g=weight, material=pred.label,
                       material_confidence=pred.confidence, probs=mixed)

    def _single_item(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Um item por foto: vira uma detecção só, com a categoria que a textura vê na foto inteira.

        No teste do TrashNet, deixar o detector decidir errava 16% das fotos: o COCO "vê" papelão como celular e
        coisas alaranjadas como banana. Se o especialista não tiver certeza, fica o que o detector achou.
        """
        (pred,) = self.texture.predict([image])
        if pred.confidence < self.material_conf:
            return detections
        waste = [d for d in detections if d.category]
        others = [d for d in detections if not d.category]
        if waste:
            best = max(waste, key=lambda d: d.confidence)
            weight = best.weight_g if best.category == top(pred.probs)[0] else DEFAULT_WEIGHT_G[top(pred.probs)[0]]
            item = replace(
                best, category=top(pred.probs)[0], weight_g=weight, material=pred.label,
                material_confidence=pred.confidence, probs=pred.probs,
            )
        else:
            h, w = image.shape[:2]
            category = top(pred.probs)[0]
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
        self._textures: dict[int, tuple[object, int]] = {}  # trilha -> (textura, quadro em que foi vista)
        self._frame = 0

    def track(self, frame: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        """Detecta e rastreia num quadro de vídeo: cada objeto ganha um ``tracker_id``, categoria estável
        (EMA de 12 quadros), caixa suavizada e, no modo ultra, o veredito do Deep Scan."""
        with self._lock:
            return self._track(frame, timestamp)

    def _track(self, frame: np.ndarray, timestamp: float | None) -> list[Detection]:
        import supervision as sv

        t = time.monotonic() if timestamp is None else timestamp
        h, w = frame.shape[:2]
        detections = self._detect(frame, True, False, False, texture=False)
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

        pairs = [(detections[int(i)], int(tid)) for tid, i in zip(tracked.tracker_id, tracked.data.get("index", []))
                 if tid >= 0]  # tid -1: ainda não virou trilha (aparece a partir do quadro seguinte)
        pairs = self._track_textures(frame, pairs)
        out = []
        for xyxy, (det, tid) in zip([b for b, t in zip(tracked.xyxy, tracked.tracker_id) if t >= 0], pairs):
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
        self._frame += 1
        for tid in [t for t in self._textures if t not in self.voter.tracks]:
            del self._textures[tid]
        return out

    def _track_textures(self, frame: np.ndarray, pairs: list[tuple[Detection, int]]) -> list[tuple[Detection, int]]:
        """Textura por trilha: o material de um objeto não muda de um quadro para o outro, então o especialista
        olha cada objeto a cada ``TEXTURE_EVERY`` quadros (e na hora em que ele aparece). Entre uma olhada e outra,
        o voto de textura guardado entra no ensemble com o voto de forma do quadro atual."""
        if not self.texture or self.mode == "rapido":
            return pairs
        stale = [k for k, (det, tid) in enumerate(pairs) if self._refinable(det) and (
            tid not in self._textures or self._frame - self._textures[tid][1] >= TEXTURE_EVERY)]
        for k, pred in zip(stale, self.texture.predict([crop(frame, pairs[k][0].box) for k in stale])):
            self._textures[pairs[k][1]] = (pred, self._frame)
        return [(self._with_texture(det, self._textures[tid][0]) if self._refinable(det) and tid in self._textures
                 else det, tid) for det, tid in pairs]


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
