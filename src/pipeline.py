"""Vídeo e câmera: detecção + rastreamento (ByteTrack) + gravação + acionamento, quadro a quadro.

Cada objeto (``tracker_id``) é contado uma vez, quando já foi visto em ``min_frames`` quadros, com a categoria
estabilizada pela votação temporal do detector. Usado pelo dashboard.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.database import Database
from src.detector import Detection, Detector, annotate
from src.hardware_controller import HardwareController
from src.review import ReviewCollector


@dataclass
class FrameResult:
    index: int
    detections: list[Detection]  # com tracker_id
    new_items: list[Detection]  # objetos contados pela primeira vez neste quadro
    annotated: np.ndarray

    @property
    def track_ids(self) -> list[int]:
        return [d.tracker_id for d in self.detections]


class WastePipeline:
    def __init__(
        self,
        detector: Detector,
        db: Database | None = None,
        controller: HardwareController | None = None,
        source: str = "video",
        review: ReviewCollector | None = None,
        min_frames: int = 3,
    ):
        """``min_frames``: quadros em que o objeto precisa aparecer antes de contar (descarta detecções-fantasma)."""
        self.detector = detector
        self.db = db
        self.controller = controller
        self.review = review
        self.source = source
        self.min_frames = min_frames
        self.counts: Counter = Counter()
        self._counted: set[int] = set()
        self._index = 0
        detector.reset_tracking()

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> FrameResult:
        detections = self.detector.track(frame, timestamp)
        if self.review:
            self.review.maybe_save(frame, detections, self.source)
        new_items = []
        for det in detections:
            state = self.detector.voter.tracks.get(det.tracker_id)
            if det.tracker_id in self._counted or state is None or state.frames < self.min_frames:
                continue
            self._counted.add(det.tracker_id)
            new_items.append(det)
            self.counts[det.category] += 1
            if self.db:
                self.db.add(det.label, det.category, det.confidence, det.weight_g, self.source, det.tracker_id)
            if self.controller:
                self.controller.actuate(det.category)
        annotated = annotate(frame, detections, [d.tracker_id for d in detections], compact=True)
        result = FrameResult(self._index, detections, new_items, annotated)
        self._index += 1
        return result

    def process_video(self, path: str | Path | int, every_n: int = 1, max_frames: int | None = None) -> Iterator[FrameResult]:
        """Processa um vídeo (ou câmera: índice/URL RTSP). ``every_n`` pula quadros para ganhar velocidade.

        Em arquivos, o tempo vem da posição do quadro (o "parado há 1,5 s" do Deep Scan vale em tempo do vídeo);
        em câmeras, do relógio.
        """
        live = isinstance(path, int) or str(path).startswith(("rtsp:", "http:", "https:"))
        cap = cv2.VideoCapture(path if isinstance(path, int) else str(path))
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        try:
            read = 0
            while max_frames is None or read < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                read += 1
                if (read - 1) % every_n == 0:
                    yield self.process_frame(frame, time.monotonic() if live else (read - 1) / fps)
        finally:
            cap.release()
