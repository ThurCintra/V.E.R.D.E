"""Detecção + rastreamento + gravação + acionamento, frame a frame. Usado pelo dashboard e pela API."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.database import Database
from src.detector import Detection, Detector, annotate
from src.hardware_controller import HardwareController
from src.review import ReviewCollector
from src.tracker import SimpleTracker, Track


@dataclass
class FrameResult:
    index: int
    detections: list[Detection]
    track_ids: list[int]
    new_items: list[Track]  # objetos contados pela primeira vez neste frame
    annotated: np.ndarray


class WastePipeline:
    def __init__(
        self,
        detector: Detector,
        db: Database | None = None,
        controller: HardwareController | None = None,
        tracker: SimpleTracker | None = None,
        source: str = "video",
        review: ReviewCollector | None = None,
    ):
        self.detector = detector
        self.db = db
        self.controller = controller
        self.review = review
        self.tracker = tracker or SimpleTracker()
        self.source = source
        self._index = 0

    @property
    def counts(self):
        return self.tracker.counts

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        detections = self.detector.detect(frame, waste_only=True)
        if self.review:
            self.review.maybe_save(frame, detections, self.source)
        pairs, new_items = self.tracker.update(detections)
        track_ids = [track_id for _, track_id in pairs]
        for track in new_items:
            if self.db and track.category:
                self.db.add(track.label, track.category, track.confidence, track.weight_g, self.source, track.id)
            if self.controller:
                self.controller.actuate(track.category)
        result = FrameResult(self._index, detections, track_ids, new_items, annotate(frame, detections, track_ids))
        self._index += 1
        return result

    def process_video(self, path: str | Path, every_n: int = 1, max_frames: int | None = None) -> Iterator[FrameResult]:
        """Processa um vídeo (ou câmera: índice/URL RTSP). ``every_n`` pula frames para ganhar velocidade."""
        cap = cv2.VideoCapture(path if isinstance(path, int) else str(path))
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {path}")
        try:
            read = 0
            while max_frames is None or read < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                read += 1
                if (read - 1) % every_n == 0:
                    yield self.process_frame(frame)
        finally:
            cap.release()
