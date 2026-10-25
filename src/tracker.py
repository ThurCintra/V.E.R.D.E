"""Rastreamento simples por IoU, para contar cada objeto do vídeo uma vez só.

Cada detecção é associada à trilha com maior sobreposição (IoU) no frame anterior. Uma trilha só conta como
item novo depois de aparecer em ``min_hits`` frames (descarta detecções-fantasma de um frame) e morre depois
de ``max_missed`` frames sem ser vista (o objeto saiu de cena, ou passou da esteira). A categoria final é a
mais votada ao longo da trilha, o que absorve o modelo hesitando entre duas classes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.detector import Detection

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


@dataclass
class Track:
    id: int
    box: Box
    hits: int = 1
    missed: int = 0
    counted: bool = False
    labels: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    confidences: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def add(self, det: Detection) -> None:
        self.box = det.box
        self.labels[det.label] += 1
        if det.category:
            self.categories[det.category] += 1
        self.confidences.append(det.confidence)
        self.weights.append(det.weight_g)

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0]

    @property
    def category(self) -> str | None:
        return self.categories.most_common(1)[0][0] if self.categories else None

    @property
    def confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences)

    @property
    def weight_g(self) -> float:
        return sum(self.weights) / len(self.weights)


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, min_hits: int = 3, max_missed: int = 10):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self.counts: Counter = Counter()  # itens contados por categoria
        self._next_id = 1

    def update(self, detections: list[Detection]) -> tuple[list[tuple[Detection, int]], list[Track]]:
        """Processa as detecções de um frame.

        Devolve (cada detecção com o id da sua trilha, trilhas que acabaram de ser confirmadas como item novo).
        """
        pairs = sorted(
            (
                (iou(t.box, d.box), ti, di)
                for ti, t in enumerate(self.tracks)
                for di, d in enumerate(detections)
            ),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        assigned: dict[int, Track] = {}
        for score, ti, di in pairs:
            if score < self.iou_threshold:
                break
            if ti in matched_tracks or di in assigned:
                continue
            matched_tracks.add(ti)
            track = self.tracks[ti]
            track.hits += 1
            track.missed = 0
            track.add(detections[di])
            assigned[di] = track

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.missed += 1

        for di, det in enumerate(detections):
            if di not in assigned:
                track = Track(self._next_id, det.box)
                track.add(det)
                self._next_id += 1
                self.tracks.append(track)
                assigned[di] = track

        confirmed = []
        for track in self.tracks:
            if not track.counted and track.hits >= self.min_hits:
                track.counted = True
                confirmed.append(track)
                if track.category:
                    self.counts[track.category] += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [(det, assigned[di].id) for di, det in enumerate(detections)], confirmed
