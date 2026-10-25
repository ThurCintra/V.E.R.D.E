"""Estágio 3: estabilização temporal por objeto rastreado (``tracker_id`` do ByteTrack).

- **Votação por histórico (EMA):** as probabilidades de material dos últimos 12 quadros entram numa média móvel
  exponencial. A categoria mostrada só troca quando a nova passa de ``switch_threshold`` *e* supera a atual por
  ``margin``: sem isso, um objeto entre "plástico" e "vidro" piscaria de cor a cada quadro.
- **Caixa suavizada:** filtro de Savitzky-Golay (scipy) sobre as últimas posições, que tira o tremor da caixa sem
  atrasá-la como uma média simples.
- **Parado no centro:** detecta objeto imóvel no meio da imagem há ``still_seconds``, o gatilho do Deep Scan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

HISTORY = 12


@dataclass
class TrackState:
    ema: dict[str, float]
    shown: str
    boxes: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    times: deque = field(default_factory=lambda: deque(maxlen=60))
    centers: deque = field(default_factory=lambda: deque(maxlen=60))
    frames: int = 0
    scanned: bool = False

    @property
    def confidence(self) -> float:
        return self.ema[self.shown]


class TemporalVoter:
    def __init__(self, history: int = HISTORY, switch_threshold: float = 0.5, margin: float = 0.1,
                 still_seconds: float = 1.5, max_missing_frames: int = 30):
        self.alpha = 2 / (history + 1)  # EMA equivalente a uma janela de ``history`` quadros
        self.switch_threshold, self.margin = switch_threshold, margin
        self.still_seconds = still_seconds
        self.max_missing_frames = max_missing_frames
        self.tracks: dict[int, TrackState] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    def update(self, tracker_id: int, probs: dict[str, float], box, timestamp: float, frame_size) -> TrackState:
        state = self.tracks.get(tracker_id)
        if state is None:
            category = max(probs, key=probs.get)
            state = self.tracks[tracker_id] = TrackState(dict(probs), category)
        else:
            for c, p in probs.items():
                state.ema[c] = (1 - self.alpha) * state.ema.get(c, 0.0) + self.alpha * p
            best = max(state.ema, key=state.ema.get)
            if (best != state.shown and state.ema[best] >= self.switch_threshold
                    and state.ema[best] - state.ema[state.shown] >= self.margin):
                state.shown = best
        state.frames += 1
        state.boxes.append(tuple(float(v) for v in box))
        w, h = frame_size
        state.times.append(timestamp)
        state.centers.append(((box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h))
        self._last_seen[tracker_id] = self._frame
        return state

    def end_frame(self) -> None:
        """Esquece objetos que sumiram há muito tempo (o ByteTrack não reaproveita ids)."""
        self._frame += 1
        for tid in [t for t, seen in self._last_seen.items() if self._frame - seen > self.max_missing_frames]:
            self.tracks.pop(tid, None)
            self._last_seen.pop(tid, None)

    @staticmethod
    def smoothed_box(state: TrackState) -> tuple[float, float, float, float]:
        boxes = np.array(state.boxes)
        if len(boxes) < 5:
            return tuple(boxes[-1])
        window = len(boxes) if len(boxes) % 2 else len(boxes) - 1
        return tuple(float(v) for v in savgol_filter(boxes, window_length=window, polyorder=2, axis=0)[-1])

    def is_still_in_center(self, state: TrackState, tolerance: float = 0.03) -> bool:
        """Parado (o centro mexeu menos de 3% da imagem) no terço central há ``still_seconds``."""
        if not state.times or state.times[-1] - state.times[0] < self.still_seconds:
            return False
        recent = [c for t, c in zip(state.times, state.centers) if state.times[-1] - t <= self.still_seconds]
        xs, ys = zip(*recent)
        cx, cy = recent[-1]
        centered = 1 / 3 <= cx <= 2 / 3 and 1 / 3 <= cy <= 2 / 3
        return centered and max(xs) - min(xs) < tolerance and max(ys) - min(ys) < tolerance
