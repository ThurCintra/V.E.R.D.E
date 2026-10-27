"""Câmera ao vivo fluida: o vídeo na tela não espera a análise.

Três ritmos independentes:

- ``CameraReader`` lê a câmera sem parar numa thread e guarda só o quadro mais recente (a câmera nunca acumula
  atraso, mesmo que a análise seja mais lenta que ela);
- ``LiveSession`` analisa numa outra thread, sempre o quadro mais novo (detector + textura + rastreamento +
  contagem, com a mesma precisão da análise quadro a quadro);
- a tela pega ``LiveSession.view()`` no ritmo dela (~20 por segundo): o quadro mais novo da câmera com as caixas
  da última análise.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from src.detector import annotate
from src.pipeline import FrameResult, WastePipeline

log = logging.getLogger(__name__)


def open_source(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, str) and source.strip().isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Não foi possível abrir a câmera: {source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # nem todas as câmeras aceitam; a thread garante o quadro novo de qualquer jeito
    return cap


class CameraReader:
    """Lê a câmera numa thread; ``latest()`` devolve o quadro mais recente e o número dele."""

    def __init__(self, source):
        # Uma VideoCapture já aberta (ou algo com read/release) também serve
        self._cap = open_source(source) if isinstance(source, (str, int)) else source
        # Um arquivo de vídeo no lugar da câmera é lido no ritmo dele (uma câmera já manda nesse ritmo)
        is_file = isinstance(source, str) and not source.strip().isdigit() and "://" not in source
        self._interval = 1 / (self._cap.get(cv2.CAP_PROP_FPS) or 15) if is_file else 0.0
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self.ended = False  # a câmera parou de mandar quadros (desligada, arquivo no fim)
        self._thread = threading.Thread(target=self._run, name="verde-camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                failures += 1
                if failures > 50:  # ~ alguns segundos sem imagem
                    break
                time.sleep(0.02)
                continue
            failures = 0
            if self._interval:
                time.sleep(self._interval)
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()
        self.ended = True
        with self._cond:
            self._cond.notify_all()
        self._cap.release()

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._cond:
            return self._seq, self._frame

    def wait_newer(self, seq: int, timeout: float = 1.0) -> tuple[int, np.ndarray | None]:
        """Espera um quadro mais novo que ``seq`` (ou o tempo acabar / a câmera parar)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seq or self.ended or self._stop.is_set(), timeout)
            return self._seq, self._frame

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2)


class LiveSession:
    """Câmera + análise em segundo plano. ``view()`` para a tela; ``result`` é a última análise."""

    def __init__(self, pipeline: WastePipeline, source):
        self.pipeline = pipeline
        self.reader = CameraReader(source)
        self.result: FrameResult | None = None
        self.error: Exception | None = None
        self.analyzed = 0
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="verde-analise", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        seq = 0
        try:
            while not self._stop.is_set():
                newer, frame = self.reader.wait_newer(seq)
                if newer == seq or frame is None:
                    if self.reader.ended:
                        break  # a câmera parou e o último quadro já foi analisado
                    continue
                seq = newer
                self.result = self.pipeline.process_frame(frame, time.monotonic(), draw=False)
                self.analyzed += 1
        except Exception as err:  # mostrado na tela em vez de matar a thread em silêncio
            log.exception("Falha na análise ao vivo")
            self.error = err

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def analysis_fps(self) -> float:
        return self.analyzed / max(time.monotonic() - self._started, 1e-6)

    def view(self) -> np.ndarray | None:
        """O quadro mais novo da câmera com as caixas da última análise."""
        _, frame = self.reader.latest()
        if frame is None:
            return None
        result = self.result
        if result is None:
            return frame
        return annotate(frame, result.detections, result.track_ids, compact=True)

    def stop(self) -> None:
        self._stop.set()
        self.reader.stop()
        self._thread.join(timeout=5)
