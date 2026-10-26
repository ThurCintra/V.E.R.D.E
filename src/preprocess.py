"""Pré-processamento da imagem antes da detecção e calibração da cena (fundo e iluminação).

Desligado por padrão: no teste do TrashNet, o CLAHE empatou no modo de 1 objeto (93,9%) e tirou 6 pontos do modo
de vários objetos com o YOLOE (31,4% → 25,6%). Pode valer a pena em cenas escuras ou com muito reflexo; ligue no
painel ("Realce de contraste") ou com ``WASTE_PREPROCESS=1`` e compare.

``enhance``: CLAHE no canal L do espaço LAB (realça contraste local sem mexer nas cores, o que reduz o efeito de
reflexos em plástico/vidro e de sombras) + filtro bilateral leve (tira ruído da webcam preservando bordas).

``SceneCalibration``: guardada quando o usuário aperta "Zerar fundo / Calibrar iluminação" com a câmera vazia.
Ajusta a força do CLAHE ao contraste da cena e descarta detecções em regiões iguais ao fundo (parede, mesa,
a própria lixeira).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def enhance(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8, denoise: bool = True) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if denoise:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=35, sigmaSpace=35)
    return out


@dataclass
class SceneCalibration:
    background: np.ndarray  # quadro da cena vazia, em tons de cinza e reduzido
    clip_limit: float

    SIZE = (160, 120)

    @classmethod
    def from_frame(cls, frame: np.ndarray) -> "SceneCalibration":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std())
        # Cena chapada (pouco contraste, luz fraca) pede CLAHE mais forte; cena já contrastada, mais suave
        clip_limit = float(np.clip(3.5 - contrast / 30, 1.2, 3.5))
        return cls(cv2.GaussianBlur(cv2.resize(gray, cls.SIZE), (5, 5), 0), clip_limit)

    def foreground_ratio(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> float:
        """Fração da caixa que mudou em relação ao fundo (0 = idêntica ao fundo vazio)."""
        h, w = frame.shape[:2]
        gray = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.SIZE), (5, 5), 0)
        sx, sy = self.SIZE[0] / w, self.SIZE[1] / h
        x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(np.ceil(box[2] * sx)), int(np.ceil(box[3] * sy))
        region = cv2.absdiff(gray[y1:y2, x1:x2], self.background[y1:y2, x1:x2])
        return float((region > 25).mean()) if region.size else 0.0
