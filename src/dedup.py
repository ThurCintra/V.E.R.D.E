"""Fotos repetidas entre datasets: a mesma imagem no treino e no teste faria a avaliação mentir.

O TrashBox, montado com imagens da internet, traz 112 das 379 fotos do teste do TrashNet e repete 116 das suas
próprias fotos de teste no treino. ``find_duplicates`` acha esses casos em duas etapas: um *dHash* de 64 bits
(resiste a redimensionar e recomprimir) separa os candidatos, e a correlação dos pixels numa miniatura confirma,
porque fotos lisas de fundo branco às vezes têm hashes parecidos sem serem a mesma foto.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

MAX_DISTANCE = 6  # bits diferentes no dHash para virar candidato
MIN_CORRELATION = 0.95  # correlação dos pixels (miniatura 64x64) para confirmar a repetição


def dhash(path: str | Path) -> int | None:
    """Hash perceptual de 64 bits (diferença entre pixels vizinhos numa miniatura 9x8). None se não abrir."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def hashes(paths: list[Path], workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """dHash de cada foto e quais abriram (as ilegíveis nunca batem com nada)."""
    with ThreadPoolExecutor(workers) as pool:  # o OpenCV solta o GIL ao ler e redimensionar
        values = list(pool.map(dhash, paths))
    ok = np.array([v is not None for v in values])
    return np.array([v or 0 for v in values], dtype=np.uint64), ok


def _thumb(path: Path) -> np.ndarray:
    pixels = cv2.resize(cv2.imread(str(path)), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    pixels -= pixels.mean()
    return pixels / (np.linalg.norm(pixels) + 1e-9)


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distância de Hamming entre cada hash de ``a`` e cada um de ``b`` (matriz len(a) x len(b))."""
    xor = a[:, None] ^ b[None, :]
    return np.unpackbits(xor.view(np.uint8).reshape(len(a), len(b), 8), axis=-1).sum(-1)


def find_duplicates(
    candidates: list[Path],
    references: list[Path],
    max_distance: int = MAX_DISTANCE,
    min_correlation: float = MIN_CORRELATION,
    workers: int = 4,
) -> set[Path]:
    """As fotos de ``candidates`` que repetem alguma de ``references`` (com outro tamanho ou compressão)."""
    if not candidates or not references:
        return set()
    (cand_h, cand_ok), (ref_h, ref_ok) = hashes(candidates, workers), hashes(references, workers)
    found: set[Path] = set()
    thumbs: dict[Path, np.ndarray] = {}

    def thumb(path: Path) -> np.ndarray:
        if path not in thumbs:
            thumbs[path] = _thumb(path)
        return thumbs[path]

    for start in range(0, len(candidates), 256):
        dist = _hamming(cand_h[start : start + 256], ref_h)
        for i, j in zip(*np.nonzero(dist <= max_distance)):
            cand = candidates[start + i]
            if cand in found or not (cand_ok[start + i] and ref_ok[j]):
                continue
            if float(thumb(cand) @ thumb(references[j])) >= min_correlation:
                found.add(cand)
    return found


def find_repeats(paths: list[Path], min_correlation: float = MIN_CORRELATION, workers: int = 4) -> set[Path]:
    """Cópias dentro de uma mesma lista (hash idêntico e pixels iguais): fica a primeira de cada grupo."""
    first_of: dict[int, list[Path]] = {}
    repeats: set[Path] = set()
    values, ok = hashes(paths, workers)
    for path, h, good in zip(paths, values.tolist(), ok):
        if not good:
            continue
        group = first_of.setdefault(h, [])
        if any(float(_thumb(path) @ _thumb(kept)) >= min_correlation for kept in group):
            repeats.add(path)
        else:
            group.append(path)
    return repeats


def readable(paths: list[Path], workers: int = 4) -> list[Path]:
    """Só as fotos que o OpenCV consegue abrir (o TrashBox tem algumas corrompidas)."""
    with ThreadPoolExecutor(workers) as pool:
        ok = list(pool.map(lambda p: cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8) is not None, paths))
    return [p for p, good in zip(paths, ok) if good]
