"""Deep Scan: segunda opinião detalhada sobre um objeto, em segundo plano (não trava o vídeo).

- Com ``GEMINI_API_KEY`` (no ambiente ou no arquivo ``.env`` da raiz do projeto): o recorte vai para o
  ``gemini-2.5-flash`` (``GEMINI_MODEL`` troca o modelo), que devolve categoria, material e uma justificativa
  visual ("Garrafa PET amassada - Plástico reciclável").
- Sem a chave, ou se a chamada falhar: análise local detalhada. O especialista de textura avalia 8 variações do
  recorte (espelhado, aproximado, com contraste realçado…) e a média decide.

Cada objeto é analisado uma vez (por ``key``, normalmente o ``tracker_id``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from src.paths import ROOT
from src.preprocess import enhance
from src.recycling import CATEGORIES

log = logging.getLogger(__name__)

GEMINI_PROMPT = (
    "Você é especialista em triagem de resíduos para a coleta seletiva brasileira (Resolução CONAMA 275). "
    "Olhe o objeto principal da imagem e diga em qual das 5 lixeiras ele vai: Plástico, Papel, Metal, Vidro ou "
    "Orgânico/Rejeito (orgânicos, papel sujo, bitucas, isopor contaminado e o que não for reciclável). "
    "Caixas longa vida vão em Papel. Responda com o material exato e uma justificativa visual curta, em "
    "português (ex.: 'Garrafa PET amassada, transparente e flexível - Plástico reciclável')."
)


@dataclass
class ScanResult:
    category: str
    material: str
    justification: str
    confidence: float
    source: str  # "gemini" ou "local"
    probs: dict[str, float] | None = None


def load_env() -> None:
    """Lê o .env da raiz do projeto (sem sobrescrever o que já está no ambiente)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def _variants(crop: np.ndarray) -> list[np.ndarray]:
    h, w = crop.shape[:2]
    center = crop[h // 10 : h - h // 10, w // 10 : w - w // 10]
    padded = cv2.copyMakeBorder(crop, h // 8, h // 8, w // 8, w // 8, cv2.BORDER_REPLICATE)
    out = [crop, cv2.flip(crop, 1), center, cv2.flip(center, 1), padded, enhance(crop), enhance(center)]
    out.append(cv2.convertScaleAbs(crop, alpha=1.15, beta=10))  # um pouco mais claro
    return [v for v in out if v.size]


def local_scan(texture_expert, crop: np.ndarray) -> ScanResult:
    preds = texture_expert.predict(_variants(crop))
    probs = {c: float(np.mean([p.probs[c] for p in preds])) for c in CATEGORIES}
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    (category, confidence), (second, second_p) = ranked[0], ranked[1]
    labels = [p.label for p in preds]
    material = max(set(labels), key=labels.count)
    justification = (
        f"Análise local de {len(preds)} variações do recorte: {confidence:.0%} {category}"
        + (f", {second_p:.0%} {second}" if second_p >= 0.05 else "")
        + f". Textura mais parecida com '{material}' do TrashNet."
    )
    return ScanResult(category, material, justification, confidence, "local", probs)


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", client=None):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client, self.model = client, model

    def scan(self, crop: np.ndarray) -> ScanResult:
        from typing import Literal

        from google.genai import types
        from pydantic import BaseModel, Field

        class Answer(BaseModel):
            categoria: Literal["Plástico", "Papel", "Metal", "Vidro", "Orgânico/Rejeito"]
            material: str = Field(description="material exato, ex.: PET, alumínio, papelão ondulado")
            justificativa: str = Field(description="justificativa visual curta, em português")
            confianca: float = Field(ge=0, le=1)

        ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=jpeg.tobytes(), mime_type="image/jpeg"), GEMINI_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=Answer, temperature=0
            ),
        )
        answer = Answer(**json.loads(response.text))
        return ScanResult(answer.categoria, answer.material, answer.justificativa, answer.confianca, "gemini")


class DeepScanner:
    def __init__(self, texture_expert, gemini: GeminiClient | None | str = "auto", workers: int = 1):
        if gemini == "auto":
            load_env()
            key = os.environ.get("GEMINI_API_KEY", "").strip()
            gemini = GeminiClient(key, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")) if key else None
        self.gemini = gemini
        self.texture_expert = texture_expert
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deep-scan")
        self._futures: dict[object, Future] = {}
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        return "gemini" if self.gemini else "local"

    def scan_now(self, crop: np.ndarray) -> ScanResult:
        """Análise síncrona (fotos). Se o Gemini falhar, cai na análise local."""
        if self.gemini:
            try:
                return self.gemini.scan(crop)
            except Exception as err:
                log.warning("Gemini indisponível, usando a análise local: %s", err)
        return local_scan(self.texture_expert, crop)

    def submit(self, key, crop: np.ndarray) -> None:
        """Agenda a análise em segundo plano, uma vez por objeto."""
        with self._lock:
            if key not in self._futures:
                self._futures[key] = self._pool.submit(self.scan_now, crop.copy())

    def result(self, key) -> ScanResult | None:
        """O resultado, se já terminou (None enquanto analisa ou se nunca foi pedido)."""
        future = self._futures.get(key)
        if future is None or not future.done():
            return None
        try:
            return future.result()
        except Exception as err:  # não deixa um erro de rede derrubar o vídeo
            log.warning("Deep Scan falhou: %s", err)
            return None

    def pending(self, key) -> bool:
        future = self._futures.get(key)
        return future is not None and not future.done()
