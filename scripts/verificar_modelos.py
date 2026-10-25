"""Verifica, peça por peça, se o motor híbrido está pronto nesta máquina (e baixa o que faltar).

    python scripts/verificar_modelos.py          (ou: iniciar verificar)

Cada etapa mostra ✅ (funcionando), ⚠️ (funcionando com o plano B) ou ❌ (não funciona), com o motivo.
Precisa de internet na primeira vez: baixa os pesos do YOLO-World, o CLIP (codificador de texto), o modelo do
Hugging Face e testa o Gemini se houver GEMINI_API_KEY.
"""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results: list[tuple[str, str, str]] = []


def step(name: str):
    def wrap(fn):
        t = time.time()
        print(f"\n▶ {name}...", flush=True)
        try:
            status, detail = fn()
        except Exception as err:
            status, detail = "❌", f"{type(err).__name__}: {err}".splitlines()[0][:300]
            if "-v" in sys.argv:
                traceback.print_exc()
        print(f"  {status} {detail} ({time.time() - t:.0f}s)", flush=True)
        results.append((status, name, detail))
        return fn

    return wrap


@step("Pacotes instalados")
def _():
    import google.genai  # noqa: F401
    import scipy  # noqa: F401
    import supervision  # noqa: F401
    import timm  # noqa: F401
    import trackers  # noqa: F401
    import transformers  # noqa: F401

    return "✅", f"supervision {supervision.__version__}, transformers {transformers.__version__}"


@step("Estágio 1: YOLO-World com vocabulário de resíduos")
def _():
    from src.open_vocab import load_open_vocab

    model, path = load_open_vocab("yolo-world")
    return "✅", f"{path.name} com {len(model.names)} descrições de resíduos"


@step("Estágio 1 (plano B): YOLOE com vocabulário de resíduos")
def _():
    from src.open_vocab import load_open_vocab, vocabulary_model_path

    if vocabulary_model_path("yolo-world").is_file():
        return "✅", "não é necessário (o YOLO-World está pronto)"
    model, path = load_open_vocab("yoloe")
    return "⚠️", f"{path.name} pronto; o sistema usa o YOLOE no lugar do YOLO-World"


@step("Estágio 2: especialista de textura do Hugging Face")
def _():
    import os

    from src.material_classifier import DEFAULT_HF_MODEL, HFTextureExpert

    model_id = os.environ.get("WASTE_HF_MODEL", DEFAULT_HF_MODEL)
    expert = HFTextureExpert(model_id)
    labels = list(expert.model.config.id2label.values())
    return "✅", f"{model_id}: {', '.join(labels[:8])}"


@step("Estágio 2 (plano B): especialista de textura local (TrashNet)")
def _():
    from src.material_classifier import LocalTextureExpert

    expert = LocalTextureExpert()
    return "✅", f"{Path(expert.model_path).name} (93,9% no teste do TrashNet)"


@step("Deep Scan com Gemini")
def _():
    import os

    import cv2

    from src.deep_scan import GeminiClient, load_env

    load_env()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return "⚠️", "sem GEMINI_API_KEY no .env: o Deep Scan usa a análise local (funciona offline)"
    image = cv2.imread(str(ROOT / "data" / "trashnet" / "glass_glass101.jpg"))
    result = GeminiClient(key, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")).scan(image)
    return "✅", f"{result.category}: {result.justification}"


@step("Motor completo numa foto (modo Ultra precisão)")
def _():
    from src.detector import Detector

    detector = Detector.from_env()
    detector.mode = "ultra"
    detections = detector.detect(ROOT / "data" / "mesa_garrafas_banana.jpg", waste_only=True)
    if not detections:
        return "❌", "nenhum resíduo encontrado na foto de exemplo"
    det = detector.deep_scan(__import__("cv2").imread(str(ROOT / "data" / "mesa_garrafas_banana.jpg")), detections[0])
    status = "⚠️" if detector.fallback_note else "✅"
    return status, (f"{detector.description}: {len(detections)} resíduo(s); Deep Scan do 1º: {det.scan.justification}"
                    + (f" | {detector.fallback_note}" if detector.fallback_note else ""))


@step("Rastreamento (ByteTrack + votação temporal) num vídeo")
def _():
    import cv2

    from src.detector import Detector

    detector = Detector.from_env()
    cap = cv2.VideoCapture(str(ROOT / "data" / "esteira.mp4"))
    ids = set()
    for f in range(15):
        ok, frame = cap.read()
        if not ok:
            break
        ids |= {d.tracker_id for d in detector.track(frame, f / 10)}
    cap.release()
    return ("✅", f"{len(ids)} objeto(s) rastreado(s) em 15 quadros") if ids else ("❌", "nenhum objeto rastreado")


print("\n" + "=" * 70)
for status, name, detail in results:
    print(f"{status} {name}")
print("=" * 70)
print("Pronto para usar." if not any(r[0] == "❌" for r in results if r[1].startswith("Motor"))
      else "O motor completo não funcionou: veja as mensagens acima (rode com -v para detalhes).")
