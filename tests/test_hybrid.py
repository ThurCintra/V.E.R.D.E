"""Motor híbrido: vocabulário aberto, especialista do Hugging Face, ensemble, pré-processamento, votação temporal,
rastreamento e Deep Scan."""

import json

import cv2
import numpy as np
import pytest

from src.deep_scan import DeepScanner, GeminiClient, ScanResult, local_scan
from src.detector import Detector
from src.material_classifier import (
    CATEGORY_ORDER,
    HFTextureExpert,
    LocalTextureExpert,
    ensemble,
    load_texture_expert,
    shape_vote,
    to_categories,
)
from src.open_vocab import IGNORED_VOCABULARY, PROMPTS, VOCAB_CATEGORY
from src.preprocess import SceneCalibration, enhance
from src.temporal import TemporalVoter
from tests.test_material import FakeTexture


# ---------------------------------------------------------------- vocabulário
def test_vocabulario_cobre_as_5_lixeiras_e_ignora_pessoas():
    assert set(VOCAB_CATEGORY.values()) == set(CATEGORY_ORDER)
    assert len(PROMPTS) == len(set(PROMPTS))
    assert all(p not in VOCAB_CATEGORY for p in IGNORED_VOCABULARY)
    assert VOCAB_CATEGORY["milk carton"] == "Papel"  # longa vida vai no azul
    assert VOCAB_CATEGORY["aluminum can"] == "Metal"


# ---------------------------------------------------------------- ensemble
def test_classes_do_modelo_viram_categorias():
    probs = to_categories({"cardboard": 0.3, "paper": 0.2, "plastic": 0.4, "glass": 0.1})
    assert probs["Papel"] == pytest.approx(0.5) and probs["Plástico"] == pytest.approx(0.4)
    assert sum(probs.values()) == pytest.approx(1)


def test_ensemble_40_forma_60_textura():
    shape = shape_vote("Plástico", 0.8)  # 0,8 plástico, 0,05 em cada outra
    texture = {c: 0.0 for c in CATEGORY_ORDER} | {"Vidro": 1.0}
    mixed = ensemble(shape, texture)
    assert mixed["Plástico"] == pytest.approx(0.4 * 0.8)
    assert mixed["Vidro"] == pytest.approx(0.4 * 0.05 + 0.6)
    assert ensemble(shape, None) == shape


# ---------------------------------------------------------------- especialista do Hugging Face
def tiny_vit(labels):
    """ViT minúsculo, criado na hora (sem download), para testar o caminho do transformers."""
    pytest.importorskip("transformers", reason="extra opcional (iniciar extras)")
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    config = ViTConfig(
        image_size=32, patch_size=8, num_hidden_layers=1, hidden_size=32, num_attention_heads=2,
        intermediate_size=37, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )
    return ViTForImageClassification(config), ViTImageProcessor(size={"height": 32, "width": 32})


def test_especialista_hugging_face_devolve_probabilidades_por_categoria():
    model, processor = tiny_vit(["cardboard", "glass", "metal", "paper", "plastic", "trash"])
    expert = HFTextureExpert("teste/vit", model=model, processor=processor)
    crops = [np.full((50, 40, 3), v, np.uint8) for v in (30, 200)]
    preds = expert.predict(crops)
    assert len(preds) == 2
    for pred in preds:
        assert set(pred.probs) == set(CATEGORY_ORDER) and sum(pred.probs.values()) == pytest.approx(1)
        assert pred.label in {"cardboard", "glass", "metal", "paper", "plastic", "trash"}


def test_modelo_do_hugging_face_sem_rotulos_de_material_e_recusado():
    model, processor = tiny_vit(["tench", "goldfish", "shark"])  # um classificador genérico (ImageNet)
    with pytest.raises(ValueError, match="materiais"):
        HFTextureExpert("teste/imagenet", model=model, processor=processor)


def test_sem_hugging_face_usa_o_especialista_local(monkeypatch, material_model):
    monkeypatch.setenv("WASTE_HF_MODEL", "nao-existe/modelo-inexistente")
    assert isinstance(load_texture_expert(), LocalTextureExpert)
    monkeypatch.setenv("WASTE_HF_MODEL", "off")
    assert isinstance(load_texture_expert(), LocalTextureExpert)


# ---------------------------------------------------------------- pré-processamento e calibração
def test_clahe_realca_contraste_sem_mudar_o_tamanho():
    low = np.full((120, 160, 3), 120, np.uint8)
    low[40:80, 50:110] = 135  # objeto quase da cor do fundo
    out = enhance(low)
    assert out.shape == low.shape and out.dtype == np.uint8
    assert out[40:80, 50:110].mean() - out[:30].mean() > 135 - 120


def test_fundo_calibrado_separa_objeto_de_parede():
    empty = np.full((240, 320, 3), 90, np.uint8)
    cal = SceneCalibration.from_frame(empty)
    with_object = empty.copy()
    cv2.rectangle(with_object, (100, 60), (220, 180), (20, 200, 240), -1)
    assert cal.foreground_ratio(empty, (100, 60, 220, 180)) == 0
    assert cal.foreground_ratio(with_object, (100, 60, 220, 180)) > 0.9
    assert 1.2 <= cal.clip_limit <= 3.5


def test_detector_ignora_o_que_e_igual_ao_fundo(model_path, coco_samples):
    detector = Detector(model_path, material_model=None)
    image = cv2.imread(str(coco_samples["vinho"]))
    assert detector.detect(image, waste_only=True)
    detector.calibrate(image)  # a própria foto como "cena vazia": tudo nela é fundo
    assert detector.detect(image, waste_only=True) == []
    detector.calibrate(None)
    assert detector.detect(image, waste_only=True)


def test_filtro_de_tamanho(model_path, coco_samples):
    detector = Detector(model_path, material_model=None)
    everything = detector.detect(coco_samples["vinho"], waste_only=True)
    detector.max_area = 0.02
    small_only = detector.detect(coco_samples["vinho"], waste_only=True)
    assert len(small_only) < len(everything)


# ---------------------------------------------------------------- votação temporal
def probs(**kw):
    return {c: 0.0 for c in CATEGORY_ORDER} | kw


def test_categoria_nao_pisca_com_o_modelo_indeciso():
    voter = TemporalVoter()
    shown = []
    for f in range(24):
        p = probs(Plástico=0.55, Vidro=0.45) if f % 2 else probs(Plástico=0.45, Vidro=0.55)
        shown.append(voter.update(1, p, (0, 0, 10, 10), f / 15, (100, 100)).shown)
        voter.end_frame()
    assert len(set(shown)) == 1


def test_mudanca_de_verdade_troca_a_categoria():
    voter = TemporalVoter()
    for f in range(12):
        voter.update(1, probs(Plástico=0.9, Vidro=0.1), (0, 0, 10, 10), f / 15, (100, 100))
    states = [voter.update(1, probs(Plástico=0.1, Vidro=0.9), (0, 0, 10, 10), (12 + f) / 15, (100, 100)).shown
              for f in range(12)]
    assert states[0] == "Plástico" and states[-1] == "Vidro"


def test_caixa_suavizada_treme_menos():
    voter = TemporalVoter()
    rng = np.random.default_rng(0)
    raw, smooth = [], []
    for f in range(12):
        box = np.array([100, 100, 200, 200]) + rng.normal(0, 4, 4)
        state = voter.update(1, probs(Metal=1.0), box, f / 15, (640, 480))
        raw.append(box[0])
        smooth.append(voter.smoothed_box(state)[0])
    assert np.std(smooth[5:]) < np.std(raw[5:])


def test_parado_no_centro_por_1_5_segundo():
    voter = TemporalVoter(still_seconds=1.5)
    center = (290, 210, 350, 270)  # centro de 640x480
    state = None
    for f in range(20):
        state = voter.update(1, probs(Metal=0.5, Plástico=0.5), center, f * 0.1, (640, 480))
    assert voter.is_still_in_center(state)  # 1,9 s parado
    corner = TemporalVoter()
    for f in range(20):
        state = corner.update(2, probs(Metal=1.0), (0, 0, 60, 60), f * 0.1, (640, 480))
    assert not corner.is_still_in_center(state)


# ---------------------------------------------------------------- rastreamento
def test_rastreamento_mantem_o_id_e_estabiliza(model_path, conveyor_video):
    detector = Detector(model_path, material_model=None)
    cap = cv2.VideoCapture(str(conveyor_video))
    ids_per_frame = []
    for f in range(12):
        ok, frame = cap.read()
        tracked = detector.track(frame, f / 10)
        ids_per_frame.append({d.tracker_id for d in tracked})
        assert all(d.probs and d.tracker_id is not None for d in tracked)
    cap.release()
    all_ids = set().union(*ids_per_frame)
    assert all_ids and len(all_ids) <= 2 * max(len(ids) for ids in ids_per_frame)  # ids não "renascem" a cada quadro


# ---------------------------------------------------------------- Deep Scan
class FakeScanner:
    backend = "fake"

    def __init__(self):
        self.submitted, self.results = [], {}

    def submit(self, key, crop):
        self.submitted.append(key)
        self.results[key] = ScanResult("Metal", "alumínio", "Lata de alumínio amassada - Metal reciclável", 0.97, "fake")

    def result(self, key):
        return self.results.get(key)

    def scan_now(self, crop):
        return ScanResult("Vidro", "vidro âmbar", "Garrafa de vidro âmbar - Vidro reciclável", 0.9, "fake")


def test_ultra_precisao_usa_o_veredito_do_deep_scan(model_path, coco_samples):
    scanner = FakeScanner()
    detector = Detector(model_path, material_model=None, mode="ultra", deep_scanner=scanner)
    detector.texture = FakeTexture()  # o Deep Scan só existe com um especialista de textura
    detector.deep_scan_all = True
    frame = cv2.imread(str(coco_samples["vinho"]))
    for f in range(3):
        tracked = detector.track(frame, f / 10)
    assert scanner.submitted and len(scanner.submitted) == len(set(scanner.submitted))  # uma vez por objeto
    assert all(d.category == "Metal" and "alumínio" in d.scan.justification for d in tracked)
    assert tracked[0].to_dict()["deep_scan"]["source"] == "fake"


def test_deep_scan_de_uma_foto(model_path, coco_samples):
    detector = Detector(model_path, material_model=None, mode="ultra", deep_scanner=FakeScanner())
    detector.texture = FakeTexture()
    image = cv2.imread(str(coco_samples["vinho"]))
    det = detector.detect(image, waste_only=True)[0]
    scanned = detector.deep_scan(image, det)
    assert scanned.category == "Vidro" and scanned.scan.material == "vidro âmbar"


def test_analise_local_explica_o_resultado():
    result = local_scan(FakeTexture("plastic", "Plástico", 0.8), np.full((80, 60, 3), 128, np.uint8))
    assert result.category == "Plástico" and result.source == "local"
    assert "variações do recorte" in result.justification and "80% Plástico" in result.justification


def test_deep_scan_em_segundo_plano_e_fallback_quando_o_gemini_falha():
    class BrokenGemini:
        def scan(self, crop):
            raise ConnectionError("sem internet")

    scanner = DeepScanner(FakeTexture("metal", "Metal", 0.9), gemini=BrokenGemini())
    scanner.submit(7, np.full((40, 40, 3), 90, np.uint8))
    scanner.submit(7, np.full((40, 40, 3), 90, np.uint8))  # o mesmo objeto não é analisado duas vezes
    scanner._pool.shutdown(wait=True)
    result = scanner.result(7)
    assert result.category == "Metal" and result.source == "local"
    assert scanner.result(99) is None


def test_gemini_com_resposta_estruturada():
    pytest.importorskip("google.genai", reason="extra opcional (iniciar extras)")

    class FakeModels:
        def generate_content(self, model, contents, config):
            self.model, self.contents, self.config = model, contents, config
            return type("R", (), {"text": json.dumps({
                "categoria": "Plástico", "material": "PET", "confianca": 0.95,
                "justificativa": "Garrafa PET amassada, transparente - Plástico reciclável",
            })})()

    fake = type("C", (), {"models": FakeModels()})()
    result = GeminiClient("chave", client=fake).scan(np.full((64, 64, 3), 200, np.uint8))
    assert (result.category, result.material, result.source) == ("Plástico", "PET", "gemini")
    assert "PET" in result.justification
    assert fake.models.model == "gemini-2.5-flash"
    assert fake.models.config.response_mime_type == "application/json"


def test_sem_chave_do_gemini_o_deep_scan_e_local(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.deep_scan.ROOT", tmp_path)  # sem .env
    assert DeepScanner(FakeTexture()).backend == "local"


def test_duas_analises_ao_mesmo_tempo_dao_o_mesmo_resultado(model_path, coco_samples):
    """Duas abas do dashboard (ou dois pedidos da API) usando o mesmo motor ao mesmo tempo."""
    from concurrent.futures import ThreadPoolExecutor

    detector = Detector(model_path, material_model=None)
    images = [cv2.imread(str(coco_samples[k])) for k in ("vinho", "garrafas_banana")] * 4
    expected = [[(d.label, d.category) for d in detector.detect(img, waste_only=True)] for img in images]
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda img: [(d.label, d.category) for d in detector.detect(img, waste_only=True)], images))
    assert got == expected
