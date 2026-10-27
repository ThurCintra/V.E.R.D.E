"""Estágio 2 (especialista de textura + ensemble 40/60) e modo de 1 item por foto, com um especialista falso."""

import numpy as np
import pytest

from src.detector import Detection, Detector
from src.material import crop
from src.material_classifier import CATEGORY_ORDER, TexturePrediction


class FakeTexture:
    """Sempre a mesma previsão (``confidence`` na categoria, o resto dividido); registra os recortes recebidos."""

    name = "fake"
    model_path = "fake.pt"

    def __init__(self, label="glass", category="Vidro", confidence=0.9):
        rest = (1 - confidence) / (len(CATEGORY_ORDER) - 1)
        probs = {c: confidence if c == category else rest for c in CATEGORY_ORDER}
        self.prediction = TexturePrediction(label, confidence, probs)
        self.seen = []

    def predict(self, images):
        self.seen += [img.shape for img in images]
        return [self.prediction for _ in images]


@pytest.fixture
def det_with(model_path):
    def make(texture, conf=0.6, mode="alta"):
        detector = Detector(model_path, material_model=None, mode=mode)
        detector.texture, detector.material_conf = texture, conf
        return detector

    return make


def test_textura_confiante_vence_o_detector_no_ensemble(det_with, coco_samples):
    # Garrafa (forma: Plástico, ~0,7) + textura 93% vidro: 0,4×0,07 + 0,6×0,93 ≈ 0,59 Vidro contra ~0,29 Plástico
    fake = FakeTexture("glass", "Vidro", 0.93)
    detections = det_with(fake).detect(coco_samples["vinho"], waste_only=True)
    bottles = [d for d in detections if d.label == "bottle"]
    assert bottles and all(d.category == "Vidro" for d in bottles)
    assert all(d.material == "glass" and d.material_confidence == pytest.approx(0.93) for d in bottles)
    assert all(sum(d.probs.values()) == pytest.approx(1) for d in bottles)
    assert fake.seen  # recebeu os recortes


def test_textura_em_duvida_nao_vence_a_forma(det_with, coco_samples):
    detections = det_with(FakeTexture("glass", "Vidro", 0.4)).detect(coco_samples["vinho"], waste_only=True)
    bottles = [d for d in detections if d.label == "bottle"]
    assert bottles and all(d.category == "Plástico" for d in bottles)
    assert all(d.material == "glass" for d in bottles)  # mas o palpite fica registrado


def test_comida_nao_passa_pela_textura(det_with, coco_samples):
    fake = FakeTexture("metal", "Metal", 0.99)
    detections = det_with(fake).detect(coco_samples["garrafas_banana"], waste_only=True)
    bananas = [d for d in detections if d.label == "banana"]
    assert bananas and all(d.category == "Orgânico/Rejeito" and d.material is None for d in bananas)


def test_modo_rapido_nao_usa_a_textura(det_with, coco_samples):
    fake = FakeTexture("glass", "Vidro", 0.99)
    detections = det_with(fake, mode="rapido").detect(coco_samples["vinho"], waste_only=True)
    assert not fake.seen
    assert all(d.category == "Plástico" for d in detections if d.label == "bottle")


PLAIN = np.full((480, 640, 3), 200, np.uint8)  # foto lisa: o detector não acha nada


def test_modo_um_item_classifica_a_foto_inteira(det_with):
    detector = det_with(FakeTexture("cardboard", "Papel", 0.88))
    assert detector.detect(PLAIN, waste_only=True, whole_image_fallback=False) == []
    (whole,) = detector.detect(PLAIN, waste_only=True, single_item=True)
    assert whole.category == "Papel" and whole.label == "cardboard"
    assert whole.box == (0.0, 0.0, 640.0, 480.0)
    assert whole.top_probs(1) == [("Papel", pytest.approx(0.88))]


def test_modo_um_item_a_foto_inteira_decide_o_material(det_with, coco_samples):
    # Na foto da mesa o COCO vê garrafas, bananas e bolo; no modo lixeira o item é um só, e a textura da foto
    # inteira decide a categoria. O detector dá a posição do objeto mais confiante.
    plain = det_with(FakeTexture("cardboard", "Papel", 0.9))
    many = plain.detect(coco_samples["garrafas_banana"], waste_only=True)
    (item,) = plain.detect(coco_samples["garrafas_banana"], waste_only=True, single_item=True)
    best = max(many, key=lambda d: d.confidence)
    assert item.category == "Papel" and item.material == "cardboard"
    assert item.box == best.box and item.label == best.label


def test_modo_um_item_nao_inventa_quando_a_textura_duvida(det_with):
    detector = det_with(FakeTexture("cardboard", "Papel", 0.3))
    assert detector.detect(PLAIN, waste_only=True, single_item=True) == []
    assert detector.detect(PLAIN, waste_only=True) == []


def test_foto_de_perto_sem_deteccao_usa_a_foto_inteira(det_with):
    # Sem "1 objeto por foto": se o detector não achar nada, a foto inteira entra como plano B
    (whole,) = det_with(FakeTexture("metal", "Metal", 0.9)).detect(PLAIN, waste_only=True)
    assert whole.category == "Metal" and whole.box == (0.0, 0.0, 640.0, 480.0)


def test_descricao_reflete_os_estagios(det_with):
    assert det_with(FakeTexture()).description == "coco + textura (fake)"
    assert det_with(FakeTexture(), mode="rapido").description == "coco"


def test_recorte_com_margem_dentro_da_imagem():
    image = np.zeros((100, 200, 3), np.uint8)
    assert crop(image, (50, 20, 150, 80)).shape == (72, 120, 3)  # 10% de margem
    assert crop(image, (0, 0, 200, 100)).shape == (100, 200, 3)  # não passa da borda


def test_configuracao_por_ambiente(monkeypatch, model_path):
    monkeypatch.setenv("WASTE_MODEL", str(model_path))
    monkeypatch.setenv("WASTE_IMGSZ", "960")
    monkeypatch.setenv("DETECTOR_CONF", "0.4")
    monkeypatch.setenv("WASTE_TTA", "1")
    monkeypatch.setenv("WASTE_MATERIAL_MODEL", "off")
    detector = Detector.from_env()
    assert (detector.imgsz, detector.conf, detector.augment, detector.material) == (960, 0.4, True, None)


def test_to_dict_inclui_o_material():
    d = Detection("bottle", 0.8, (0, 0, 1, 1), "Vidro", 250, "glass", 0.91)
    assert d.to_dict()["material"] == {"label": "glass", "confidence": 0.91}
    assert Detection("bottle", 0.8, (0, 0, 1, 1), "Plástico", 30).to_dict()["material"] is None


# ---------------------------------------------------------------- classificador real (models/material_cls.pt)
def test_classificador_treinado_tem_as_classes_do_trashnet(material_model):
    from src.material import MaterialClassifier

    names = set(MaterialClassifier(material_model).class_names.values())
    assert names == {"cardboard", "glass", "metal", "paper", "plastic", "trash"}


def test_classificador_treinado_acerta_as_amostras(material_model):
    """Fotos do conjunto de *teste* do TrashNet (o modelo não as viu no treino), nomeadas <classe>_<original>.jpg."""
    import cv2

    from src.material import MaterialClassifier
    from src.paths import DATA_DIR

    samples = sorted((DATA_DIR / "trashnet").glob("*.jpg"))
    assert len(samples) == 6
    preds = MaterialClassifier(material_model).classify([cv2.imread(str(p)) for p in samples])
    got = {p.stem.split("_")[0]: pred.label for p, pred in zip(samples, preds)}
    assert sum(expected == label for expected, label in got.items()) >= 5, got
