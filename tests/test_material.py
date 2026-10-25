"""Segunda etapa (classificador de material) e modo de 1 item por foto, com um classificador falso controlado."""

import numpy as np
import pytest

from src.detector import Detection, Detector
from src.material import MaterialPrediction, crop


class FakeClassifier:
    """Devolve sempre a mesma previsão e registra o tamanho dos recortes recebidos."""

    model_path = type("P", (), {"name": "fake.pt"})()

    def __init__(self, label="glass", category="Vidro", confidence=0.9):
        self.prediction = MaterialPrediction(label, category, confidence)
        self.seen = []

    def classify(self, images):
        self.seen += [img.shape for img in images]
        return [self.prediction for _ in images]


@pytest.fixture
def det_with(model_path):
    def make(classifier, conf=0.6):
        detector = Detector(model_path, material_model=None)
        detector.material, detector.material_conf = classifier, conf
        return detector

    return make


def test_garrafa_vira_vidro_quando_o_classificador_tem_certeza(det_with, coco_samples):
    fake = FakeClassifier("glass", "Vidro", 0.93)
    detections = det_with(fake).detect(coco_samples["vinho"], waste_only=True)
    bottles = [d for d in detections if d.label == "bottle"]
    assert bottles and all(d.category == "Vidro" and d.command == "V" for d in bottles)
    assert all(d.material == "glass" and d.material_confidence == pytest.approx(0.93) for d in bottles)
    assert fake.seen  # recebeu os recortes


def test_classificador_em_duvida_nao_muda_a_categoria(det_with, coco_samples):
    detections = det_with(FakeClassifier("glass", "Vidro", 0.4)).detect(coco_samples["vinho"], waste_only=True)
    bottles = [d for d in detections if d.label == "bottle"]
    assert bottles and all(d.category == "Plástico" for d in bottles)
    assert all(d.material == "glass" for d in bottles)  # mas o palpite fica registrado


def test_comida_nao_passa_pelo_classificador(det_with, coco_samples):
    fake = FakeClassifier("metal", "Metal", 0.99)
    detections = det_with(fake).detect(coco_samples["garrafas_banana"], waste_only=True)
    bananas = [d for d in detections if d.label == "banana"]
    assert bananas and all(d.category == "Orgânico/Rejeito" and d.material is None for d in bananas)


PLAIN = np.full((480, 640, 3), 200, np.uint8)  # foto lisa: o detector não acha nada


def test_modo_um_item_classifica_a_foto_inteira(det_with):
    detector = det_with(FakeClassifier("cardboard", "Papel", 0.88))
    assert detector.detect(PLAIN, waste_only=True) == []
    (whole,) = detector.detect(PLAIN, waste_only=True, single_item=True)
    assert whole.category == "Papel" and whole.label == "cardboard"
    assert whole.box == (0.0, 0.0, 640.0, 480.0)


def test_modo_um_item_a_foto_inteira_decide_o_material(det_with, coco_samples):
    # Na foto da mesa o COCO vê garrafas, bananas e bolo; no modo lixeira o item é um só, e o classificador
    # (que olhou a foto inteira) decide a categoria. O detector dá a posição do objeto mais confiante.
    plain = det_with(FakeClassifier("cardboard", "Papel", 0.9))
    many = plain.detect(coco_samples["garrafas_banana"], waste_only=True)
    (item,) = plain.detect(coco_samples["garrafas_banana"], waste_only=True, single_item=True)
    best = max(many, key=lambda d: d.confidence)
    assert item.category == "Papel" and item.material == "cardboard"
    assert item.box == best.box and item.label == best.label


def test_modo_um_item_nao_inventa_quando_o_classificador_duvida(det_with):
    detector = det_with(FakeClassifier("cardboard", "Papel", 0.3))
    assert detector.detect(PLAIN, waste_only=True, single_item=True) == []


def test_modo_reflete_a_segunda_etapa(det_with):
    assert det_with(FakeClassifier()).mode == "coco+material"


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
