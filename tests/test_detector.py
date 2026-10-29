import cv2
import numpy as np
import pytest

from src.detector import Detection, Detector, annotate, load_image


def test_modelo_baixado_em_models(model_path):
    assert model_path.name == "yolov8n.pt"
    assert model_path.parent.name == "models"
    assert model_path.stat().st_size > 5_000_000


def test_carrega_modelo_com_80_classes_coco(detector):
    assert len(detector.class_names) == 80
    assert detector.class_names[0] == "person"
    assert detector.backend == "coco" and detector.texture is None


def test_modelo_inexistente_da_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model"):
        Detector(tmp_path / "nao-existe.pt")


def test_detecta_pessoas_e_onibus_na_foto_de_exemplo(detector, sample_image):
    detections = detector.detect(sample_image)
    labels = [d.label for d in detections]
    assert labels.count("person") >= 3
    assert "bus" in labels
    # Pessoas e ônibus não são resíduo
    assert all(d.category is None for d in detections if d.label in ("person", "bus"))


def test_so_residuos_esconde_pessoas_e_onibus(detector, sample_image):
    assert detector.detect(sample_image, waste_only=True) == []


def test_garrafas_e_bananas_viram_plastico_e_organico(detector, coco_samples):
    detections = detector.detect(coco_samples["garrafas_banana"], waste_only=True)
    by_label = {d.label: d.category for d in detections}
    assert by_label["bottle"] == "Plástico"
    assert by_label["banana"] == "Orgânico/Rejeito"
    assert all(d.weight_g > 0 and d.color_hex for d in detections)


def test_taca_vira_vidro(detector, coco_samples):
    detections = detector.detect(coco_samples["vinho"], waste_only=True)
    glass = [d for d in detections if d.label == "wine glass"]
    assert glass and glass[0].category == "Vidro"


def test_formato_das_deteccoes(detector, sample_image):
    h, w = cv2.imread(str(sample_image)).shape[:2]
    for d in detector.detect(sample_image):
        assert isinstance(d, Detection)
        assert detector.conf <= d.confidence <= 1
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w + 1
        assert 0 <= y1 < y2 <= h + 1


def test_aceita_caminho_bytes_e_array(detector, coco_samples):
    path = coco_samples["garrafas_banana"]
    by_path = sorted(d.label for d in detector.detect(path))
    assert sorted(d.label for d in detector.detect(cv2.imread(str(path)))) == by_path
    assert sorted(d.label for d in detector.detect(path.read_bytes())) == by_path


def test_imagem_sintetica_roda_sem_erro(detector, synthetic_image):
    detections = detector.detect(synthetic_image)
    assert all(d.label not in ("person", "bus") for d in detections)


def test_entradas_invalidas(detector, tmp_path):
    with pytest.raises(FileNotFoundError):
        detector.detect(tmp_path / "nada.jpg")
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        load_image(b"isto nao e uma imagem")


def test_annotate_pinta_com_a_cor_da_categoria():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    out = annotate(image, [Detection("bottle", 0.9, (20, 40, 180, 190), "Plástico", 30)])
    assert image.sum() == 0  # não altera a original
    assert tuple(out[120, 20]) == (0x28, 0x28, 0xD6)  # borda vermelha (BGR de #D62828)
