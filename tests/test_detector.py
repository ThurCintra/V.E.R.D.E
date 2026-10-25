import cv2
import numpy as np
import pytest

from src.detector import Detection, Detector


def test_modelo_baixado_em_models(model_path):
    assert model_path.name == "yolov8n.pt"
    assert model_path.parent.name == "models"
    assert model_path.stat().st_size > 5_000_000


def test_carrega_modelo_com_80_classes_coco(detector):
    assert len(detector.class_names) == 80
    assert detector.class_names[0] == "person"


def test_modelo_inexistente_da_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model"):
        Detector(tmp_path / "nao-existe.pt")


def test_detecta_pessoas_e_onibus_na_foto_de_exemplo(detector, sample_image):
    detections = detector.detect(sample_image)
    labels = [d.label for d in detections]
    assert labels.count("person") >= 3
    assert "bus" in labels


def test_formato_das_deteccoes(detector, sample_image):
    h, w = cv2.imread(str(sample_image)).shape[:2]
    for d in detector.detect(sample_image):
        assert isinstance(d, Detection)
        assert detector.conf <= d.confidence <= 1
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w + 1
        assert 0 <= y1 < y2 <= h + 1


def test_aceita_array_bgr_igual_ao_caminho(detector, sample_image):
    by_path = detector.detect(sample_image)
    by_array = detector.detect(cv2.imread(str(sample_image)))
    assert sorted(d.label for d in by_path) == sorted(d.label for d in by_array)


def test_imagem_sintetica_roda_sem_erro(detector, synthetic_image):
    detections = detector.detect(synthetic_image)
    assert isinstance(detections, list)
    assert all(d.label not in ("person", "bus") for d in detections)


def test_imagem_inexistente(detector, tmp_path):
    with pytest.raises(FileNotFoundError):
        detector.detect(tmp_path / "nada.jpg")


def test_array_invalido(detector):
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))
