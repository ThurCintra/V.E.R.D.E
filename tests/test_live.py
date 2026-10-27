"""Câmera fluida: aceleração com OpenVINO, textura por trilha e análise em segundo plano."""

import time

import cv2
import pytest

from src import accel
from src.detector import Detector
from src.live import CameraReader, LiveSession
from src.pipeline import WastePipeline
from src.paths import DATA_DIR, MODELS_DIR
from tests.test_material import FakeTexture

EXAMPLE = DATA_DIR / "mesa_garrafas_banana.jpg"


def summary(detections):
    return [(d.label, d.category, round(d.confidence, 2)) for d in detections]


@pytest.fixture(scope="module")
def yoloe_path():
    path = MODELS_DIR / "verde_yoloe.pt"
    if not path.is_file() or not accel.openvino_dir(path).is_dir() or not EXAMPLE.is_file():
        pytest.skip("modelo YOLOE (e a versão OpenVINO) não estão em models/")
    return path


def test_openvino_da_as_mesmas_deteccoes_do_pytorch(monkeypatch, yoloe_path):
    pytest.importorskip("openvino")
    fast = Detector(None, material_model=None)
    assert fast.model_path == yoloe_path
    assert fast.accelerator == "openvino" and "OpenVINO" in fast.description
    monkeypatch.setenv("WASTE_OPENVINO", "0")
    slow = Detector(None, material_model=None)
    assert slow.accelerator == "pytorch"
    assert fast.class_names == slow.class_names
    found = summary(fast.detect(EXAMPLE, waste_only=True))
    assert found and found == summary(slow.detect(EXAMPLE, waste_only=True))


def test_sem_versao_openvino_usa_o_pytorch(model_path):
    assert not accel.openvino_dir(model_path).exists()
    assert Detector(model_path, material_model=None).accelerator == "pytorch"


def test_textura_por_trilha_a_cada_5_quadros(model_path, conveyor_video):
    detector = Detector(model_path, material_model=None)
    detector.texture = texture = FakeTexture()
    cap = cv2.VideoCapture(str(conveyor_video))
    shown = 0
    for f in range(12):
        ok, frame = cap.read()
        tracked = detector.track(frame, f / 10)
        refinable = [d for d in tracked if detector._refinable(d)]
        assert all(d.material == "glass" for d in refinable)  # o voto de textura vale mesmo nos quadros sem olhar
        shown += len(refinable)
    cap.release()
    assert shown and len(texture.seen) < shown / 2  # olhou bem menos vezes do que mostrou


def test_camera_que_nao_abre():
    with pytest.raises(ValueError):
        CameraReader("camera-que-nao-existe.mp4")


class SlowCamera:
    """Um vídeo lido no ritmo de uma câmera de verdade (10 quadros por segundo)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))

    def read(self):
        time.sleep(0.1)
        return self.cap.read()

    def release(self):
        self.cap.release()


def test_sessao_ao_vivo_analisa_em_paralelo(model_path, conveyor_video):
    pipeline = WastePipeline(Detector(model_path, material_model=None))
    session = LiveSession(pipeline, SlowCamera(conveyor_video))
    try:
        deadline = time.monotonic() + 60
        while session.running and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not session.running and session.error is None
        assert session.analyzed > 0 and session.result is not None and session.result.annotated is None
        view = session.view()
        assert view is not None and view.shape[2] == 3
        assert sum(pipeline.counts.values()) >= 1
    finally:
        session.stop()
