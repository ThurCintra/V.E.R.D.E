import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.hardware_controller import HardwareController, SimulatedBackend


@pytest.fixture
def backend():
    return SimulatedBackend()


@pytest.fixture
def client(detector, db, backend):
    app = create_app(detector=detector, db=db, controller=HardwareController(backend, cooldown_s=0))
    with TestClient(app) as c:
        yield c


def post_image(client, path, **params):
    with open(path, "rb") as f:
        return client.post("/detect", params=params, files={"file": (path.name, f, "image/jpeg")})


def test_health_e_categorias(client):
    health = client.get("/health").json()
    assert health == {"status": "ok", "model": "yolov8n.pt", "mode": "coco-heuristico", "hardware": "sim"}
    cats = client.get("/categories").json()
    assert [c["command"] for c in cats] == ["P", "C", "M", "V", "O"]


def test_detect_devolve_classificacao_coordenadas_e_comando(client, coco_samples, db, backend):
    r = post_image(client, coco_samples["garrafas_banana"], actuate=True)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "coco-heuristico"
    assert body["image"] == {"width": 480, "height": 640}
    assert body["count"] == len(body["detections"]) > 0
    first = body["detections"][0]
    assert set(first) == {"label", "category", "confidence", "box", "color", "weight_g", "command"}
    assert set(first["box"]) == {"x1", "y1", "x2", "y2"}

    action = body["action"]
    best = max(body["detections"], key=lambda d: d["confidence"])
    assert action["category"] == best["category"] and action["command"] == best["command"]
    assert action["sent"] is True
    assert backend.sent == [action["command"]]
    assert len(db.records()) == body["count"]


def test_detect_sem_acionar_nem_gravar(client, coco_samples, db, backend):
    body = post_image(client, coco_samples["vinho"], save=False).json()
    assert body["action"]["sent"] is False
    assert backend.sent == []
    assert db.records() == []


def test_imagem_sem_residuo(client, sample_image):
    body = post_image(client, sample_image).json()
    assert body["count"] == 0 and body["action"] is None
    all_objects = post_image(client, sample_image, waste_only=False).json()
    assert {d["label"] for d in all_objects["detections"]} >= {"person", "bus"}


def test_arquivo_invalido_da_400(client):
    r = client.post("/detect", files={"file": ("x.jpg", b"nao e imagem", "image/jpeg")})
    assert r.status_code == 400


def test_impacto_reflete_o_historico(client, coco_samples):
    assert client.get("/impact").json()["total"]["items"] == 0
    count = post_image(client, coco_samples["garrafas_banana"]).json()["count"]
    impact = client.get("/impact", params={"days": 7}).json()
    assert impact["total"]["items"] == count
    assert impact["total"]["co2_kg"] > 0
