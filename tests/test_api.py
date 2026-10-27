import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.review import ReviewCollector


@pytest.fixture
def client(detector, db, tmp_path):
    app = create_app(detector=detector, db=db, review=ReviewCollector(tmp_path / "review", min_interval_s=0))
    with TestClient(app) as c:
        yield c


def post_image(client, path, **params):
    with open(path, "rb") as f:
        return client.post("/detect", params=params, files={"file": (path.name, f, "image/jpeg")})


def test_health_e_categorias(client):
    health = client.get("/health").json()
    assert health == {
        "status": "ok", "model": "yolov8n.pt", "detector": "coco", "mode": "alta", "texture": None,
        "pipeline": "coco",
    }
    cats = client.get("/categories").json()
    assert [c["color"] for c in cats] == ["vermelha", "azul", "amarela", "verde", "marrom"]


def test_detect_devolve_classificacao_coordenadas_e_lixeira(client, coco_samples, db):
    r = post_image(client, coco_samples["garrafas_banana"])
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "coco" and body["mode"] == "alta"
    assert body["image"] == {"width": 480, "height": 640}
    assert body["count"] == len(body["detections"]) > 0
    first = body["detections"][0]
    assert set(first) == {
        "label", "category", "confidence", "box", "color", "weight_g", "material", "probs", "tracker_id",
        "deep_scan",
    }
    assert set(first["box"]) == {"x1", "y1", "x2", "y2"}

    action = body["action"]
    best = max(body["detections"], key=lambda d: d["confidence"])
    assert action["category"] == best["category"] and action["label"] == best["label"]
    assert action["bin"].startswith(best["category"])
    assert len(db.records()) == body["count"]


def test_detect_sem_gravar(client, coco_samples, db):
    body = post_image(client, coco_samples["vinho"], save=False).json()
    assert body["action"] is not None
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


def test_casos_duvidosos_vao_para_revisao(client, coco_samples):
    # Na foto da mesa há bananas com 25-41% de confiança: abaixo de 50%, viram caso para revisar
    body = post_image(client, coco_samples["garrafas_banana"]).json()
    assert body["review"]["needed"] and body["review"]["saved"]
    assert any("baixa confiança" in r for r in body["review"]["reasons"])
    assert client.get("/review").json() == {"enabled": True, "count": 1}
    export = client.get("/review/export")
    assert export.status_code == 200 and export.headers["content-type"] == "application/zip"


def test_foto_sem_duvida_nao_vai_para_revisao(client, sample_image):
    body = post_image(client, sample_image).json()  # ônibus e pessoas: nenhum resíduo, nada duvidoso
    assert body["review"] == {"needed": False, "reasons": [], "saved": False}
    assert client.get("/review").json()["count"] == 0
