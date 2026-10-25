from src.paths import MODELS_DIR
from src.weights import resolve


def test_resolve_nome_e_caminho(model_path):
    assert resolve("yolov8n.pt") == MODELS_DIR / "yolov8n.pt"
    assert resolve(model_path) == model_path


def test_resolve_nao_inventa_pesos_desconhecidos(tmp_path):
    missing = tmp_path / "meu.pt"
    assert resolve(missing) == missing and not missing.exists()
