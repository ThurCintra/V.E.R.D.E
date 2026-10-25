import shutil

import pytest

from src.detector import Detection
from src.evaluate import NONE, evaluate_system, labeled_images, metrics, plot_confusion


def test_metricas_e_matriz_de_confusao():
    truth = ["Vidro", "Vidro", "Plástico", "Plástico"]
    pred = ["Vidro", "Plástico", "Plástico", NONE]
    m = metrics(truth, pred, ["Plástico", "Vidro"])
    assert m["accuracy"] == 0.5
    assert m["predicted_labels"] == ["Plástico", "Vidro", NONE]
    assert m["confusion_matrix"] == [[1, 0, 1], [1, 1, 0]]
    assert m["per_class"]["Vidro"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(0.6667, abs=1e-4), "support": 2}


def test_grafico_da_matriz(tmp_path):
    m = metrics(["a", "b"], ["a", "a"], ["a", "b"])
    path = plot_confusion(m, "teste", tmp_path / "m.png")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_avaliacao_do_sistema_numa_pasta_rotulada(tmp_path, coco_samples):
    for cls in ("glass", "plastic"):
        (tmp_path / cls).mkdir()
        shutil.copy(coco_samples["vinho"], tmp_path / cls / "a.jpg")

    class AlwaysGlass:
        mode = "teste"

        def detect(self, path, waste_only, single_item):
            return [Detection("bottle", 0.9, (0, 0, 1, 1), "Vidro", 250)]

    assert [c for _, c in labeled_images(tmp_path)] == ["glass", "plastic"]
    result = evaluate_system(AlwaysGlass(), tmp_path)
    assert result["accuracy"] == 0.5 and result["recognized"] == 1.0
    assert result["per_class"]["Vidro"]["recall"] == 1.0 and result["per_class"]["Plástico"]["recall"] == 0.0
