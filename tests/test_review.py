import json
import zipfile
import io

import numpy as np

from src.detector import Detection
from src.review import ReviewCollector, review_reasons


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def det(conf=0.9, category="Plástico", material=None, material_conf=None, box=(10, 20, 110, 220)):
    return Detection("bottle", conf, box, category, 30, material, material_conf)


def test_motivos():
    assert review_reasons([det(0.9)]) == []
    assert review_reasons([det(0.3)]) == ["baixa confiança: bottle 30%"]
    assert review_reasons([det(0.9, material="glass", material_conf=0.45)]) == ["material incerto: glass 45%"]
    assert review_reasons([], single_item=True) == ["nenhum resíduo reconhecido"]
    assert review_reasons([]) == []  # sem modo lixeira, foto vazia é normal
    assert review_reasons([Detection("person", 0.2, (0, 0, 1, 1))]) == []  # não é resíduo


def test_grava_imagem_rotulos_yolo_e_metadados(tmp_path):
    collector = ReviewCollector(tmp_path, min_interval_s=0)
    image = np.zeros((400, 200, 3), np.uint8)
    path = collector.maybe_save(image, [det(0.3), Detection("person", 0.9, (0, 0, 5, 5))], source="video")
    assert path and path.is_file()
    (label_line,) = path.with_suffix(".txt").read_text().splitlines()
    assert label_line == "0 0.300000 0.300000 0.500000 0.500000"  # Plástico, centro e tamanho normalizados
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["source"] == "video" and meta["reasons"] == ["baixa confiança: bottle 30%"]
    assert (tmp_path / "classes.txt").read_text(encoding="utf-8").splitlines()[0] == "Plástico"
    assert collector.count() == 1


def test_nao_grava_sem_motivo(tmp_path):
    assert ReviewCollector(tmp_path).maybe_save(np.zeros((10, 10, 3), np.uint8), [det(0.95)]) is None
    assert not any(tmp_path.iterdir())


def test_intervalo_minimo_e_limite_de_disco(tmp_path):
    clock = Clock()
    collector = ReviewCollector(tmp_path, min_interval_s=2, max_items=2, clock=clock)
    image = np.zeros((10, 10, 3), np.uint8)
    assert collector.maybe_save(image, [det(0.3)])
    assert collector.maybe_save(image, [det(0.3)]) is None  # frame seguinte do mesmo vídeo
    clock.t = 3
    assert collector.maybe_save(image, [det(0.3)])
    clock.t = 6
    assert collector.maybe_save(image, [det(0.3)]) is None  # chegou no limite
    assert collector.count() == 2


def test_exporta_zip(tmp_path):
    collector = ReviewCollector(tmp_path, min_interval_s=0)
    collector.maybe_save(np.zeros((10, 10, 3), np.uint8), [det(0.3)])
    names = zipfile.ZipFile(io.BytesIO(collector.export_zip())).namelist()
    assert "classes.txt" in names and sum(n.endswith(".jpg") for n in names) == 1


def test_desligado_por_ambiente(monkeypatch):
    monkeypatch.setenv("WASTE_REVIEW", "0")
    assert ReviewCollector.from_env() is None
