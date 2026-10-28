import json

import cv2
import numpy as np

from src.detector import Detection, merge_duplicates
from src.evaluate import evaluate_scenes, match_scene, scene_items
from src.train import prepare_taco_scenes
from tests.test_train import fake_taco


def det(box, category, confidence, label="x"):
    return Detection(label, confidence, box, category)


def test_junta_a_mesma_comida_contada_duas_vezes():
    # A foto de exemplo do site: a banana (dentro) e o sanduíche (fora) são o mesmo resíduo
    banana = det((104, 376, 382, 568), "Orgânico/Rejeito", 0.75, "banana")
    sanduiche = det((71, 375, 435, 639), "Orgânico/Rejeito", 0.49, "sandwich")
    garrafa = det((80, 0, 268, 375), "Plástico", 0.87, "bottle")
    assert merge_duplicates([garrafa, banana, sanduiche], 0.7) == [garrafa, banana]


def test_nao_junta_categorias_diferentes_nem_caixas_separadas():
    lata = det((0, 0, 50, 100), "Metal", 0.8)
    rotulo = det((5, 20, 45, 60), "Papel", 0.6)  # dentro, mas de outra categoria
    longe = det((200, 0, 250, 100), "Metal", 0.5)
    pessoa = det((0, 0, 60, 110), None, 0.9, "person")
    assert merge_duplicates([lata, rotulo, longe, pessoa], 0.7) == [lata, rotulo, longe, pessoa]


def test_casa_caixas_com_objetos_como_no_map():
    objects = [
        {"box": [0, 0, 100, 100], "category": "Plástico"},
        {"box": [200, 0, 300, 100], "category": "Metal"},
        {"box": [400, 0, 500, 100], "category": None},  # material desconhecido: não conta
        {"box": [0, 300, 10, 310], "category": "Vidro"},  # minúsculo: não conta
    ]
    preds = [
        ((2, 2, 98, 98), "Plástico", 0.9),  # acerto
        ((0, 0, 90, 90), "Plástico", 0.8),  # a mesma garrafa de novo: o objeto já foi usado
        ((205, 5, 295, 95), "Papel", 0.7),  # achou a lata, lixeira errada
        ((402, 0, 500, 100), "Plástico", 0.6),  # caiu no objeto sem categoria: ignorada
        ((600, 600, 700, 700), "Vidro", 0.5),  # nada ali
    ]
    m = match_scene(preds, objects, image_area=1000 * 1000, min_area=0.0005)
    assert (m["tp"], m["wrong"], m["fp"], m["objects"]) == (1, 1, 2, 2)
    assert m["pairs"] == [("Plástico", "Plástico"), ("Metal", "Papel")]
    assert m["truth_bins"] == {"Plástico", "Metal"}


class FakeDetector:
    min_area = 0.0

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def detect(self, image, waste_only=False):
        return next(self.outputs)


def test_avaliacao_de_cenas_soma_precisao_revocacao_e_lixeiras(tmp_path):
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((100, 100, 3), np.uint8))
    items = [(img, [{"box": [0, 0, 50, 50], "category": "Metal"}, {"box": [50, 50, 100, 100], "category": "Papel"}]),
             (img, [{"box": [0, 0, 100, 100], "category": "Vidro"}])]
    detector = FakeDetector([
        [det((0, 0, 50, 50), "Metal", 0.9), det((60, 0, 100, 40), "Plástico", 0.4)],  # 1 acerto, 1 caixa a mais
        [det((0, 0, 100, 100), "Vidro", 0.8)],  # acerto
    ])
    r = evaluate_scenes(detector, items)
    assert (r["precision"], r["recall"]) == (round(2 / 3, 4), round(2 / 3, 4))
    assert r["category_accuracy"] == 1.0  # o que achou, achou com a lixeira certa
    assert (r["bins_precision"], r["bins_recall"]) == (round(2 / 3, 4), round(2 / 3, 4))
    assert r["boxes_per_image"] == 1.5


def test_prepara_as_cenas_do_taco(tmp_path):
    ann, images_dir = fake_taco(tmp_path, n_images=10)
    coco = json.loads(ann.read_text())
    coco["categories"].append({"id": 3, "name": "Unlabeled litter", "supercategory": "Unlabeled litter"})
    coco["annotations"].append({"id": 99, "image_id": 0, "category_id": 3, "bbox": [0, 0, 8, 8]})
    coco["images"][1]["width"] = 999  # tamanho diferente do das anotações: a foto é pulada
    (images_dir / "annotations.json").write_text(json.dumps(coco))

    index = prepare_taco_scenes(tmp_path / "cenas", max_size=64, raw_dir=images_dir)
    scenes = json.loads(index.read_text())["scenes"]
    assert len(scenes) == 9
    assert sorted({s["split"] for s in scenes}) == ["test", "train", "val"]
    first = next(s for s in scenes if s["file"] == "images/batch_1_000000.jpg")
    assert cv2.imread(str(index.parent / first["file"])).shape[:2] == (48, 64)  # 128x96 → no máximo 64 px
    assert first["objects"][0] == {"box": [5.0, 10.0, 25.0, 40.0], "category": "Plástico", "taco": "Plastic bottle"}
    assert first["objects"][-1]["category"] is None  # Unlabeled litter: material desconhecido
    assert len(scene_items(index, "train")) + len(scene_items(index, "val")) + len(scene_items(index, "test")) == 9
