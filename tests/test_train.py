import json

import cv2
import numpy as np
import yaml

from src.train import CATEGORY_NAMES, convert_coco_to_yolo, train


def fake_taco(tmp_path, n_images=4):
    """Mini dataset no formato do TACO: fotos em subpasta e classes com nomes do TACO."""
    images_dir = tmp_path / "taco"
    (images_dir / "batch_1").mkdir(parents=True)
    categories = [
        {"id": 0, "name": "Plastic bottle", "supercategory": "Bottle"},
        {"id": 1, "name": "Drink can", "supercategory": "Can"},
        {"id": 2, "name": "Glass bottle", "supercategory": "Bottle"},
    ]
    images, annotations = [], []
    for i in range(n_images):
        img = np.full((96, 128, 3), 200, np.uint8)
        cv2.rectangle(img, (10, 20), (50, 80), (0, 0, 200), -1)
        cv2.rectangle(img, (70, 30), (110, 70), (0, 200, 200), -1)
        cv2.imwrite(str(images_dir / "batch_1" / f"{i:06d}.jpg"), img)
        images.append({"id": i, "file_name": f"batch_1/{i:06d}.jpg", "width": 128, "height": 96})
        annotations += [
            {"id": 2 * i, "image_id": i, "category_id": i % 3, "bbox": [10, 20, 40, 60]},
            {"id": 2 * i + 1, "image_id": i, "category_id": 1, "bbox": [70, 30, 40, 40]},
        ]
    ann = tmp_path / "annotations.json"
    ann.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return ann, images_dir


def test_converte_taco_para_yolo_com_as_5_categorias(tmp_path):
    ann, images_dir = fake_taco(tmp_path)
    data_yaml = convert_coco_to_yolo(ann, images_dir, tmp_path / "out", val_fraction=0.25)

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert list(config["names"].values()) == CATEGORY_NAMES
    train_labels = sorted((tmp_path / "out" / "labels" / "train").glob("*.txt"))
    val_labels = list((tmp_path / "out" / "labels" / "val").glob("*.txt"))
    assert len(train_labels) == 3 and len(val_labels) == 1
    assert (tmp_path / "out" / "images" / "train" / f"{train_labels[0].stem}.jpg").is_file()

    lines = (tmp_path / "out" / "labels").rglob("batch_1_000000.txt").__next__().read_text().splitlines()
    plastic, metal = lines
    assert plastic.split()[0] == str(CATEGORY_NAMES.index("Plástico"))
    assert metal.split()[0] == str(CATEGORY_NAMES.index("Metal"))
    # bbox 10,20,40,60 em 128x96 → centro (30/128, 50/96), tamanho (40/128, 60/96)
    assert [round(float(v), 4) for v in plastic.split()[1:]] == [0.2344, 0.5208, 0.3125, 0.625]


def test_fine_tuning_gera_o_modelo_customizado(tmp_path, model_path):
    from src.detector import Detector

    ann, images_dir = fake_taco(tmp_path)
    data_yaml = convert_coco_to_yolo(ann, images_dir, tmp_path / "out")
    out = train(
        data_yaml, model_path, epochs=1, imgsz=64, batch=2, device="cpu",
        project_dir=tmp_path / "runs", output=tmp_path / "waste.pt", workers=0, plots=False, verbose=False,
    )
    assert out.is_file()

    custom = Detector(out)
    assert custom.mode == "customizado"
    assert list(custom.class_names.values()) == CATEGORY_NAMES
