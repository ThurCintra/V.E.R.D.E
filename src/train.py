"""Fine-tuning do YOLOv8 para resíduos recicláveis.

Fontes suportadas:

- **TACO** (http://tacodataset.org): 1.500 fotos de lixo anotadas em formato COCO com 60 classes. As classes
  são convertidas para as 5 categorias da coleta seletiva (``category_from_name``), então o modelo treinado
  já sai com Plástico, Papel, Metal, Vidro e Orgânico/Rejeito.
- **Roboflow Universe**: qualquer dataset de detecção exportado em ``yolov8`` (precisa de ``pip install
  roboflow`` e da chave em ``ROBOFLOW_API_KEY``). As classes do dataset são mantidas; o detector as converte
  para categoria pelo nome.
- **Um data.yaml seu** (exportado do Ultralytics HUB, CVAT, Label Studio…).
- **TrashNet** (github.com/garythung/trashnet, MIT): 2.527 fotos de um objeto cada, sem caixas. Treina o
  *classificador de material* (``models/material_cls.pt``), a segunda etapa do detector: ele olha o recorte de
  cada objeto e decide se é plástico, papel, metal, vidro ou rejeito. Roda bem em CPU.

Exemplos::

    python -m src.train taco --epochs 50
    python -m src.train coco-json --annotations taco/annotations.json --images taco/ --epochs 50
    python -m src.train roboflow --workspace minha-org --project lixo --version 3
    python -m src.train yaml --data datasets/meu/data.yaml
    python -m src.train trashnet --epochs 30          # classificador de material

Ao terminar, o melhor peso é copiado para ``models/waste_detector.pt`` (ou ``models/material_cls.pt``) e o
detector passa a usá-lo sozinho.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from src.paths import CUSTOM_MODEL, DEFAULT_MODEL, MATERIAL_MODEL, MODELS_DIR, ROOT
from src.recycling import CATEGORIES, category_from_name

DATASETS_DIR = ROOT / "datasets"
CATEGORY_NAMES = list(CATEGORIES)  # ordem fixa das classes do modelo treinado
TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"


def convert_coco_to_yolo(
    annotations_json: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 0,
    max_size: int = 1280,
) -> Path:
    """Converte um dataset em formato COCO (como o TACO) para YOLO, com as 5 categorias. Devolve o data.yaml.

    As fotos são regravadas com no máximo ``max_size`` px no lado maior (as do TACO chegam a 4000 px, o que deixa
    o treino lento sem ganho a 640 px) e já na orientação do EXIF, que é a das anotações. Fotos cujo tamanho não
    bate com o das anotações são puladas: dariam caixas no lugar errado.
    """
    import cv2

    annotations_json, images_dir, out_dir = Path(annotations_json), Path(images_dir), Path(out_dir)
    coco = json.loads(annotations_json.read_text(encoding="utf-8"))

    class_of = {c["id"]: CATEGORY_NAMES.index(category_from_name(c["name"]).name) for c in coco["categories"]}
    boxes_by_image: dict[int, list[str]] = {}
    images = {img["id"]: img for img in coco["images"]}
    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        iw, ih = img["width"], img["height"]
        cx, cy = (x + w / 2) / iw, (y + h / 2) / ih
        boxes_by_image.setdefault(img["id"], []).append(
            f"{class_of[ann['category_id']]} {cx:.6f} {cy:.6f} {w / iw:.6f} {h / ih:.6f}"
        )

    ids = sorted(i for i in boxes_by_image if (images_dir / images[i]["file_name"]).is_file())
    if not ids:
        raise FileNotFoundError(f"Nenhuma imagem anotada encontrada em {images_dir}")
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction)) if len(ids) > 1 else 0
    splits = {"val": ids[:n_val], "train": ids[n_val:]}

    written = {split: 0 for split in splits}
    skipped = 0
    for split, split_ids in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_id in split_ids:
            info = images[image_id]
            pixels = cv2.imread(str(images_dir / info["file_name"]))  # aplica a rotação do EXIF
            if pixels is None or pixels.shape[:2] != (info["height"], info["width"]):
                skipped += 1
                continue
            scale = max_size / max(pixels.shape[:2])
            if scale < 1:
                pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # TACO guarda as fotos em subpastas (batch_1/000001.jpg): o nome achatado evita colisão
            stem = info["file_name"].replace("/", "_").rsplit(".", 1)[0]
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), pixels, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(boxes_by_image[image_id]) + "\n")
            written[split] += 1
    if skipped:
        print(f"{skipped} foto(s) puladas: ilegíveis ou com tamanho diferente do das anotações")
    if not written["train"]:
        raise ValueError("Nenhuma foto válida para treino")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val" if written["val"] else "images/train",
                "names": dict(enumerate(CATEGORY_NAMES)),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def download_taco(out_dir: Path = DATASETS_DIR / "taco_raw", workers: int = 8) -> tuple[Path, Path]:
    """Baixa as anotações do TACO (GitHub) e as fotos (Flickr). Devolve (annotations.json, pasta das fotos)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations = out_dir / "annotations.json"
    if not annotations.is_file():
        urllib.request.urlretrieve(TACO_ANNOTATIONS_URL, annotations)
    coco = json.loads(annotations.read_text(encoding="utf-8"))

    def fetch(img: dict) -> None:
        dest = out_dir / img["file_name"]
        if dest.is_file():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = img.get("flickr_url") or img.get("flickr_640_url")
        try:
            urllib.request.urlretrieve(url, dest)
        except OSError as err:
            print(f"Falhou {url}: {err}")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, coco["images"]))
    return annotations, out_dir


def download_roboflow(workspace: str, project: str, version: int, api_key: str | None = None) -> Path:
    """Baixa um dataset do Roboflow no formato yolov8. Devolve o data.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError as err:
        raise SystemExit("Instale o cliente do Roboflow: pip install roboflow") from err
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Defina ROBOFLOW_API_KEY (ou passe --api-key)")
    dataset = (
        Roboflow(api_key=api_key)
        .workspace(workspace)
        .project(project)
        .version(version)
        .download("yolov8", location=str(DATASETS_DIR / f"{project}-v{version}"))
    )
    return Path(dataset.location) / "data.yaml"


def prepare_trashnet(
    out_dir: str | Path = DATASETS_DIR / "trashnet",
    zip_path: str | Path | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> Path:
    """Baixa (ou usa ``zip_path``) o TrashNet e separa treino/validação/teste por classe, no formato de pastas
    do ``yolo classify``. Devolve a pasta do dataset. Não refaz o que já existe."""
    import zipfile

    out_dir = Path(out_dir)
    if (out_dir / "test").is_dir():
        return out_dir
    if zip_path is None:
        zip_path = DATASETS_DIR / "trashnet.zip"
        if not zip_path.is_file():
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(TRASHNET_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        by_class: dict[str, list[str]] = {}
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) == 3 and parts[2].lower().endswith(".jpg") and not parts[2].startswith("."):
                by_class.setdefault(parts[1], []).append(name)

        rng = random.Random(seed)
        for cls, names in sorted(by_class.items()):
            names = sorted(names)
            rng.shuffle(names)
            n_test, n_val = round(len(names) * test_fraction), round(len(names) * val_fraction)
            splits = {"test": names[:n_test], "val": names[n_test : n_test + n_val], "train": names[n_test + n_val :]}
            for split, members in splits.items():
                dest = out_dir / split / cls
                dest.mkdir(parents=True, exist_ok=True)
                for member in members:
                    (dest / Path(member).name).write_bytes(archive.read(member))
    return out_dir


def train_classifier(
    data_dir: str | Path,
    base_model: str | Path = MODELS_DIR / "yolov8n-cls.pt",
    epochs: int = 30,
    imgsz: int = 224,
    batch: int = 32,
    device: str | None = None,
    project_dir: str | Path = ROOT / "runs",
    output: str | Path = MATERIAL_MODEL,
    name: str = "material",
    on_epoch=None,
    **overrides,
) -> Path:
    """Fine-tuning do classificador de material. Copia o melhor peso para ``output`` e devolve esse caminho.

    ``on_epoch(época, total)`` é chamado ao fim de cada época (barra de progresso do dashboard).
    """
    from ultralytics import YOLO

    model = YOLO(str(base_model))
    if on_epoch:
        model.add_callback("on_train_epoch_end", lambda trainer: on_epoch(trainer.epoch + 1, trainer.epochs))
    model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(Path(project_dir).resolve()),
        name=name,
        exist_ok=True,
        **overrides,
    )
    return _copy_best(model, output)


def _copy_best(model, output: str | Path) -> Path:
    best = Path(model.trainer.best)
    if not best.is_file():
        best = Path(model.trainer.last)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, output)
    return output


def train(
    data_yaml: str | Path,
    base_model: str | Path = DEFAULT_MODEL,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project_dir: str | Path = ROOT / "runs",
    output: str | Path = CUSTOM_MODEL,
    **overrides,
) -> Path:
    """Faz o fine-tuning e copia o melhor peso para ``output``. Devolve o caminho de ``output``."""
    from ultralytics import YOLO

    model = YOLO(str(base_model))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(Path(project_dir).resolve()),
        name="waste",
        exist_ok=True,
        **overrides,
    )
    return _copy_best(model, output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)

    sub.add_parser("taco", help="baixa o TACO, converte e treina")
    coco = sub.add_parser("coco-json", help="converte um dataset local em formato COCO e treina")
    coco.add_argument("--annotations", required=True)
    coco.add_argument("--images", required=True)
    rf = sub.add_parser("roboflow", help="baixa um dataset do Roboflow e treina")
    rf.add_argument("--workspace", required=True)
    rf.add_argument("--project", required=True)
    rf.add_argument("--version", type=int, required=True)
    rf.add_argument("--api-key")
    own = sub.add_parser("yaml", help="treina num data.yaml já pronto")
    own.add_argument("--data", required=True)
    trash = sub.add_parser("trashnet", help="treina o classificador de material com o TrashNet")
    trash.add_argument("--zip", help="dataset-resized.zip já baixado")

    for name, p in sub.choices.items():
        classify = name == "trashnet"
        p.add_argument("--epochs", type=int, default=30 if classify else 50)
        p.add_argument("--imgsz", type=int, default=224 if classify else 640)
        p.add_argument("--batch", type=int, default=32 if classify else 16)
        p.add_argument("--device", help="cpu, 0 (GPU), mps…")
        p.add_argument(
            "--base", default=str(MODELS_DIR / "yolov8n-cls.pt" if classify else DEFAULT_MODEL), help="peso inicial"
        )
        p.add_argument("--runs-dir", default=str(ROOT / "runs"), help="onde salvar o treino (ex.: pasta do Drive)")

    args = parser.parse_args(argv)
    if args.source == "trashnet":
        data_dir = prepare_trashnet(zip_path=args.zip)
        out = train_classifier(data_dir, args.base, args.epochs, args.imgsz, args.batch, args.device, args.runs_dir)
        print(f"Classificador de material salvo em {out}")
        return
    if args.source == "taco":
        annotations, images = download_taco()
        data = convert_coco_to_yolo(annotations, images, DATASETS_DIR / "taco")
    elif args.source == "coco-json":
        data = convert_coco_to_yolo(args.annotations, args.images, DATASETS_DIR / "coco_custom")
    elif args.source == "roboflow":
        data = download_roboflow(args.workspace, args.project, args.version, args.api_key)
    else:
        data = Path(args.data)

    out = train(data, args.base, args.epochs, args.imgsz, args.batch, args.device, args.runs_dir)
    print(f"Modelo treinado salvo em {out}")


if __name__ == "__main__":
    main()
