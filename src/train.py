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
- **TrashNet + TrashBox** (comando ``material``): o TrashBox (github.com/nikhilvenkatkumsetty/TrashBox) junta
  17.785 fotos da internet em 7 classes, com fundos e ângulos variados. As fotos que repetem alguma da
  validação ou do teste saem do treino antes (``src/dedup.py``), e o teste do TrashNet continua o mesmo, para
  comparar com os modelos antigos.

Exemplos::

    python -m src.train taco --epochs 50
    python -m src.train coco-json --annotations taco/annotations.json --images taco/ --epochs 50
    python -m src.train roboflow --workspace minha-org --project lixo --version 3
    python -m src.train yaml --data datasets/meu/data.yaml
    python -m src.train trashnet --epochs 30          # classificador de material (só TrashNet)
    python -m src.train material --epochs 20          # classificador de material (TrashNet + TrashBox)

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
TRASHBOX_REPOS = {  # (repositório, pasta com as fotos); o teste e a validação ficam num repositório à parte
    "TrashBox": ("https://github.com/nikhilvenkatkumsetty/TrashBox", "TrashBox_train_set"),
    "TrashBox-testandvalid": (
        "https://github.com/nikhilvenkatkumsetty/TrashBox-testandvalid", "TrashBox_testandvalid_set",
    ),
}
# Classes do TrashBox → classes do classificador (as do TrashNet). Lixo eletrônico e hospitalar não vão para a
# coleta seletiva comum: viram "trash" (Orgânico/Rejeito), como os eletrônicos do COCO. Manter as 6 classes do
# TrashNet deixa o re-treino com as correções do usuário (que monta o treino com elas) compatível com o modelo.
TRASHBOX_CLASSES = {
    "cardboard": "cardboard", "glass": "glass", "metal": "metal", "paper": "paper", "plastic": "plastic",
    "e-waste": "trash", "medical": "trash",
}
MATERIAL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MATERIAL_BASE = "yolo11s-cls.pt"  # 75% no ImageNet (o yolov8n-cls tem 69%): generaliza melhor para fotos variadas
# AdamW com uma taxa perto da que o próprio Ultralytics escolheria num treino curto, decaindo em cosseno, e dropout
# na saída: as fotos da internet têm rótulos imperfeitos (bitucas estão em "plástico" no TrashBox)
MATERIAL_TRAINING = {"optimizer": "AdamW", "lr0": 0.0008, "cos_lr": True, "dropout": 0.1, "workers": 3}
# 2ª etapa: poucas épocas, taxa baixa, com as fotos do TrashNet repetidas. No dataset combinado elas são só 13% do
# treino, e o "trash" delas (embalagens) some no meio dos eletrônicos e do lixo hospitalar do TrashBox
MATERIAL_FINETUNE = {"epochs": 3, "lr0": 0.0002, "warmup_epochs": 0, "trashnet_repeat": 3}


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


def prepare_taco_scenes(
    out_dir: str | Path = DATASETS_DIR / "taco_cenas", max_size: int = 1280, seed: int = 0,
    raw_dir: str | Path | None = None,
) -> Path:
    """O TACO como cenas para avaliar o sistema inteiro: fotos com vários resíduos, com a caixa e a categoria de cada
    um. Devolve o ``cenas.json``. Não refaz o que já existe.

    As fotos são regravadas com no máximo ``max_size`` px no lado maior e separadas por foto em treino, validação e
    teste (60/20/20). "Unlabeled litter" fica sem categoria: o material é desconhecido, e a avaliação ignora o que
    o sistema disser desses objetos.
    """
    import cv2

    out_dir = Path(out_dir)
    index = out_dir / "cenas.json"
    if index.is_file():
        return index
    annotations, raw_dir = (Path(raw_dir) / "annotations.json", Path(raw_dir)) if raw_dir else download_taco()
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    objects: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        objects.setdefault(ann["image_id"], []).append(ann)
    ids = sorted(objects)
    random.Random(seed).shuffle(ids)
    n_train, n_val = round(len(ids) * 0.6), round(len(ids) * 0.2)
    split_of = {i: "train" if k < n_train else "val" if k < n_train + n_val else "test" for k, i in enumerate(ids)}

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    scenes, skipped = [], 0
    for info in coco["images"]:
        if info["id"] not in objects:
            continue
        pixels = cv2.imread(str(raw_dir / info["file_name"]))  # aplica a rotação do EXIF, que é a das anotações
        if pixels is None or pixels.shape[:2] != (info["height"], info["width"]):
            skipped += 1
            continue
        scale = min(1.0, max_size / max(pixels.shape[:2]))
        if scale < 1:
            pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        name = info["file_name"].replace("/", "_")
        cv2.imwrite(str(out_dir / "images" / name), pixels, [cv2.IMWRITE_JPEG_QUALITY, 92])
        items = []
        for ann in objects[info["id"]]:
            x, y, w, h = (v * scale for v in ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            taco = names[ann["category_id"]]
            items.append({
                "box": [round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)],
                "category": None if taco == "Unlabeled litter" else category_from_name(taco).name,
                "taco": taco,
            })
        scenes.append({"file": f"images/{name}", "split": split_of[info["id"]], "objects": items})
    if skipped:
        print(f"{skipped} foto(s) puladas: ilegíveis ou com tamanho diferente do das anotações")
    index.write_text(json.dumps({"scenes": scenes}, ensure_ascii=False), encoding="utf-8")
    return index


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


def download_trashbox(out_dir: str | Path = DATASETS_DIR / "trashbox_raw") -> dict[str, Path]:
    """Baixa o TrashBox com o git (só a pasta das fotos, ~2,7 GB). Devolve a pasta de cada parte do dataset.

    O repositório do treino traz as fotos duas vezes (``TrashBox_train_set`` e ``..._subfolders``): só uma é
    baixada. Não baixa de novo o que já existe.
    """
    import subprocess

    out_dir = Path(out_dir)
    for name, (url, folder) in TRASHBOX_REPOS.items():
        repo = out_dir / name
        if (repo / folder).is_dir():
            continue
        if shutil.which("git") is None:
            raise SystemExit(f"Instale o git, ou clone {url} em {repo} e rode de novo")
        out_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", url, str(repo)],
                       check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "--no-cone", f"/{folder}/"], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout"], check=True, env=env)
    test_and_val = out_dir / "TrashBox-testandvalid" / TRASHBOX_REPOS["TrashBox-testandvalid"][1]
    return {
        "train": out_dir / "TrashBox" / TRASHBOX_REPOS["TrashBox"][1],
        "val": test_and_val / "val",
        "test": test_and_val / "test",
    }


def _labeled(folder: Path, classes: dict[str, str] | None = None) -> list[tuple[Path, str, str]]:
    """(foto, classe do classificador, classe original) de uma pasta <classe>/<foto>."""
    from src.evaluate import IMAGE_SUFFIXES

    return [
        (img, classes[cls_dir.name] if classes else cls_dir.name, cls_dir.name)
        for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir() and (not classes or p.name in classes))
        for img in sorted(cls_dir.iterdir())
        if img.suffix.lower() in IMAGE_SUFFIXES
    ]


def _resized_copy(src: Path, dest: Path, short_side: int) -> bool:
    """Grava a foto com no máximo ``short_side`` px no lado menor (o treino é a 224 px: o resto só deixa a leitura
    lenta). Devolve se deu certo."""
    import cv2

    pixels = cv2.imread(str(src))
    if pixels is None:
        return False
    scale = short_side / min(pixels.shape[:2])
    if scale < 1:
        pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dest), pixels, [cv2.IMWRITE_JPEG_QUALITY, 92])


def prepare_material_dataset(
    out_dir: str | Path = DATASETS_DIR / "material",
    trashnet_dir: str | Path | None = None,
    trashbox: dict[str, Path] | None = None,
    max_per_class: int | None = 2600,
    short_side: int = 256,
    seed: int = 0,
) -> Path:
    """Junta TrashNet e TrashBox para o ``yolo classify``, sem fotos de teste ou validação vazando para o treino.

    - ``train/`` e ``val/``: as duas fontes juntas, com as 6 classes do TrashNet. Sai do treino toda foto que repete
      uma da validação ou do teste (de qualquer das fontes), e as cópias dentro do próprio treino; sai da
      validação o que repete o teste.
    - ``test_trashnet/`` e ``test_trashbox/``: os testes originais, intactos (o do TrashNet é o mesmo de sempre).
    - ``max_per_class``: limite de fotos de treino por classe. Sem ele, "trash" (eletrônicos + hospitalar) teria
      quase o dobro das outras e o modelo puxaria para o rejeito. As do TrashNet entram sempre.

    Devolve a pasta do dataset; ``info.json`` nela resume o que foi feito. Não refaz o que já existe.
    """
    from src.dedup import find_duplicates, find_repeats, readable

    out_dir = Path(out_dir)
    if (out_dir / "info.json").is_file():
        return out_dir
    trashnet = Path(trashnet_dir) if trashnet_dir else prepare_trashnet()
    trashbox = trashbox or download_trashbox()
    splits = {
        split: [(p, c, "trashnet", o) for p, c, o in _labeled(trashnet / split)]
        + [(p, c, "trashbox", o) for p, c, o in _labeled(Path(trashbox[split]), TRASHBOX_CLASSES)]
        for split in ("train", "val", "test")
    }
    info: dict = {"fontes": {"trashnet": str(trashnet), "trashbox": {k: str(v) for k, v in trashbox.items()}}}

    paths_of = {split: [item[0] for item in items] for split, items in splits.items()}
    ok = set(readable(paths_of["train"]))
    info["ilegiveis_no_treino"] = len(paths_of["train"]) - len(ok)
    leaked = find_duplicates([p for p in paths_of["train"] if p in ok], paths_of["val"] + paths_of["test"])
    repeated = find_repeats([p for p in paths_of["train"] if p in ok and p not in leaked])
    val_leaked = find_duplicates(paths_of["val"], paths_of["test"])
    info["treino_repete_validacao_ou_teste"] = len(leaked)
    info["copias_dentro_do_treino"] = len(repeated)
    info["validacao_repete_teste"] = len(val_leaked)

    train = [it for it in splits["train"] if it[0] in ok and it[0] not in leaked and it[0] not in repeated]
    if max_per_class:
        rng = random.Random(seed)
        capped = []
        for cls in MATERIAL_CLASSES:
            items = [it for it in train if it[1] == cls]
            keep = [it for it in items if it[2] == "trashnet"]
            rest = [it for it in items if it[2] != "trashnet"]
            rng.shuffle(rest)
            capped += keep + rest[: max(0, max_per_class - len(keep))]
        train = capped
    val = [it for it in splits["val"] if it[0] not in val_leaked]

    def name(item) -> str:
        path, _, source, original = item
        return f"{source}_{original}_{path.stem}".replace(" ", "_") + ".jpg"

    jobs = [(it[0], out_dir / split / it[1] / name(it)) for split, items in (("train", train), ("val", val))
            for it in items]
    with ThreadPoolExecutor(4) as pool:
        written = sum(pool.map(lambda job: _resized_copy(*job, short_side), jobs))
    for source in ("trashnet", "trashbox"):  # testes intactos, só com os nomes de classe do classificador
        for path, cls, src, original in splits["test"]:
            if src == source:
                dest = out_dir / f"test_{source}" / cls / name((path, cls, src, original))
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)

    def count(items) -> dict[str, int]:
        return {cls: sum(it[1] == cls for it in items) for cls in MATERIAL_CLASSES}

    info["fotos"] = {
        "train": count(train), "val": count(val),
        **{f"test_{src}": count([it for it in splits["test"] if it[2] == src]) for src in ("trashnet", "trashbox")},
    }
    info["gravadas"] = written
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def oversampled(data_dir: str | Path, out_dir: str | Path, prefix: str = "trashnet_", times: int = 3) -> Path:
    """Cópia do dataset (por hard link: quase não ocupa disco) com as fotos de treino que começam com ``prefix``
    repetidas ``times`` vezes. A validação fica igual."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    for split in ("train", "val"):
        for img in sorted((data_dir / split).rglob("*.jpg")):
            copies = times if split == "train" and img.name.startswith(prefix) else 1
            for k in range(copies):
                dest = out_dir / split / img.parent.name / (img.name if k == 0 else f"rep{k}_{img.name}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(img, dest)
                except OSError:  # outro disco, ou sistema de arquivos sem hard link
                    shutil.copyfile(img, dest)
    return out_dir


def train_material(
    data_dir: str | Path,
    base_model: str | Path,
    epochs: int = 20,
    imgsz: int = 224,
    batch: int = 32,
    device: str | None = None,
    project_dir: str | Path = ROOT / "runs",
    output: str | Path = MATERIAL_MODEL,
) -> Path:
    """O classificador de material em duas etapas: o dataset combinado inteiro e, depois, o ajuste fino com o
    TrashNet repetido (``MATERIAL_FINETUNE``). Copia o resultado para ``output`` e devolve esse caminho."""
    project_dir = Path(project_dir)
    first = train_classifier(
        data_dir, base_model, epochs, imgsz, batch, device, project_dir, output=project_dir / "material" / "etapa1.pt",
        name="material", **MATERIAL_TRAINING,
    )
    finetune = dict(MATERIAL_FINETUNE)
    repeat, ft_epochs = finetune.pop("trashnet_repeat"), finetune.pop("epochs")
    data_ft = oversampled(data_dir, Path(data_dir).parent / f"{Path(data_dir).name}_ajuste", times=repeat)
    return train_classifier(
        data_ft, first, ft_epochs, imgsz, batch, device, project_dir, output=output, name="material_ajuste",
        **{**MATERIAL_TRAINING, **finetune},
    )


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
    mix = sub.add_parser("material", help="treina o classificador de material com o TrashNet + TrashBox")
    mix.add_argument("--zip", help="dataset-resized.zip do TrashNet já baixado")
    mix.add_argument("--trashbox", help="pasta com os clones de TrashBox e TrashBox-testandvalid (padrão: baixa)")
    mix.add_argument("--max-per-class", type=int, default=2600)

    bases = {"trashnet": MODELS_DIR / "yolov8n-cls.pt", "material": MODELS_DIR / MATERIAL_BASE}
    for name, p in sub.choices.items():
        classify = name in bases
        p.add_argument("--epochs", type=int, default={"trashnet": 30, "material": 20}.get(name, 50))
        p.add_argument("--imgsz", type=int, default=224 if classify else 640)
        p.add_argument("--batch", type=int, default=32 if classify else 16)
        p.add_argument("--device", help="cpu, 0 (GPU), mps…")
        p.add_argument("--base", default=str(bases.get(name, DEFAULT_MODEL)), help="peso inicial")
        p.add_argument("--runs-dir", default=str(ROOT / "runs"), help="onde salvar o treino (ex.: pasta do Drive)")

    args = parser.parse_args(argv)
    if args.source == "material":
        from src.weights import resolve

        trashbox = None
        if args.trashbox:
            trashbox = download_trashbox(args.trashbox)  # só confere: com os clones prontos, não baixa nada
        data_dir = prepare_material_dataset(
            trashnet_dir=prepare_trashnet(zip_path=args.zip), trashbox=trashbox, max_per_class=args.max_per_class
        )
        print(json.dumps(json.loads((data_dir / "info.json").read_text(encoding="utf-8"))["fotos"], indent=1))
        out = train_material(data_dir, resolve(args.base), args.epochs, args.imgsz, args.batch, args.device, args.runs_dir)
        print(f"Classificador de material salvo em {out}")
        return
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
