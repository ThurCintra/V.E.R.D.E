"""Correções do usuário e re-treino do classificador de material com elas.

Quando o sistema erra (ou acerta) um objeto, o dashboard guarda o recorte com a categoria certa em
``storage/corrections/<classe>/``. O re-treino ajusta o classificador em uso com o TrashNet + essas correções
(repetidas, para pesarem) e só troca o modelo se o novo não piorar no teste do TrashNet nem nas correções
separadas para validação. O modelo anterior fica guardado para voltar atrás.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src import paths
from src.material import crop
from src.recycling import CATEGORIES

# Categoria da coleta → classe do classificador (papelão e papel viram "Papel" do mesmo jeito)
CLASS_FOR_CATEGORY = {
    "Plástico": "plastic",
    "Papel": "paper",
    "Metal": "metal",
    "Vidro": "glass",
    "Orgânico/Rejeito": "trash",
}
NOT_WASTE = "Não é resíduo"
NOT_WASTE_DIR = "nao_residuo"  # guardado para um futuro treino do detector; o classificador não usa
OPTIONS = list(CATEGORIES) + [NOT_WASTE]

REPEAT = 5  # cada correção entra 5 vezes no treino: são poucas perto das ~1.800 fotos do TrashNet
MAX_TEST_DROP = 0.02  # tolera até 2 pontos de queda no teste do TrashNet em troca de acertar as correções
MIN_CORRECTIONS = 5


class CorrectionStore:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or paths.CORRECTIONS_DIR)

    def add(
        self,
        image: np.ndarray,
        box: tuple[float, float, float, float],
        predicted: str | None,
        correct: str,
        label: str = "",
        source: str = "dashboard",
    ) -> Path:
        """Guarda o recorte do objeto com a categoria certa. Devolve o arquivo salvo."""
        if correct not in OPTIONS:
            raise ValueError(f"Categoria desconhecida: {correct}")
        folder = CLASS_FOR_CATEGORY.get(correct, NOT_WASTE_DIR)
        dest = self.directory / folder
        dest.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc)
        path = dest / f"{stamp:%Y%m%d-%H%M%S-%f}.jpg"
        cv2.imwrite(str(path), crop(image, box))
        with open(self.directory / "log.jsonl", "a", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "ts": stamp.isoformat(timespec="seconds"),
                        "file": str(path.relative_to(self.directory)),
                        "label": label,
                        "predicted": predicted,
                        "correct": correct,
                        "source": source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return path

    def items(self) -> list[tuple[Path, str]]:
        """(recorte, classe do classificador) de todas as correções úteis para o treino."""
        if not self.directory.is_dir():
            return []
        classes = set(CLASS_FOR_CATEGORY.values())
        return sorted(
            (img, folder.name)
            for folder in self.directory.iterdir()
            if folder.is_dir() and folder.name in classes
            for img in folder.glob("*.jpg")
        )

    def summary(self) -> dict:
        """Quantas correções há por categoria, e quantas eram erros do sistema (e não confirmações)."""
        per_category = {name: 0 for name in OPTIONS}
        errors = 0
        log = self.directory / "log.jsonl"
        if log.is_file():
            for line in log.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                per_category[entry["correct"]] = per_category.get(entry["correct"], 0) + 1
                errors += entry["predicted"] != entry["correct"]
        return {"total": sum(per_category.values()), "errors": errors, "per_category": per_category}


@dataclass
class RetrainResult:
    accepted: bool
    message: str
    old_test_accuracy: float
    new_test_accuracy: float
    old_corrections_accuracy: float | None
    new_corrections_accuracy: float | None
    corrections_used: int
    corrections_holdout: int


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:  # outro disco, ou sistema de arquivos sem hard link
        shutil.copyfile(src, dest)


def _category_accuracy(model_path: Path, items: list[tuple[Path, str]]) -> float | None:
    from src.material import MaterialClassifier
    from src.recycling import category_from_name

    if not items:
        return None
    classifier = MaterialClassifier(model_path)
    hits = 0
    for start in range(0, len(items), 32):
        batch = items[start : start + 32]
        preds = classifier.classify([cv2.imread(str(p)) for p, _ in batch])
        hits += sum(pred.category == category_from_name(cls).name for pred, (_, cls) in zip(preds, batch))
    return hits / len(items)


def split_holdout(items: list[tuple[Path, str]]) -> tuple[list, list]:
    """1 em cada 5 correções fica de fora do treino para conferir se o modelo novo aprendeu (com 10 ou mais)."""
    if len(items) < 10:
        return items, []
    return [x for i, x in enumerate(items) if i % 5], [x for i, x in enumerate(items) if i % 5 == 0]


def retrain(
    store: CorrectionStore | None = None,
    epochs: int = 8,
    trashnet_dir: str | Path | None = None,
    on_epoch=None,
    imgsz: int = 224,
    work_dir: str | Path | None = None,
    lr0: float = 0.0005,
    freeze: int | None = 8,
    **train_overrides,
) -> RetrainResult:
    """Ajusta o classificador em uso com o TrashNet + as correções e troca o modelo só se não piorar.

    ``freeze=8`` congela as camadas de base (texturas e formas) e ajusta só as finais (a decisão do material).
    Sem isso, o modelo aprendia as correções mas "esquecia" parte do resto: com 14 correções de uma foto de
    pilha de lixo, o acerto no teste do TrashNet caía de 95,3% para 92,9%; congelado, foi para 95,8%.
    """
    from src.evaluate import labeled_images
    from src.train import DATASETS_DIR, prepare_trashnet, train_classifier

    store = store or CorrectionStore()
    items = store.items()
    if len(items) < MIN_CORRECTIONS:
        raise ValueError(f"Faça pelo menos {MIN_CORRECTIONS} correções antes de treinar (há {len(items)}).")
    trashnet = Path(trashnet_dir) if trashnet_dir else prepare_trashnet()
    train_items, holdout = split_holdout(items)

    work = Path(work_dir) if work_dir else DATASETS_DIR / "retreino"
    shutil.rmtree(work, ignore_errors=True)
    for split in ("train", "val"):
        for img, cls in labeled_images(trashnet / split):
            _link(img, work / split / cls / img.name)
    for img, cls in train_items:
        for k in range(REPEAT):
            _link(img, work / "train" / cls / f"correcao_{k}_{img.name}")
    for img, cls in holdout:
        _link(img, work / "val" / cls / f"correcao_{img.name}")

    base = paths.active_material_model()
    candidate = work / "candidato.pt"
    train_classifier(
        work, base, epochs=epochs, imgsz=imgsz, device="cpu", output=candidate, name="retreino",
        project_dir=work / "runs",
        optimizer="AdamW", lr0=lr0, freeze=freeze, warmup_epochs=0, patience=epochs, plots=False, on_epoch=on_epoch,
        **train_overrides,
    )

    test = labeled_images(trashnet / "test")
    old_test, new_test = _category_accuracy(base, test), _category_accuracy(candidate, test)
    check = holdout or items  # sem validação separada, mostra pelo menos o acerto nas próprias correções
    old_corr, new_corr = _category_accuracy(base, check), _category_accuracy(candidate, check)

    if new_test < old_test - MAX_TEST_DROP:
        accepted, message = False, "O modelo novo piorou no teste do TrashNet; o atual foi mantido."
    elif holdout and new_corr < old_corr:
        accepted, message = False, "O modelo novo não acertou mais as suas correções; o atual foi mantido."
    else:
        accepted, message = True, "Modelo atualizado com as suas correções."
        user_model = paths.USER_MATERIAL_MODEL
        user_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base, previous_model_path())
        shutil.copyfile(candidate, user_model)

    result = RetrainResult(
        accepted, message, round(old_test, 4), round(new_test, 4),
        None if old_corr is None else round(old_corr, 4), None if new_corr is None else round(new_corr, 4),
        len(items), len(holdout),
    )
    log = paths.STORAGE_DIR / "retreinos.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **asdict(result)}) + "\n")
    return result


def previous_model_path() -> Path:
    return paths.STORAGE_DIR / "material_cls.anterior.pt"


def restore_previous() -> bool:
    """Volta ao classificador de antes do último re-treino. Devolve se havia o que restaurar."""
    previous = previous_model_path()
    if not previous.is_file():
        return False
    if previous.read_bytes() == paths.MATERIAL_MODEL.read_bytes():
        paths.USER_MATERIAL_MODEL.unlink(missing_ok=True)  # o anterior era o do repositório
    else:
        shutil.copyfile(previous, paths.USER_MATERIAL_MODEL)
    previous.unlink()
    return True
