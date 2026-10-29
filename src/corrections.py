"""Correções do usuário e re-treino do classificador de material com elas.

Quando o sistema erra (ou acerta) um objeto, o dashboard guarda o recorte com a categoria certa em
``storage/corrections/<classe>/``. O re-treino ajusta o classificador em uso com fotos de referência + essas
correções (repetidas, para pesarem) e só troca o modelo se o novo não piorar nos testes de referência nem nas
correções separadas para validação. O modelo anterior fica guardado para voltar atrás.

As fotos de referência são uma amostra do dataset combinado TrashNet + TrashBox (``datasets/material``), se ele
existir no computador, ou o TrashNet. Revisar também as fotos variadas do TrashBox evita que o modelo "esqueça"
o que aprendeu com elas.
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

REPEAT = 5  # cada correção entra 5 vezes no treino: são poucas perto das ~1.800 fotos de referência
MAX_TEST_DROP = 0.02  # tolera até 2 pontos de queda em cada teste de referência em troca de acertar as correções
REPLAY_PER_CLASS = {"train": 300, "val": 60}  # amostra do dataset combinado: o re-treino continua em minutos
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

    def remove(self, path: str | Path) -> None:
        """Desfaz uma correção (o usuário mudou de ideia): apaga o recorte e registra no log."""
        path = Path(path)
        path.unlink(missing_ok=True)
        with open(self.directory / "log.jsonl", "a", encoding="utf-8") as log:
            log.write(json.dumps({"removed": str(path.relative_to(self.directory))}) + "\n")

    def _entries(self) -> list[dict]:
        """As correções em vigor, na ordem (sem as desfeitas)."""
        log = self.directory / "log.jsonl"
        if not log.is_file():
            return []
        entries: dict[str, dict] = {}
        for line in log.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if "removed" in entry:
                entries.pop(entry["removed"], None)
            else:
                entries[entry["file"]] = entry
        return list(entries.values())

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
        for entry in self._entries():
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
    """Põe ``src`` em ``dest`` sem ocupar espaço (hard link) ou, se não der, copiando."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if os.path.samefile(src, dest):
            return  # já é o mesmo arquivo (sobra de um treino anterior): aproveita
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:  # outro disco, ou sistema de arquivos sem hard link
        shutil.copyfile(src, dest)


def _fresh_work_dir(root: Path) -> Path:
    """Uma pasta nova por treino. No Windows, um arquivo aberto (pelo dashboard, pelo antivírus) não pode ser
    apagado: reaproveitar a mesma pasta fazia o 2º treino tropeçar nas sobras do 1º. As antigas são apagadas
    quando possível; se alguma estiver presa, fica para a próxima."""
    if root.is_dir():
        for old in root.iterdir():
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
            else:
                old.unlink(missing_ok=True)
    work = root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    work.mkdir(parents=True)
    return work


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


def _reference(trashnet_dir: str | Path | None) -> tuple[Path, list[Path], bool]:
    """(pasta com train/ e val/, testes que o modelo novo não pode piorar, se é o dataset combinado)."""
    from src.train import DATASETS_DIR, prepare_trashnet

    if trashnet_dir:
        return Path(trashnet_dir), [Path(trashnet_dir) / "test"], False
    mix = DATASETS_DIR / "material"
    if (mix / "info.json").is_file() and (mix / "test_trashbox").is_dir():
        return mix, [mix / "test_trashnet", mix / "test_trashbox"], True
    trashnet = prepare_trashnet()
    return trashnet, [trashnet / "test"], False


def _sample(items: list[tuple[Path, str]], per_class: int, seed: int = 0) -> list[tuple[Path, str]]:
    import random

    rng, out = random.Random(seed), []
    for cls in sorted({c for _, c in items}):
        of_class = [x for x in items if x[1] == cls]
        out += rng.sample(of_class, min(per_class, len(of_class)))
    return out


def retrain(
    store: CorrectionStore | None = None,
    epochs: int = 8,
    trashnet_dir: str | Path | None = None,
    on_epoch=None,
    imgsz: int = 224,
    work_dir: str | Path | None = None,
    lr0: float = 0.0001,
    freeze: int | None = 8,
    **train_overrides,
) -> RetrainResult:
    """Ajusta o classificador em uso com as fotos de referência + as correções e troca o modelo só se não piorar.

    ``freeze=8`` congela as camadas de base (texturas e formas) e ajusta só as finais (a decisão do material).
    Sem isso, o modelo aprendia as correções mas "esquecia" parte do resto: com 14 correções de uma foto de
    pilha de lixo, o acerto no teste do TrashNet caía de 95,3% para 92,9%; congelado, foi para 95,8%.

    ``lr0=0.0001``: o classificador atual sai do treino com uma taxa bem baixa. Com 15 correções de fotos que ele
    errava, 0.0005 aprendia 12 delas mas derrubava o teste do TrashNet de 92,4% para 87,1% (e era recusado);
    0.0001 aprendeu 10 e subiu o teste para 92,9%.
    """
    from src.evaluate import labeled_images
    from src.train import DATASETS_DIR, train_classifier

    store = store or CorrectionStore()
    items = store.items()
    if len(items) < MIN_CORRECTIONS:
        raise ValueError(f"Faça pelo menos {MIN_CORRECTIONS} correções antes de treinar (há {len(items)}).")
    reference, test_dirs, combined = _reference(trashnet_dir)
    train_items, holdout = split_holdout(items)

    work = _fresh_work_dir(Path(work_dir) if work_dir else DATASETS_DIR / "retreino")
    for split in ("train", "val"):
        base_items = labeled_images(reference / split)
        if combined:
            base_items = _sample(base_items, REPLAY_PER_CLASS[split])
        for img, cls in base_items:
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

    # O primeiro teste é o do TrashNet (o número mostrado no dashboard); com o dataset combinado, o do TrashBox
    # também não pode piorar
    tests = [labeled_images(d) for d in test_dirs]
    old_tests = [_category_accuracy(base, t) for t in tests]
    new_tests = [_category_accuracy(candidate, t) for t in tests]
    old_test, new_test = old_tests[0], new_tests[0]
    check = holdout or items  # sem validação separada, mostra pelo menos o acerto nas próprias correções
    old_corr, new_corr = _category_accuracy(base, check), _category_accuracy(candidate, check)

    if any(new < old - MAX_TEST_DROP for old, new in zip(old_tests, new_tests)):
        accepted, message = False, "O modelo novo piorou nas fotos de referência; o atual foi mantido."
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
