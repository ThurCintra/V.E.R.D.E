"""Treino grande do classificador de material: modelo maior, mais fotos e mais épocas, na CPU.

O re-treino do dashboard (``src/corrections.py``) só ajusta as camadas finais do modelo em uso com as suas
correções. Este treina um classificador novo do zero (a partir do ImageNet), para quando você quer que ele acerte
mais em geral, não só nos casos que corrigiu:

- **modelo maior:** ``yolov8s-cls`` (5 M de parâmetros) no lugar do ``yolov8n-cls`` (1,4 M);
- **mais fotos:** TrashNet (2.527 fotos de estúdio, fundo branco) + RealWaste (4.752 fotos de lixo de verdade, num
  aterro: amassado, sujo, com fundo variado; UCI, CC BY 4.0) + as suas correções (cada uma conta 5 vezes);
- **mais épocas**, com parada antecipada se a validação parar de melhorar.

As 9 classes do RealWaste viram as 6 do TrashNet (comida, vegetação, tecido e rejeito → ``trash``), então o modelo
novo funciona no sistema sem mudar mais nada.

No fim, o modelo novo é comparado com o atual no teste do TrashNet, no do RealWaste e nas correções separadas
para validação. Ele só entra em uso se não cair mais de 1 ponto em nenhum deles e melhorar na média; o anterior fica
guardado (botão **Voltar ao modelo anterior** no dashboard).

    python -m src.big_train                          # o treino completo (horas, na CPU)
    python -m src.big_train --continuar              # retoma de onde parou (se o PC desligou no meio)
    python -m src.big_train --epocas 1 --fracao 0.05 --sem-trocar   # teste rápido: só mede, não troca nada

O treino fica em ``runs/treino_grande/`` (``results.csv`` mostra a acurácia de cada época).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src import paths
from src.corrections import REPEAT, CorrectionStore, _category_accuracy, _link, previous_model_path, split_holdout
from src.evaluate import IMAGE_SUFFIXES, REPORTS_DIR, labeled_images
from src.train import DATASETS_DIR, prepare_trashnet, train_classifier

REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
# Classe do RealWaste → classe do classificador (as do TrashNet)
REALWASTE_CLASSES = {
    "Cardboard": "cardboard",
    "Food Organics": "trash",
    "Glass": "glass",
    "Metal": "metal",
    "Miscellaneous Trash": "trash",
    "Paper": "paper",
    "Plastic": "plastic",
    "Textile Trash": "trash",
    "Vegetation": "trash",
}
WORK_DIR = DATASETS_DIR / "treino_grande"
RUN_NAME = "treino_grande"
MAX_DROP = 0.01  # o modelo novo pode cair até 1 ponto num teste, desde que melhore na média


def _realwaste_members(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    """Fotos do zip agrupadas pela classe do RealWaste (a pasta logo acima da foto)."""
    by_class: dict[str, list[str]] = {}
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) >= 2 and parts[-2] in REALWASTE_CLASSES and Path(parts[-1]).suffix.lower() in IMAGE_SUFFIXES:
            by_class.setdefault(parts[-2], []).append(name)
    return by_class


def prepare_realwaste(
    out_dir: str | Path = DATASETS_DIR / "realwaste",
    zip_path: str | Path | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> Path:
    """Baixa (ou usa ``zip_path``) o RealWaste e separa treino/validação/teste, já nas classes do TrashNet.

    A separação é feita por classe original (comida e vegetação separadas, mesmo indo as duas para ``trash``), para o
    teste ter de tudo. Não refaz o que já existe."""
    out_dir = Path(out_dir)
    if (out_dir / "test").is_dir():
        return out_dir
    if zip_path is None:
        zip_path = DATASETS_DIR / "realwaste.zip"
        if not zip_path.is_file():
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Baixando o RealWaste de {REALWASTE_URL}…")
            urllib.request.urlretrieve(REALWASTE_URL, zip_path)

    with zipfile.ZipFile(zip_path) as outer:
        by_class = _realwaste_members(outer)
        inner = [n for n in outer.namelist() if n.lower().endswith(".zip")]
        if not by_class and inner:  # a UCI às vezes embrulha o dataset num segundo zip
            nested = Path(zip_path).with_name("realwaste_interno.zip")
            nested.write_bytes(outer.read(inner[0]))
            return prepare_realwaste(out_dir, nested, val_fraction, test_fraction, seed)
        if not by_class:
            raise ValueError(f"Nenhuma foto do RealWaste encontrada em {zip_path}")

        rng = random.Random(seed)
        for source_cls, names in sorted(by_class.items()):
            names = sorted(names)
            rng.shuffle(names)
            n_test, n_val = round(len(names) * test_fraction), round(len(names) * val_fraction)
            splits = {"test": names[:n_test], "val": names[n_test : n_test + n_val], "train": names[n_test + n_val :]}
            prefix = "rw_" + source_cls.lower().replace(" ", "_")
            for split, members in splits.items():
                dest = out_dir / split / REALWASTE_CLASSES[source_cls]
                dest.mkdir(parents=True, exist_ok=True)
                for member in members:
                    (dest / f"{prefix}_{Path(member).name}").write_bytes(outer.read(member))
    return out_dir


def build_dataset(work: Path, sources: list[Path], train_items: list, holdout: list) -> Path:
    """Junta os datasets (hard links, sem ocupar espaço) e as correções numa pasta no formato do ``yolo classify``."""
    shutil.rmtree(work, ignore_errors=True)
    for source in sources:
        for split in ("train", "val"):
            for img, cls in labeled_images(source / split):
                _link(img, work / split / cls / f"{source.name}_{img.name}")
    for img, cls in train_items:
        for k in range(REPEAT):
            _link(img, work / "train" / cls / f"correcao_{k}_{img.name}")
    for img, cls in holdout:
        _link(img, work / "val" / cls / f"correcao_{img.name}")
    return work


def compare(old: Path, new: Path, tests: dict[str, list]) -> tuple[bool, str, dict]:
    """Acerto por categoria do modelo atual e do novo em cada teste, e se o novo deve entrar em uso."""
    scores = {name: {"atual": _category_accuracy(old, items), "novo": _category_accuracy(new, items)}
              for name, items in tests.items() if items}
    drops = [name for name, s in scores.items() if s["novo"] < s["atual"] - MAX_DROP]
    mean_old = sum(s["atual"] for s in scores.values()) / len(scores)
    mean_new = sum(s["novo"] for s in scores.values()) / len(scores)
    if drops:
        return False, f"O modelo novo piorou em: {', '.join(drops)}. O atual foi mantido.", scores
    if mean_new <= mean_old:
        return False, "O modelo novo não acertou mais que o atual na média. O atual foi mantido.", scores
    return True, f"Modelo novo em uso: média de {mean_old:.1%} para {mean_new:.1%}.", scores


def run(
    epochs: int = 40,
    base: str | Path = paths.MODELS_DIR / "yolov8s-cls.pt",
    batch: int = 32,
    patience: int = 10,
    fraction: float = 1.0,
    use_realwaste: bool = True,
    resume: bool = False,
    swap: bool = True,
) -> dict:
    trashnet = prepare_trashnet()
    sources = [trashnet]
    realwaste = prepare_realwaste() if use_realwaste else None
    if realwaste:
        sources.append(realwaste)
    train_items, holdout = split_holdout(CorrectionStore().items())

    last = paths.ROOT / "runs" / RUN_NAME / "weights" / "last.pt"
    if resume and last.is_file():
        from ultralytics import YOLO

        model = YOLO(str(last))
        model.train(resume=True)
        candidate = Path(model.trainer.best if Path(model.trainer.best).is_file() else last)
    else:
        work = build_dataset(WORK_DIR, sources, train_items, holdout)
        candidate = train_classifier(
            work, base, epochs=epochs, imgsz=224, batch=batch, device="cpu", output=WORK_DIR.parent / "candidato_grande.pt",
            name=RUN_NAME, optimizer="AdamW", lr0=0.001, cos_lr=True, patience=patience, dropout=0.1, fraction=fraction,
        )

    tests = {"TrashNet": labeled_images(trashnet / "test")}
    if realwaste:
        tests["RealWaste"] = labeled_images(realwaste / "test")
    tests["correções"] = holdout
    current = paths.active_material_model()
    accepted, message, scores = compare(current, candidate, tests)
    if accepted and swap:
        paths.USER_MATERIAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current, previous_model_path())
        shutil.copyfile(candidate, paths.USER_MATERIAL_MODEL)
    elif accepted:
        message = "O modelo novo seria aceito, mas --sem-trocar manteve o atual."

    result = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo": "treino_grande",
        "accepted": accepted and swap,
        "message": message,
        "base": Path(base).name,
        "epochs": epochs,
        "fraction": fraction,
        "datasets": [s.name for s in sources],
        "corrections_used": len(train_items),
        "corrections_holdout": len(holdout),
        "scores": {k: {m: round(v, 4) for m, v in s.items()} for k, s in scores.items()},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "treino_grande.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log = paths.STORAGE_DIR / "retreinos.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--base", default=str(paths.MODELS_DIR / "yolov8s-cls.pt"),
                        help="peso inicial (yolov8n-cls.pt: mais rápido; yolov8m-cls.pt: maior e bem mais lento)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--paciencia", type=int, default=10, help="para se a validação não melhorar em N épocas")
    parser.add_argument("--fracao", type=float, default=1.0, help="usa só essa fração das fotos de treino (teste rápido)")
    parser.add_argument("--sem-realwaste", action="store_true", help="treina só com o TrashNet e as correções")
    parser.add_argument("--continuar", action="store_true", help="retoma o último treino interrompido")
    parser.add_argument("--sem-trocar", action="store_true", help="só mede: não troca o modelo em uso")
    args = parser.parse_args(argv)

    result = run(args.epocas, args.base, args.batch, args.paciencia, args.fracao, not args.sem_realwaste,
                 args.continuar, not args.sem_trocar)
    for name, s in result["scores"].items():
        print(f"{name:10} atual {s['atual']:.1%}  →  novo {s['novo']:.1%}")
    print(result["message"])


if __name__ == "__main__":
    main()
