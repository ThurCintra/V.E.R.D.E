"""Ajuste fino do classificador de material com recortes do TACO, medido antes de trocar o modelo.

    python .github/scripts/train_taco.py resultado.md

1. Prepara as cenas e os recortes do TACO e o dataset combinado TrashNet + TrashBox (revisão e testes).
2. Tira da avaliação as fotos de validação/teste do TACO que repetem uma foto de treino (de qualquer fonte).
3. Treina as variantes e escolhe pela validação: o acerto nos recortes do TACO, sem cair mais de 2 pontos na
   validação do dataset combinado (o mesmo limite do re-treino com as correções).
4. Mede no teste: recortes do TACO, TrashNet, TrashBox e o sistema inteiro nas cenas do TACO.

O melhor candidato fica em ``models/material_cls_taco.pt`` e o resumo em ``reports/material_taco.json``.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dedup import find_duplicates  # noqa: E402
from src.detector import Detector  # noqa: E402
from src.evaluate import IMAGE_SUFFIXES, evaluate_crops, evaluate_material, evaluate_scenes, scene_items  # noqa: E402
from src.material_classifier import LocalTextureExpert  # noqa: E402
from src.paths import MATERIAL_MODEL, MODELS_DIR  # noqa: E402
from src.train import (  # noqa: E402
    DATASETS_DIR, prepare_material_dataset, prepare_taco_crops, prepare_taco_scenes, train_material_taco,
)

VARIANTS = {
    "todas as camadas": {"freeze": None},
    "base congelada": {"freeze": 8},
}
MAX_REFERENCE_DROP = 0.02  # como no re-treino com as correções (corrections.MAX_TEST_DROP)
EPOCHS = int(os.environ.get("TACO_EPOCHS", "8"))
# Pós-processamentos do detector medidos nas cenas (atributos do Detector): o de hoje e o candidato do cenas.yml
SCENE_SETTINGS = json.loads(os.environ.get("SCENE_SETTINGS") or '[{"conf": 0.25}, {"conf": 0.35, "merge_overlap": 0.7}]')


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def scores(model: Path, val, test, reference: Path) -> dict:
    texture = LocalTextureExpert(model)
    out = {
        "taco_val": evaluate_crops(texture, val)["accuracy"],
        "taco_teste": evaluate_crops(texture, test)["accuracy"],
        "referencia_val": evaluate_material(model, reference / "val")["categories"]["accuracy"],
        "trashnet_teste": evaluate_material(model, reference / "test_trashnet")["categories"]["accuracy"],
        "trashbox_teste": evaluate_material(model, reference / "test_trashbox")["categories"]["accuracy"],
    }
    detector = Detector(detector_kind="coco", material_model=model)
    out["cenas_teste"] = []
    for settings in SCENE_SETTINGS:
        for name, value in settings.items():
            setattr(detector, name, frozenset(value) if name == "ignore_labels" else value)
        out["cenas_teste"].append(evaluate_scenes(detector, test))
    return out


def main(out: str) -> None:
    start = time.time()
    index = Path(os.environ["CENAS_JSON"]) if os.environ.get("CENAS_JSON") else prepare_taco_scenes()
    crops = prepare_taco_crops(index)
    reference = prepare_material_dataset()
    if os.environ.get("CI"):  # no GitHub, os originais já copiados (e reduzidos) só ocupam o disco
        for raw in ("taco_raw", "trashbox_raw"):
            shutil.rmtree(DATASETS_DIR / raw, ignore_errors=True)
    val, test = scene_items(index, "val"), scene_items(index, "test")

    # Fotos de avaliação do TACO que repetem uma de treino (o TrashBox veio da internet e pode ter fotos do TACO)
    train_photos = images(reference / "train") + [p for p, _ in scene_items(index, "train")]
    repeated = find_duplicates([p for p, _ in val + test], train_photos)
    val = [(p, o) for p, o in val if p not in repeated]
    test = [(p, o) for p, o in test if p not in repeated]
    lines = [f"Fotos do TACO: {len(val)} de validação e {len(test)} de teste ({len(repeated)} tiradas por repetirem "
             f"uma foto de treino). Recortes de treino: {len(images(crops / 'train'))}.", ""]

    results = {"atual": scores(MATERIAL_MODEL, val, test, reference)}
    for name, settings in VARIANTS.items():
        candidate = DATASETS_DIR / f"candidato_{len(results)}.pt"
        train_material_taco(crops, reference, MATERIAL_MODEL, candidate, epochs=EPOCHS, **settings)
        results[name] = scores(candidate, val, test, reference) | {"arquivo": str(candidate)}
        print(name, json.dumps({k: v for k, v in results[name].items() if k != "cenas_teste"}), flush=True)

    base = results["atual"]
    allowed = {n: r for n, r in results.items() if n != "atual"
               and r["referencia_val"] >= base["referencia_val"] - MAX_REFERENCE_DROP}
    best = max(allowed, key=lambda n: allowed[n]["taco_val"]) if allowed else None

    header = ("| modelo | recortes TACO (val) | recortes TACO (teste) | TrashNet (teste) | TrashBox (teste) "
              "| cenas TACO, F1 (teste) | validação TrashNet+TrashBox |\n|---|---|---|---|---|---|---|")
    lines += ["Cenas: F1 com cada pós-processamento, na ordem " + " · ".join(map(str, SCENE_SETTINGS)), "", header]
    for name, r in results.items():
        mark = " **(escolhido)**" if name == best else ""
        scenes = " / ".join(f"{s['f1']:.1%}" for s in r["cenas_teste"])
        lines.append(f"| {name}{mark} | {r['taco_val']:.1%} | {r['taco_teste']:.1%} | {r['trashnet_teste']:.1%} "
                     f"| {r['trashbox_teste']:.1%} | {scenes} | {r['referencia_val']:.1%} |")
    if best:
        shutil.copyfile(results[best]["arquivo"], MODELS_DIR / "material_cls_taco.pt")
        report = {name: {k: v for k, v in r.items() if k != "arquivo"} for name, r in results.items()}
        report["escolhido"] = best
        (ROOT / "reports" / "material_taco.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    else:
        lines.append("\nNenhuma variante manteve a validação de referência: o modelo atual fica.")
    lines.append(f"\n_{(time.time() - start) / 60:.0f} min_")
    Path(out).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
