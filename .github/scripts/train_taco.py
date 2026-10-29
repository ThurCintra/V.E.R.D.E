"""Ajuste fino do classificador de material com recortes do TACO, medido antes de trocar o modelo.

    python .github/scripts/train_taco.py resultado.md     # treina as variantes de TACO_VARIANTS
    python .github/scripts/train_taco.py --so-dados        # só prepara os dados

1. Prepara as cenas e os recortes do TACO e o dataset combinado TrashNet + TrashBox (revisão e testes).
2. Tira da avaliação as fotos de validação/teste do TACO que repetem uma foto de treino (de qualquer fonte).
3. Treina as variantes; a escolha é pela validação: o acerto nos recortes do TACO, sem cair mais de 1 ponto na
   validação do dataset combinado.
4. Mede no teste: recortes do TACO, TrashNet, TrashBox e o sistema inteiro nas cenas do TACO.

O candidato fica em ``models/material_cls_taco.pt`` e o resumo em ``reports/material_taco.json``.
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
    "todas": {"freeze": None},  # 1ª rodada: 65,7% nos recortes de validação do TACO, mas -1,7 na validação de referência
    "congelada8": {"freeze": 8},  # 1ª rodada: 62,6% e -0,7
    "congelada5": {"freeze": 5},
    "todas_lr1e4": {"freeze": None, "lr0": 0.0001},
}
# Cada workflow treina as variantes de TACO_VARIANTS (separadas por vírgula; no GitHub, uma por máquina, em paralelo)
SELECTED = [v for v in os.environ.get("TACO_VARIANTS", ",".join(VARIANTS)).split(",") if v]
# Fotos de estúdio e da internet (a validação do TrashNet + TrashBox) não podem perder mais de 1 ponto: o mesmo limite
# que o teste do site aceita (.github/scripts/precision_check.py)
MAX_REFERENCE_DROP = 0.01
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


def main(out: str | None) -> None:
    """``out`` None: só prepara os dados (as cenas do TACO e o dataset combinado), para o cache do GitHub."""
    start = time.time()
    index = Path(os.environ["CENAS_JSON"]) if os.environ.get("CENAS_JSON") else prepare_taco_scenes()
    reference = prepare_material_dataset()
    if os.environ.get("CI"):  # no GitHub, os originais já copiados (e reduzidos) só ocupam o disco
        for raw in ("taco_raw", "trashbox_raw"):
            shutil.rmtree(DATASETS_DIR / raw, ignore_errors=True)
    if out is None:
        return
    crops = prepare_taco_crops(index)
    val, test = scene_items(index, "val"), scene_items(index, "test")

    # Fotos de avaliação do TACO que repetem uma de treino (o TrashBox veio da internet e pode ter fotos do TACO)
    train_photos = images(reference / "train") + [p for p, _ in scene_items(index, "train")]
    repeated = find_duplicates([p for p, _ in val + test], train_photos)
    val = [(p, o) for p, o in val if p not in repeated]
    test = [(p, o) for p, o in test if p not in repeated]
    lines = [f"Fotos do TACO: {len(val)} de validação e {len(test)} de teste ({len(repeated)} tiradas por repetirem "
             f"uma foto de treino). Recortes de treino: {len(images(crops / 'train'))}.", ""]

    results = {"atual": scores(MATERIAL_MODEL, val, test, reference)}
    for name in SELECTED:
        candidate = DATASETS_DIR / f"candidato_{name}.pt"
        train_material_taco(crops, reference, MATERIAL_MODEL, candidate, epochs=EPOCHS, **VARIANTS[name])
        results[name] = scores(candidate, val, test, reference) | {"arquivo": str(candidate)}
        print(name, json.dumps({k: v for k, v in results[name].items() if k != "cenas_teste"}), flush=True)

    base = results["atual"]
    for name, r in results.items():
        r["passa"] = name != "atual" and r["referencia_val"] >= base["referencia_val"] - MAX_REFERENCE_DROP
    # O candidato guardado: o melhor na validação do TACO entre os que passam (ou, se nenhum passar, o melhor de
    # todos, para comparar com os das outras máquinas; o resumo diz se passou)
    trained = [n for n in results if n != "atual"]
    passing = [n for n in trained if results[n]["passa"]]
    best = max(passing or trained, key=lambda n: results[n]["taco_val"]) if trained else None

    header = ("| modelo | recortes TACO (val) | recortes TACO (teste) | TrashNet (teste) | TrashBox (teste) "
              "| cenas TACO, F1 (teste) | validação TrashNet+TrashBox |\n|---|---|---|---|---|---|---|")
    lines += ["Cenas: F1 com cada pós-processamento, na ordem " + " · ".join(map(str, SCENE_SETTINGS)), "", header]
    for name, r in results.items():
        mark = (" **(guardado)**" if name == best else "") + ("" if r["passa"] or name == "atual" else " (perde mais de 1 ponto na referência)")
        scenes = " / ".join(f"{s['f1']:.1%}" for s in r["cenas_teste"])
        lines.append(f"| {name}{mark} | {r['taco_val']:.1%} | {r['taco_teste']:.1%} | {r['trashnet_teste']:.1%} "
                     f"| {r['trashbox_teste']:.1%} | {scenes} | {r['referencia_val']:.1%} |")
    if best:
        shutil.copyfile(results[best]["arquivo"], MODELS_DIR / "material_cls_taco.pt")
        report = {name: {k: v for k, v in r.items() if k != "arquivo"} for name, r in results.items()}
        report["guardado"] = best
        (ROOT / "reports" / "material_taco.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    lines.append(f"\n_{(time.time() - start) / 60:.0f} min_")
    Path(out).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main(None if sys.argv[1] == "--so-dados" else sys.argv[1])
