"""Avaliação em cenas reais (TACO): mede o sistema e variações dele em fotos com vários resíduos.

    python .github/scripts/scenes_grid.py resultado.md [coco yoloe yolo-world]

Para cada detector, as previsões (a partir de 5% de confiança) e as do especialista de textura ficam em memória: cada
variação (limite de confiança, classes ignoradas, junção de caixas repetidas, pesos do ensemble) só refaz a parte
barata. As variações são comparadas na validação; o teste mede só a de hoje e a melhor de cada detector.
"""

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detector import Detector  # noqa: E402
from src.evaluate import evaluate_crops, evaluate_scenes, scene_items  # noqa: E402
from src.paths import MODELS_DIR  # noqa: E402
from src.train import prepare_taco_scenes  # noqa: E402

FLOOR = 0.05


class CachedModel:
    """O detector com as previsões de cada foto guardadas: um limite de confiança maior só filtra as caixas (o NMS
    dá as mesmas caixas acima do limite, porque uma caixa só é suprimida por outra mais confiante)."""

    def __init__(self, model):
        self.model, self.cache = model, {}

    def predict(self, source, conf=0.25, **kwargs):
        key = hashlib.md5(source.tobytes()).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.model.predict(source, conf=FLOOR, **kwargs)[0]
        result = self.cache[key]
        return [result[result.boxes.conf >= conf]]

    def __getattr__(self, name):
        return getattr(self.model, name)


class CachedTexture:
    def __init__(self, texture):
        self.texture, self.cache = texture, {}

    def predict(self, crops):
        crops = [c for c in crops if c is not None and c.size]
        keys = [hashlib.md5(c.tobytes() + str(c.shape).encode()).hexdigest() for c in crops]
        todo = {k: c for k, c in zip(keys, crops) if k not in self.cache}
        for k, p in zip(todo, self.texture.predict(list(todo.values()))):
            self.cache[k] = p
        return [self.cache[k] for k in keys]

    def __getattr__(self, name):
        return getattr(self.texture, name)


def configs(kind: str) -> list[dict]:
    ignore = [frozenset(), frozenset({"bowl"})] if kind == "coco" else [frozenset()]
    grid = itertools.product([0.25, 0.15, 0.35, 0.45, 0.55], [None, 0.7], [(0.4, 0.6), (0.6, 0.4)], ignore)
    return [{"conf": c, "merge_overlap": m, "weights": w, "ignore_labels": i} for c, m, w, i in grid]


def load(kind: str) -> Detector:
    """``taco``: o detector de uma classe só treinado nas cenas do TACO (workflow treinar-detector.yml);
    ``coco+taco`` (ou ``yoloe+taco``…): o primeiro e, onde ele não achou nada, esse detector. ``TEXTURE_MODEL`` troca
    o especialista de textura (o padrão é o do sistema)."""
    taco, texture = MODELS_DIR / "residuos_taco.pt", os.environ.get("TEXTURE_MODEL") or "auto"
    if kind == "taco":
        return Detector(model_path=taco, material_model=texture)
    base, _, extra = kind.partition("+")
    return Detector(detector_kind=base, material_model=texture, extra_model=taco if extra == "taco" else None)


def describe(cfg: dict) -> str:
    parts = [f"conf {cfg['conf']:.2f}", "junta repetidas" if cfg["merge_overlap"] else "sem juntar",
             f"forma/textura {cfg['weights'][0]:.0%}/{cfg['weights'][1]:.0%}"]
    if cfg["ignore_labels"]:
        parts.append("ignora " + ", ".join(sorted(cfg["ignore_labels"])))
    return " · ".join(parts)


def row(name: str, r: dict) -> str:
    return (f"| {name} | {r['f1']:.1%} | {r['precision']:.1%} | {r['recall']:.1%} | {r['category_accuracy']:.1%} "
            f"| {r['bins_f1']:.1%} | {r['boxes_per_image']:.1f} |")


HEADER = ("| sistema | F1 | precisão | revocação | lixeira certa dos achados | lixeiras da foto (F1) | caixas/foto |\n"
          "|---|---|---|---|---|---|---|")


def main(out: str, kinds: list[str]) -> None:
    index = Path(os.environ["CENAS_JSON"]) if os.environ.get("CENAS_JSON") else prepare_taco_scenes()
    val, test = scene_items(index, "val"), scene_items(index, "test")
    lines = [f"Fotos: {len(val)} de validação, {len(test)} de teste (TACO, separadas por foto).", ""]
    summary = {}
    for kind in kinds:
        start = time.time()
        det = Detector.from_env() if kind.startswith("padrao") else load(kind)
        if kind == "padrao-neutro":  # o mesmo, com o voto do detector do TACO neutro (a textura decide sozinha)
            det.agnostic_prior = {c: 1 / len(det.agnostic_prior) for c in det.agnostic_prior}
        det.model, det.texture = CachedModel(det.model), CachedTexture(det.texture)
        if det.extra_model is not None:
            det.extra_model = CachedModel(det.extra_model)
        if kind == kinds[0]:
            crops = {split: evaluate_crops(det.texture, items) for split, items in (("val", val), ("test", test))}
            lines += ["## Especialista de textura nos recortes dos resíduos", "",
                      f"Acerto da categoria: validação {crops['val']['accuracy']:.1%} "
                      f"({crops['val']['n']} recortes), teste {crops['test']['accuracy']:.1%} ({crops['test']['n']}).",
                      "", "| categoria | revocação (teste) | precisão (teste) | recortes |", "|---|---|---|---|"]
            lines += [f"| {c} | {v['recall']:.1%} | {v['precision']:.1%} | {v['support']} |"
                      for c, v in crops["test"]["per_class"].items()]
            lines += ["", "Confusão (linhas: verdade; colunas: previsto): "
                      + json.dumps({"labels": crops["test"]["labels"], "matrix": crops["test"]["confusion_matrix"]},
                                   ensure_ascii=False), ""]
        if kind.startswith("padrao"):  # o sistema como ele vem (Detector.from_env), sem variações
            r_val, r_test = evaluate_scenes(det, val), evaluate_scenes(det, test)
            merge = f"junta repetidas ({det.merge_overlap})" if det.merge_overlap else "sem juntar"
            neutral = " (voto do detector do TACO neutro)" if kind == "padrao-neutro" else ""
            lines += [f"## Padrão do sistema{neutral}: {det.description}, conf {det.conf:.2f}, {merge}", "", HEADER,
                      row("validação", r_val), row("teste", r_test)]
            if r_test["found_by_category"]:
                lines += ["", "Lixeira certa por categoria, nos resíduos achados (teste): " + ", ".join(
                    f"{c} {v['recall']:.0%} ({v['support']})" for c, v in r_test["found_by_category"]["per_class"].items())]
            lines += ["", f"_{time.time() - start:.0f} s_", ""]
            summary[kind] = {"validacao": r_val, "teste": r_test}
            print(f"{kind}: F1 validação {r_val['f1']:.1%}, teste {r_test['f1']:.1%}", flush=True)
            continue
        results = []
        for cfg in configs(kind):
            for name, value in cfg.items():
                setattr(det, name, value)
            results.append((cfg, evaluate_scenes(det, val)))
        results.sort(key=lambda x: -x[1]["f1"])
        today = next(r for c, r in results if c == configs(kind)[0])
        lines += [f"## {det.description} (validação)", "", HEADER, row("**o padrão antigo** (" + describe(configs(kind)[0]) + ")", today)]
        lines += [row(describe(c), r) for c, r in results[:12]]
        best_cfg = results[0][0]
        tested = {}
        for label, cfg in (("o padrão antigo", configs(kind)[0]), ("melhor na validação", best_cfg)):
            for name, value in cfg.items():
                setattr(det, name, value)
            tested[label] = evaluate_scenes(det, test)
        lines += ["", f"**Teste** ({det.description}):", "", HEADER]
        lines += [row(f"{label} ({describe(cfg)})", tested[label])
                  for label, cfg in (("o padrão antigo", configs(kind)[0]), ("melhor na validação", best_cfg))]
        per_cat = tested["melhor na validação"]["found_by_category"]
        if per_cat:
            lines += ["", "Lixeira certa por categoria, nos resíduos achados (melhor, teste): " + ", ".join(
                f"{c} {v['recall']:.0%} ({v['support']})" for c, v in per_cat["per_class"].items())]
        lines += ["", f"_{time.time() - start:.0f} s_", ""]
        summary[kind] = {"padrao_antigo": tested["o padrão antigo"], "melhor": tested["melhor na validação"],
                         "melhor_config": describe(best_cfg)}
        print(f"{kind}: padrão antigo F1 {tested['o padrão antigo']['f1']:.1%} → melhor {tested['melhor na validação']['f1']:.1%} "
              f"({describe(best_cfg)})", flush=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    Path(out).with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["coco", "yoloe", "yolo-world"])
