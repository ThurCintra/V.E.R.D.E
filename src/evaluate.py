"""Avaliação: mede o quanto o sistema acerta, para comparar versões antes de trocar o modelo em produção.

    python -m src.evaluate material                 # classificador de material no teste do TrashNet
    python -m src.evaluate material --test datasets/material/test_trashbox --report material_trashbox
    python -m src.evaluate system                   # sistema inteiro: só COCO × COCO + material
    python -m src.evaluate detector --models yolov8n.pt yolov8s.pt   # detectores no coco128

Os resultados (JSON, Markdown e matrizes de confusão em PNG) vão para ``reports/``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.paths import MATERIAL_MODEL, ROOT
from src.recycling import CATEGORIES, COCO_RULES, category_from_name

REPORTS_DIR = ROOT / "reports"
TRASHNET_TEST = ROOT / "datasets" / "trashnet" / "test"
NONE = "nenhum"  # o sistema não reconheceu resíduo nenhum
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def labeled_images(test_dir: str | Path) -> list[tuple[Path, str]]:
    """(imagem, classe) de uma pasta no formato do ``yolo classify``: test/<classe>/<imagem>."""
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"{test_dir} não existe. Prepare com: python -m src.train trashnet --epochs 0")
    return [
        (img, cls_dir.name)
        for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir())
        for img in sorted(cls_dir.iterdir())
        if img.suffix.lower() in IMAGE_SUFFIXES
    ]


def metrics(truth: list[str], pred: list[str], labels: list[str]) -> dict:
    """Acurácia, precisão/revocação/F1 por classe e matriz de confusão (linhas = verdade, colunas = previsto)."""
    pred_labels = labels + sorted({p for p in pred if p not in labels})
    index = {label: i for i, label in enumerate(pred_labels)}
    matrix = np.zeros((len(labels), len(pred_labels)), dtype=int)
    for t, p in zip(truth, pred):
        matrix[labels.index(t), index[p]] += 1
    per_class = {}
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": support,
        }
    return {
        "accuracy": round(float(np.trace(matrix[:, : len(labels)]) / max(1, len(truth))), 4),
        "macro_f1": round(float(np.mean([c["f1"] for c in per_class.values()])), 4),
        "n": len(truth),
        "per_class": per_class,
        "labels": labels,
        "predicted_labels": pred_labels,
        "confusion_matrix": matrix.tolist(),
    }


def plot_confusion(result: dict, title: str, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(result["confusion_matrix"])
    rows = matrix / np.maximum(1, matrix.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(1.3 * matrix.shape[1] + 2, 1.1 * matrix.shape[0] + 1.5))
    ax.imshow(rows, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(matrix.shape[1]), result["predicted_labels"], rotation=35, ha="right")
    ax.set_yticks(range(matrix.shape[0]), result["labels"])
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdade")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="white" if rows[i, j] > 0.5 else "black")
    ax.set_title(f"{title}\nacurácia {result['accuracy']:.1%} · F1 macro {result['macro_f1']:.1%}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def evaluate_material(model_path: str | Path = MATERIAL_MODEL, test_dir: str | Path = TRASHNET_TEST, batch: int = 32) -> dict:
    """Classificador sozinho: acerto por classe do TrashNet e por categoria da coleta seletiva."""
    import cv2

    from src.material import MaterialClassifier

    classifier = MaterialClassifier(model_path)
    items = labeled_images(test_dir)
    preds = []
    for start in range(0, len(items), batch):
        preds += classifier.classify([cv2.imread(str(p)) for p, _ in items[start : start + batch]])
    classes = sorted({c for _, c in items})
    categories = [c for c in CATEGORIES if c in {category_from_name(x).name for x in classes}]
    return {
        "model": Path(model_path).name,
        "classes": metrics([c for _, c in items], [p.label for p in preds], classes),
        "categories": metrics(
            [category_from_name(c).name for _, c in items], [p.category for p in preds], categories
        ),
    }


def evaluate_system(detector, test_dir: str | Path = TRASHNET_TEST, single_item: bool = True) -> dict:
    """Sistema inteiro, como a API decide: a categoria do resíduo mais confiante da foto (ou "nenhum")."""
    from src.detector import best_waste

    items = labeled_images(test_dir)
    truth, pred = [], []
    for path, cls in items:
        best = best_waste(detector.detect(path, waste_only=True, single_item=single_item))
        truth.append(category_from_name(cls).name)
        pred.append(best.category if best else NONE)
    categories = [c for c in CATEGORIES if c in set(truth)]
    result = metrics(truth, pred, categories)
    result["recognized"] = round(sum(p != NONE for p in pred) / len(pred), 4)
    result["mode"] = getattr(detector, "description", type(detector).__name__)
    return result


def scene_items(index: str | Path, split: str) -> list[tuple[Path, list[dict]]]:
    """(foto, objetos) de um ``cenas.json`` (``src.train.prepare_taco_scenes``). Cada objeto tem ``box`` (x1, y1,
    x2, y2) e ``category`` (None: material desconhecido)."""
    index = Path(index)
    scenes = json.loads(index.read_text(encoding="utf-8"))["scenes"]
    return [(index.parent / s["file"], s["objects"]) for s in scenes if s["split"] == split]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_scene(preds: list[tuple[tuple, str, float]], objects: list[dict], image_area: float,
                min_area: float = 0.0, iou_min: float = 0.5) -> dict:
    """Casa as caixas do sistema (caixa, categoria, confiança) com os objetos da foto, das mais confiantes para as
    menos, como no mAP: cada objeto vale para uma caixa só. Objetos sem categoria ou menores que ``min_area`` (fração
    da foto) não contam, e a caixa que cair num deles é ignorada."""
    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    ignored = [o["category"] is None or area(o["box"]) <= max(0.0, min_area * image_area) for o in objects]
    taken = [False] * len(objects)
    out = {"tp": 0, "fp": 0, "wrong": 0, "pairs": []}
    for box, category, _ in sorted(preds, key=lambda p: -p[2]):
        scores = [iou(box, o["box"]) if not taken[i] else 0.0 for i, o in enumerate(objects)]
        best = int(np.argmax(scores)) if scores else -1
        if best < 0 or scores[best] < iou_min:
            out["fp"] += 1
            continue
        taken[best] = True
        if ignored[best]:
            continue
        truth = objects[best]["category"]
        out["pairs"].append((truth, category))
        if category == truth:
            out["tp"] += 1
        else:
            out["wrong"] += 1
    out["objects"] = sum(not i for i in ignored)
    out["truth_bins"] = {o["category"] for o, i in zip(objects, ignored) if not i}
    out["pred_bins"] = {c for _, c, _ in preds}
    return out


def evaluate_scenes(detector, items: list[tuple[Path, list[dict]]], iou_min: float = 0.5) -> dict:
    """Sistema inteiro em fotos com vários resíduos, no modo normal: um resíduo só conta como acerto se uma caixa
    (IoU >= 0,5) o achou com a categoria certa.

    - **precisão**: das caixas que o sistema mostrou, quantas acertaram o objeto e a categoria;
    - **revocação**: dos resíduos das fotos, quantos o sistema achou com a categoria certa;
    - **categoria dos achados**: dos resíduos que o sistema achou, em quantos acertou a lixeira;
    - **lixeiras da foto**: as categorias que o sistema mostrou × as que a foto tem (precisão e revocação somadas
      em todas as fotos).
    """
    import cv2

    tp = fp = wrong = objects = 0
    bins_tp = bins_pred = bins_truth = 0
    pairs, boxes = [], 0
    for path, gt in items:
        image = cv2.imread(str(path))
        dets = [d for d in detector.detect(image, waste_only=True) if d.category]
        boxes += len(dets)
        m = match_scene([(d.box, d.category, d.confidence) for d in dets], gt, image.shape[0] * image.shape[1],
                        detector.min_area, iou_min)
        tp, fp, wrong, objects = tp + m["tp"], fp + m["fp"], wrong + m["wrong"], objects + m["objects"]
        pairs += m["pairs"]
        bins_tp += len(m["truth_bins"] & m["pred_bins"])
        bins_pred += len(m["pred_bins"])
        bins_truth += len(m["truth_bins"])
    shown = tp + fp + wrong
    precision, recall = tp / max(1, shown), tp / max(1, objects)
    bins_p, bins_r = bins_tp / max(1, bins_pred), bins_tp / max(1, bins_truth)
    categories = [c for c in CATEGORIES if any(t == c for t, _ in pairs)]
    found = metrics([t for t, _ in pairs], [p for _, p in pairs], categories) if pairs else None
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall) if precision + recall else 0.0, 4),
        "found": round(len(pairs) / max(1, objects), 4),
        "category_accuracy": found["accuracy"] if found else 0.0,
        "bins_precision": round(bins_p, 4),
        "bins_recall": round(bins_r, 4),
        "bins_f1": round(2 * bins_p * bins_r / (bins_p + bins_r) if bins_p + bins_r else 0.0, 4),
        "boxes_per_image": round(boxes / max(1, len(items)), 2),
        "images": len(items),
        "objects": objects,
        "found_by_category": found,
    }


def evaluate_crops(texture, items: list[tuple[Path, list[dict]]], min_area: float = 0.0005) -> dict:
    """Especialista de textura sozinho, nos recortes dos resíduos das cenas (a caixa certa, com a margem de sempre):
    mede o material em fotos do mundo real, com rótulos, sujeira e fundo."""
    import cv2

    from src.material import crop

    truth, pred = [], []
    for path, gt in items:
        image = cv2.imread(str(path))
        area = image.shape[0] * image.shape[1]
        objs = [o for o in gt if o["category"]
                and (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) >= min_area * area]
        crops = [(o, crop(image, o["box"])) for o in objs]
        crops = [(o, c) for o, c in crops if c.size]
        for (o, _), p in zip(crops, texture.predict([c for _, c in crops])):
            truth.append(o["category"])
            pred.append(max(p.probs, key=p.probs.get))
    return metrics(truth, pred, [c for c in CATEGORIES if c in set(truth)])


def evaluate_detector(model: str, data: str = "coco128.yaml", imgsz: int = 640) -> dict:
    """mAP do detector num dataset YOLO. No COCO, também a média só das classes que viram resíduo."""
    from src.weights import resolve
    from ultralytics import YOLO

    yolo = YOLO(str(resolve(model)))
    metrics_ = yolo.val(data=data, imgsz=imgsz, device="cpu", plots=False, verbose=False, project=str(ROOT / "runs"), name="val", exist_ok=True)
    names = yolo.names
    result = {
        "model": Path(model).name,
        "imgsz": imgsz,
        "mAP50": round(float(metrics_.box.map50), 4),
        "mAP50_95": round(float(metrics_.box.map), 4),
        "ms_per_image": round(float(metrics_.speed["inference"]), 1),
    }
    waste_idx = [int(i) for i in metrics_.box.ap_class_index if names[int(i)] in COCO_RULES]
    if waste_idx:
        ap50 = dict(zip((int(i) for i in metrics_.box.ap_class_index), metrics_.box.ap50))
        ap = dict(zip((int(i) for i in metrics_.box.ap_class_index), metrics_.box.ap))
        result["waste_classes"] = len(waste_idx)
        result["waste_mAP50"] = round(float(np.mean([ap50[i] for i in waste_idx])), 4)
        result["waste_mAP50_95"] = round(float(np.mean([ap[i] for i in waste_idx])), 4)
    return result


def _save(name: str, data: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _per_class_table(result: dict) -> str:
    lines = ["| Classe | Precisão | Revocação | F1 | Fotos |", "| --- | --- | --- | --- | --- |"]
    for label, m in result["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} | {m['support']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    mat = sub.add_parser("material")
    mat.add_argument("--model", default=str(MATERIAL_MODEL))
    mat.add_argument("--test", default=str(TRASHNET_TEST))
    mat.add_argument("--report", default="material", help="nome dos arquivos em reports/")
    sysp = sub.add_parser("system")
    sysp.add_argument("--test", default=str(TRASHNET_TEST))
    sysp.add_argument("--report", default="sistema", help="nome dos arquivos em reports/")
    sysp.add_argument("--detector", default=None, help="peso do detector (padrão: o mesmo do sistema)")
    det = sub.add_parser("detector")
    det.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    det.add_argument("--data", default="coco128.yaml")
    det.add_argument("--imgsz", type=int, nargs="+", default=[640])
    args = parser.parse_args(argv)

    if args.what == "material":
        result = evaluate_material(args.model, args.test)
        _save(args.report, result)
        where = Path(args.test).name
        plot_confusion(result["classes"], f"Classificador de material ({where})", REPORTS_DIR / f"{args.report}_classes.png")
        plot_confusion(result["categories"], f"Classificador de material: categorias ({where})",
                       REPORTS_DIR / f"{args.report}_categorias.png")
        print(f"Classes:    acurácia {result['classes']['accuracy']:.1%}")
        print(f"Categorias: acurácia {result['categories']['accuracy']:.1%}")
        print(_per_class_table(result["classes"]))
    elif args.what == "system":
        from src.detector import Detector

        results = {}
        for name, material in (("so_coco", None), ("coco_material", "auto")):
            detector = Detector(args.detector, material_model=material)
            results[name] = evaluate_system(detector, args.test)
            plot_confusion(results[name], f"Sistema ({detector.description})", REPORTS_DIR / f"{args.report}_{name}.png")
            r = results[name]
            print(f"{detector.description:28} acurácia {r['accuracy']:.1%} · reconheceu algo em {r['recognized']:.1%}")
        _save(args.report, results)
    else:
        results = [evaluate_detector(m, args.data, s) for m in args.models for s in args.imgsz]
        _save("detectores", {"data": args.data, "results": results})
        for r in results:
            print(
                f"{r['model']:12} {r['imgsz']:>4}px  mAP50 {r['mAP50']:.3f}  resíduos mAP50 {r.get('waste_mAP50', 0):.3f}"
                f"  {r['ms_per_image']} ms/img"
            )


if __name__ == "__main__":
    main()
