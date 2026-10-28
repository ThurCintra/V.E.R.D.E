"""Mede a precisão no ambiente do site e compara com o relatório (``reports/sistema_modos.json``).

    python .github/scripts/precision_check.py

O mesmo sistema do site público (``WASTE_PUBLIC=1``: COCO + textura) nas 379 fotos de teste do TrashNet, baixadas e
separadas como no treino, nos modos "1 objeto por foto" e normal. Falha se o acerto cair mais de 1 ponto. Os
resultados viram anotações do GitHub Actions (::notice/::error).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WASTE_PUBLIC", "1")

from src.detector import Detector  # noqa: E402
from src.evaluate import TRASHNET_TEST, evaluate_system  # noqa: E402
from src.train import prepare_trashnet  # noqa: E402

TEST_HASH = "595a804696507b0f2f882376b3cf2d35"  # o conteúdo das 379 fotos de teste usadas no relatório
MODES = {"lixeira": ("1 objeto por foto", True), "normal": ("normal", False)}
TOLERANCE = 0.01


def content_hash(folder: Path) -> str:
    return hashlib.md5("".join(sorted(hashlib.md5(p.read_bytes()).hexdigest() for p in folder.rglob("*.jpg"))).encode()).hexdigest()


prepare_trashnet()
if (digest := content_hash(TRASHNET_TEST)) != TEST_HASH:
    print(f"::error title=Precisão::as fotos de teste não são as do relatório (hash {digest})")
    sys.exit(1)

report = json.loads((ROOT / "reports" / "sistema_modos.json").read_text(encoding="utf-8"))
detector = Detector.from_env()
failed = False
for key, (name, single_item) in MODES.items():
    expected = report[f"novo|coco|test_trashnet|{key}"]
    result = evaluate_system(detector, TRASHNET_TEST, single_item=single_item)
    same = "as mesmas respostas por categoria" if result["confusion_matrix"] == expected["confusion"] else "respostas diferentes"
    line = f"{name}: {result['accuracy']:.1%} (relatório: {expected['accuracy']:.1%}), {same} · {detector.description}"
    if result["accuracy"] < expected["accuracy"] - TOLERANCE:
        failed = True
        print(f"::error title=Precisão::{line}")
    else:
        print(f"::notice title=Precisão::{line}")
sys.exit(1 if failed else 0)
