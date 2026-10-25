"""Caminhos do projeto, resolvidos a partir da raiz (não do diretório atual).

O banco e os modelos podem ser trocados por variáveis de ambiente (usado no Docker, com volume persistente).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("WASTE_MODELS_DIR", ROOT / "models"))
DATA_DIR = ROOT / "data"
DEFAULT_MODEL = MODELS_DIR / "yolov8n.pt"
CUSTOM_MODEL = MODELS_DIR / "waste_yolov8n.pt"  # gerado por src/train.py
DB_PATH = Path(os.environ.get("WASTE_DB_PATH", ROOT / "storage" / "detections.db"))
