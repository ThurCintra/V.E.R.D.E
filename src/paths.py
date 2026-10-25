"""Caminhos do projeto, resolvidos a partir da raiz (não do diretório atual)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
DEFAULT_MODEL = MODELS_DIR / "yolov8n.pt"
