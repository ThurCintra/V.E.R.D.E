"""Caminhos do projeto, resolvidos a partir da raiz (não do diretório atual).

O banco e os modelos podem ser trocados por variáveis de ambiente (usado no Docker, com volume persistente).

``PUBLIC``: o dashboard está num site público (várias pessoas no mesmo servidor). Liga sozinho no Streamlit
Community Cloud, que roda os apps em /mount/src; ``WASTE_PUBLIC=1`` liga e ``WASTE_PUBLIC=0`` desliga.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("WASTE_MODELS_DIR", ROOT / "models"))
DATA_DIR = ROOT / "data"
DEFAULT_MODEL = MODELS_DIR / "yolov8s.pt"  # o yolov8n (mais rápido) acha bem menos objetos em pilhas de lixo
CUSTOM_MODEL = MODELS_DIR / "waste_detector.pt"  # detector de resíduos, gerado por src/train.py
MATERIAL_MODEL = MODELS_DIR / "material_cls.pt"  # classificador de material (TrashNet), idem
TACO_DETECTOR = MODELS_DIR / "residuos_taco.pt"  # detector de uma classe só ("residuo"), treinado nas cenas do TACO
STORAGE_DIR = Path(os.environ.get("WASTE_STORAGE_DIR", ROOT / "storage"))  # dados do usuário, fora do Git
USER_MATERIAL_MODEL = STORAGE_DIR / "material_cls.pt"  # classificador re-treinado com as correções do usuário
CORRECTIONS_DIR = Path(os.environ.get("WASTE_CORRECTIONS_DIR", STORAGE_DIR / "corrections"))  # do usuário
REVIEW_DIR = Path(os.environ.get("WASTE_REVIEW_DIR", STORAGE_DIR / "review"))  # casos duvidosos
DB_PATH = Path(os.environ.get("WASTE_DB_PATH", STORAGE_DIR / "detections.db"))
PUBLIC = os.environ.get("WASTE_PUBLIC", "1" if Path("/mount/src").is_dir() else "0") == "1"
DESKTOP_APP_URL = "https://github.com/ThurCintra/V.E.R.D.E/releases/latest"  # instalador para Windows


def active_material_model() -> Path:
    """O classificador em uso: o re-treinado pelo usuário, se existir; senão o que vem no repositório."""
    return USER_MATERIAL_MODEL if USER_MATERIAL_MODEL.is_file() else MATERIAL_MODEL
