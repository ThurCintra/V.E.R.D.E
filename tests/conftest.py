import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.download_model import download  # noqa: E402
from scripts.prepare_data import copy_sample, fetch_coco_samples, make_conveyor_video, make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def model_path():
    return download()


@pytest.fixture(scope="session")
def detector(model_path):
    """Só o detector COCO (sem a 2ª etapa), para os testes não dependerem do classificador treinado."""
    from src.detector import Detector

    return Detector(model_path, material_model=None)


@pytest.fixture(scope="session")
def material_model():
    from src.paths import MATERIAL_MODEL

    if not MATERIAL_MODEL.is_file():
        pytest.skip("models/material_cls.pt não existe (treine com: python -m src.train trashnet)")
    return MATERIAL_MODEL


@pytest.fixture(scope="session")
def sample_image():
    return copy_sample()


@pytest.fixture(scope="session")
def coco_samples():
    bottles_banana, wine = fetch_coco_samples()
    return {"garrafas_banana": bottles_banana, "vinho": wine}


@pytest.fixture(scope="session")
def synthetic_image():
    return make_synthetic()


@pytest.fixture(scope="session")
def conveyor_video(coco_samples):
    return make_conveyor_video(coco_samples["garrafas_banana"])


@pytest.fixture
def db(tmp_path):
    from src.database import Database

    database = Database(tmp_path / "test.db")
    yield database
    database.close()
