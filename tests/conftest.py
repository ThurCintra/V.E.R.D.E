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
    from src.detector import Detector

    return Detector(model_path)


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
