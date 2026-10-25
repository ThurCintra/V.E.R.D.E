import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.download_model import download  # noqa: E402
from scripts.prepare_data import copy_sample, make_synthetic  # noqa: E402


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
def synthetic_image():
    return make_synthetic()
