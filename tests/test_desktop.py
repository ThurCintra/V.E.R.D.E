import importlib.util
import os
import socket
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parent.parent / "desktop" / "verde.py"


@pytest.fixture
def verde(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("verde_desktop", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundled = tmp_path / "pacote"
    (bundled / "models").mkdir(parents=True)
    (bundled / "models" / "material_cls.pt").write_bytes(b"x" * 10)
    monkeypatch.setattr(module, "ROOT", bundled)
    monkeypatch.setattr(module, "data_dir", lambda: tmp_path / "usuario")
    for var in ("WASTE_MODELS_DIR", "WASTE_STORAGE_DIR"):
        monkeypatch.setenv(var, "")  # registra o valor original: o que o lançador definir é desfeito no fim
        monkeypatch.delenv(var)
    return module


def test_prepara_pasta_do_usuario_e_copia_modelos(verde, tmp_path):
    home = verde.prepare_environment()
    assert home == tmp_path / "usuario"
    copied = home / "models" / "material_cls.pt"
    assert copied.read_bytes() == b"x" * 10
    assert os.environ["WASTE_MODELS_DIR"] == str(home / "models")
    assert os.environ["WASTE_STORAGE_DIR"] == str(home / "storage")

    # Um modelo que o usuário já tem (mesmo tamanho) não é copiado de novo
    copied.write_bytes(b"y" * 10)
    verde.prepare_environment()
    assert copied.read_bytes() == b"y" * 10


def test_porta_livre_pula_a_ocupada(verde):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert verde.free_port(port) != port


def test_servidor_que_nao_sobe(verde):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # reservada, mas sem listen: ninguém responde
        assert verde.wait_for_server(port, timeout=0.5) is False
