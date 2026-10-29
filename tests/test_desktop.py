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


def test_so_uma_copia_do_programa_por_vez(verde, tmp_path):
    home = tmp_path / "usuario"
    first = verde.single_instance_lock(home)
    assert first is not None
    assert verde.single_instance_lock(home) is None  # a segunda cópia só abre uma janela
    (home / "endereco.txt").write_text("http://127.0.0.1:8502\n")
    assert verde.address_of_running(home, wait_s=0.1) == "http://127.0.0.1:8502"
    first.close()  # a primeira terminou
    assert verde.single_instance_lock(home) is not None


def test_tela_de_carregando_aponta_para_o_servidor(verde, tmp_path, monkeypatch):
    monkeypatch.setattr(verde, "find_app_browser", lambda: None)
    opened = []
    monkeypatch.setattr(verde.webbrowser, "open", opened.append)
    home = tmp_path / "usuario"
    home.mkdir()
    assert verde.open_window("http://127.0.0.1:8599", home) is None
    page = (home / "carregando.html").read_text(encoding="utf-8")
    assert 'const url="http://127.0.0.1:8599"' in page and opened == [(home / "carregando.html").as_uri()]
