from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


@pytest.fixture
def public_app(monkeypatch, app):
    """O dashboard como no site público (Streamlit Community Cloud)."""
    import importlib

    import src.paths

    monkeypatch.setenv("WASTE_PUBLIC", "1")
    importlib.reload(src.paths)
    return app


@pytest.fixture
def app(tmp_path, monkeypatch, model_path):
    monkeypatch.setenv("WASTE_PUBLIC", "0")
    monkeypatch.setenv("WASTE_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("WASTE_DB_PATH", str(tmp_path / "dash.db"))
    monkeypatch.setenv("WASTE_MODEL", str(model_path))  # yolov8n: mais rápido nos testes
    # paths.py lê o ambiente na importação: recarrega para as pastas temporárias valerem
    import importlib

    import src.paths

    importlib.reload(src.paths)
    yield AppTest.from_file(DASHBOARD, default_timeout=120)
    monkeypatch.undo()
    importlib.reload(src.paths)


def test_abre_na_pagina_analisar_sem_erros(app):
    app.run()
    assert not app.exception
    assert any("Analisar resíduos" in m.value for m in app.markdown)
    assert [b.label for b in app.main.button] == ["✨ Experimentar com uma foto de exemplo"]
    assert "🧹 Zerar fundo e calibrar luz" in [b.label for b in app.sidebar.button]  # painel de calibração


def test_foto_de_exemplo_mostra_itens_e_lixeiras(app):
    app.run()
    app.button[0].click().run()
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    # A caixa de leite (o COCO vê "garrafa") vira papelão pelo classificador: lixeira azul. O resultado completo da
    # foto (caixa, pote e banana) é conferido com os modelos de verdade em .github/scripts/site_check.py; aqui o
    # detector é o yolov8n, mais rápido, que acha a banana com menos de 55% de confiança
    assert "Lixeira AZUL" in html
    assert any(b.label == "✅ Registrar no meu impacto" for b in app.button)


def test_corrigir_um_item_guarda_o_exemplo(app, tmp_path):
    app.run()
    app.button[0].click().run()
    first = app.pills[1]  # [0] é a escolha da fonte (enviar/tirar foto/câmera)
    first.set_value("Metal").run()
    assert not app.exception
    saved = list((tmp_path / "storage" / "corrections" / "metal").glob("*.jpg"))
    assert len(saved) == 1


def test_meu_impacto_mostra_o_historico(app, tmp_path):
    from src.database import Database

    db = Database(tmp_path / "dash.db")
    db.add("bottle", "Plástico", 0.9, 30)
    db.add("book", "Papel", 0.8, 300)
    db.close()
    app.run()
    app.switch_page("views/impacto.py").run()
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    assert "itens separados" in html and ">2<" in html


def test_paginas_de_ensinar_e_configuracoes_abrem(app):
    app.run()
    for page in ("views/ensinar.py", "views/configuracoes.py"):
        app.switch_page(page).run()
        assert not app.exception, page


# ---------------------------------------------------------------- site público
def test_site_publico_sem_camera_do_servidor_nem_calibracao(public_app):
    public_app.run()
    assert not public_app.exception
    assert public_app.pills[0].options == ["Enviar arquivo", "Tirar foto"]  # o emoji vira ícone
    assert "🧹 Zerar fundo e calibrar luz" not in [b.label for b in public_app.sidebar.button]


def test_site_publico_nao_guarda_fotos_nem_configuracoes(public_app, tmp_path):
    public_app.run()
    public_app.toggle[0].set_value(True).run()  # "Só um objeto por foto"
    public_app.button[0].click().run()
    public_app.pills[1].set_value("Metal").run()
    assert not public_app.exception
    storage = tmp_path / "storage"
    assert not (storage / "corrections").exists() and not (storage / "review").exists()
    assert not (storage / "configuracoes.json").exists()


def test_site_publico_sem_retreino(public_app):
    public_app.run()
    public_app.switch_page("views/ensinar.py").run()
    assert not public_app.exception
    assert any("re-treino fica desligado" in i.value for i in public_app.info)
    assert not [b for b in public_app.button if "Treinar" in b.label]


def test_site_publico_usa_o_coco_e_nao_guarda_casos(monkeypatch):
    import importlib

    import src.paths
    from src.detector import Detector
    from src.review import ReviewCollector

    for var in ("WASTE_DETECTOR", "WASTE_REVIEW", "WASTE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WASTE_PUBLIC", "1")
    importlib.reload(src.paths)
    try:
        seen = {}
        monkeypatch.setattr(Detector, "__init__", lambda self, **kw: seen.update(kw))
        Detector.from_env()
        assert seen["detector_kind"] == "coco"
        assert ReviewCollector.from_env() is None
    finally:
        monkeypatch.undo()
        importlib.reload(src.paths)
