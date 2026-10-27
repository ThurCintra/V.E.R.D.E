from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


@pytest.fixture
def app(tmp_path, monkeypatch, model_path):
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
    # A caixa de leite (o COCO vê "garrafa") vira papelão pelo classificador: lixeira azul. Bananas: marrom.
    assert "Lixeira AZUL" in html and "Lixeira MARROM" in html
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
