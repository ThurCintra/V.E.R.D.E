from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


@pytest.fixture
def app(tmp_path, monkeypatch, model_path):
    monkeypatch.setenv("WASTE_DB_PATH", str(tmp_path / "dash.db"))
    monkeypatch.setenv("HARDWARE_BACKEND", "sim")
    # paths.py lê o ambiente na importação: recarrega para o banco temporário valer
    import importlib

    import src.paths

    importlib.reload(src.paths)
    yield AppTest.from_file(DASHBOARD, default_timeout=120)
    monkeypatch.undo()
    importlib.reload(src.paths)


def test_dashboard_abre_com_as_tres_abas(app):
    app.run()
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Monitoramento ao Vivo", "Análise de Imagem/Vídeo", "Relatório de Impacto Ambiental",
    ]
    assert any("Nenhuma detecção" in i.value for i in app.info)


def test_relatorio_mostra_o_historico(app, tmp_path):
    from src.database import Database

    db = Database(tmp_path / "dash.db")
    db.add("bottle", "Plástico", 0.9, 30)
    db.add("book", "Papel", 0.8, 300)
    db.close()
    app.run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Itens"] == "2"
    assert metrics["Peso estimado"] == "0.33 kg"
