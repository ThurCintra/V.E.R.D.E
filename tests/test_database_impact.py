from datetime import datetime, timedelta, timezone

import pytest

from src.detector import Detection
from src.impact import FACTORS, compute_impact, impact_of
from src.report import to_csv, to_pdf, totals_from_records


def test_grava_e_le_o_historico(db):
    db.add("bottle", "Plástico", 0.9, 30, "video", track_id=7)
    (row,) = db.records()
    assert row["label"] == "bottle" and row["category"] == "Plástico" and row["track_id"] == 7
    assert row["weight_g"] == 30 and row["confidence"] == 0.9 and row["source"] == "video"
    assert datetime.fromisoformat(row["ts"]).tzinfo is not None


def test_add_detections_ignora_o_que_nao_e_residuo(db):
    n = db.add_detections(
        [Detection("bottle", 0.9, (0, 0, 1, 1), "Plástico", 30), Detection("person", 0.9, (0, 0, 1, 1))]
    )
    assert n == 1
    assert len(db.records()) == 1


def test_filtro_por_periodo_e_totais(db):
    now = datetime.now(timezone.utc)
    db.add("bottle", "Plástico", 0.9, 30, ts=now - timedelta(days=40))
    db.add("bottle", "Plástico", 0.9, 30, ts=now)
    db.add("book", "Papel", 0.8, 300, ts=now)
    assert len(db.records(start=now - timedelta(days=1))) == 2
    totals = db.totals_by_category(start=now - timedelta(days=1))
    assert totals["Plástico"] == {"items": 1, "weight_kg": pytest.approx(0.03)}
    assert totals["Papel"]["weight_kg"] == pytest.approx(0.3)
    assert len(db.records(end=now - timedelta(days=1))) == 1


def test_persiste_entre_conexoes(tmp_path):
    from src.database import Database

    first = Database(tmp_path / "h.db")
    first.add("cup", "Plástico", 0.7, 10)
    first.close()
    second = Database(tmp_path / "h.db")
    assert len(second.records()) == 1
    second.close()


def test_impacto_por_kg():
    i = impact_of("Plástico", 2.0, 80)
    assert i.co2_kg == pytest.approx(2 * FACTORS["Plástico"].co2_kg)
    assert i.energy_kwh == pytest.approx(2 * FACTORS["Plástico"].energy_kwh)
    assert i.water_l == pytest.approx(2 * FACTORS["Plástico"].water_l)


def test_impacto_total_soma_as_categorias():
    report = compute_impact({"Plástico": {"items": 10, "weight_kg": 0.3}, "Papel": {"items": 2, "weight_kg": 0.6}})
    total = report["total"]
    assert total["items"] == 12
    assert total["weight_kg"] == pytest.approx(0.9)
    assert total["co2_kg"] == pytest.approx(0.3 * 1.1 + 0.6 * 3.5, abs=1e-3)
    assert report["equivalents"]["car_km"] > 0
    assert compute_impact({})["total"]["items"] == 0


def test_todas_as_categorias_tem_fator():
    from src.recycling import CATEGORIES

    assert set(FACTORS) == set(CATEGORIES)


def test_exporta_csv_e_pdf(db):
    db.add("bottle", "Plástico", 0.9, 30)
    db.add("wine glass", "Vidro", 0.8, 150)
    records = db.records()
    csv_text = to_csv(records).decode("utf-8-sig")
    assert csv_text.splitlines()[0].startswith("id,ts,source,label,category")
    assert "Vidro" in csv_text
    pdf = to_pdf(records, "Teste")
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
    assert totals_from_records(records)["Vidro"]["items"] == 1


def test_pdf_vazio_nao_quebra():
    assert to_pdf([]).startswith(b"%PDF")
