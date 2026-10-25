"""Exportação do relatório de impacto em CSV e PDF."""

from __future__ import annotations

import csv
import io
from datetime import datetime

from fpdf import FPDF

from src.database import COLUMNS
from src.impact import compute_impact


def to_csv(records: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode("utf-8-sig")  # BOM: o Excel abre os acentos certos


def totals_from_records(records: list[dict]) -> dict[str, dict]:
    totals: dict[str, dict] = {}
    for r in records:
        t = totals.setdefault(r["category"], {"items": 0, "weight_kg": 0.0})
        t["items"] += 1
        t["weight_kg"] += r["weight_g"] / 1000
    return totals


def _fmt(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def to_pdf(records: list[dict], period: str = "Todo o histórico") -> bytes:
    impact = compute_impact(totals_from_records(records))
    total, eq = impact["total"], impact["equivalents"]

    pdf = FPDF()
    pdf.set_title("Relatório de Impacto Ambiental")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Relatório de Impacto Ambiental", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Período: {period}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Gerado em {datetime.now():%d/%m/%Y %H:%M}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Resumo", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for line in (
        f"Itens detectados: {total['items']}",
        f"Peso estimado: {_fmt(total['weight_kg'], 3)} kg",
        f"CO2 evitado: {_fmt(total['co2_kg'])} kg (= {_fmt(eq['car_km'], 1)} km de carro)",
        f"Energia economizada: {_fmt(total['energy_kwh'])} kWh (= {_fmt(eq['home_days'], 1)} dias de uma casa)",
        f"Água poupada: {_fmt(total['water_l'], 1)} L (= {_fmt(eq['showers'], 1)} banhos de 8 min)",
    ):
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Por categoria", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    with pdf.table(col_widths=(40, 20, 28, 28, 34, 30), text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT")) as table:
        table.row(("Categoria", "Itens", "Peso (kg)", "CO2 (kg)", "Energia (kWh)", "Água (L)"))
        for cat, i in sorted(impact["per_category"].items()):
            table.row(
                (cat, str(i["items"]), _fmt(i["weight_kg"], 3), _fmt(i["co2_kg"]), _fmt(i["energy_kwh"]), _fmt(i["water_l"], 1))
            )

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0, 4,
        "Pesos e fatores de impacto são estimativas médias por item e por kg reciclado (referências EPA WARM e "
        "CEMPRE); servem para acompanhar tendências, não como medição.",
    )
    return bytes(pdf.output())
