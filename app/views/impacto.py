"""Meu impacto: o que já foi registrado e quanto isso poupou de CO2, energia e água."""

import os
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

import ui
from src.impact import FACTORS, compute_impact
from src.recycling import CATEGORIES
from src.report import to_csv, to_pdf, totals_from_records

TZ = ZoneInfo(os.environ.get("DISPLAY_TZ", "America/Sao_Paulo"))
COLORS = {name: c.hex for name, c in CATEGORIES.items()}
ORDER = list(CATEGORIES)  # legendas na ordem das lixeiras, não em ordem alfabética
PERIODS = ["Últimos 7 dias", "Últimos 30 dias", "Este ano", "Tudo", "Escolher datas"]


def br(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def impact_card(col, icon: str, value: str, label: str, equivalent: str) -> None:
    col.markdown(
        f"<div class='verde-card verde-impact'><p class='label'>{icon} {label}</p><p class='value'>{value}</p>"
        f"<p class='eq'>{equivalent}</p></div>",
        unsafe_allow_html=True,
    )


def chart_style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="", font=dict(size=13), title_font=dict(size=16),
    )
    return fig


ui.hero("Meu impacto", "Tudo o que você registrou e o quanto isso ajudou o meio ambiente.")

today = datetime.now(TZ).date()
period = st.pills("Período", PERIODS, default=PERIODS[1], required=True, label_visibility="collapsed")
if period == "Escolher datas":
    picked = st.date_input("Datas", (today - timedelta(days=30), today), max_value=today, format="DD/MM/YYYY")
    if isinstance(picked, tuple) and len(picked) == 2:
        start_day, end_day = picked
    else:  # ainda escolhendo a data final
        start_day = end_day = picked[0] if isinstance(picked, tuple) else picked
elif period == "Últimos 7 dias":
    start_day, end_day = today - timedelta(days=6), today
elif period == "Últimos 30 dias":
    start_day, end_day = today - timedelta(days=29), today
elif period == "Este ano":
    start_day, end_day = today.replace(month=1, day=1), today
else:
    start_day, end_day = None, today

start = datetime.combine(start_day, time.min, TZ) if start_day else None
end = datetime.combine(end_day + timedelta(days=1), time.min, TZ)
records = ui.db().records(start, end)
impact = compute_impact(totals_from_records(records))
total, eq = impact["total"], impact["equivalents"]

c = st.columns(4)
impact_card(c[0], "♻️", f"{total['items']}", "itens separados", f"{br(total['weight_kg'], 2)} kg no total")
impact_card(c[1], "🌳", f"{br(total['co2_kg'], 2)} kg", "de CO₂ evitado", f"= {br(eq['car_km'])} km sem usar o carro")
impact_card(c[2], "⚡", f"{br(total['energy_kwh'], 2)} kWh", "de energia poupada", f"= {br(eq['home_days'])} dias de uma casa")
impact_card(c[3], "💧", f"{br(total['water_l'])} L", "de água poupada", f"= {br(eq['showers'])} banhos de 8 minutos")
st.write("")

if not records:
    st.info("Ainda não há nada registrado neste período. Analise uma foto e toque em **Registrar no meu impacto**.")
    st.page_link("views/analisar.py", label="Ir para Analisar", icon="📷")
    st.stop()

df = pd.DataFrame(records)
df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TZ)
df["peso_kg"] = df["weight_g"] / 1000
df["co2_kg"] = df.apply(lambda r: FACTORS[r["category"]].co2_kg * r["peso_kg"], axis=1)

with st.container(border=True):
    # Agrupa pela data de verdade e só depois formata: "25/10" não pode vir antes de "15/10"
    daily = df.groupby([df["ts"].dt.date.rename("data"), "category"]).size().reset_index(name="itens")
    daily = daily.sort_values("data")
    daily["dia"] = pd.to_datetime(daily["data"]).dt.strftime("%d/%m")
    fig = px.bar(daily, x="dia", y="itens", color="category", color_discrete_map=COLORS, title="Itens por dia",
                 labels={"dia": "", "itens": "itens", "category": ""},
                 category_orders={"category": ORDER, "dia": list(dict.fromkeys(daily["dia"]))})
    st.plotly_chart(chart_style(fig).update_xaxes(type="category"), width="stretch")

left, right = st.columns(2)
with left.container(border=True):
    by_cat = df.groupby("category")["peso_kg"].sum().reset_index()
    fig = px.pie(by_cat, names="category", values="peso_kg", color="category", color_discrete_map=COLORS,
                 title="Peso por lixeira", hole=0.55, category_orders={"category": ORDER})
    st.plotly_chart(chart_style(fig), width="stretch")
with right.container(border=True):
    cumulative = df.sort_values("ts").assign(co2=lambda d: d["co2_kg"].cumsum())
    fig = px.area(cumulative, x="ts", y="co2", title="CO₂ evitado ao longo do tempo (kg)",
                  labels={"ts": "", "co2": "kg"}, color_discrete_sequence=["#1F8A4C"])
    st.plotly_chart(chart_style(fig), width="stretch")

with st.expander("Ver a tabela por lixeira"):
    table = pd.DataFrame(impact["per_category"]).T.rename(
        columns={"items": "Itens", "weight_kg": "Peso (kg)", "co2_kg": "CO₂ (kg)", "energy_kwh": "Energia (kWh)",
                 "water_l": "Água (L)"}
    )
    st.dataframe(table)
st.caption("Pesos e fatores de impacto são estimativas médias (referências EPA WARM e CEMPRE).")

label = f"{start_day:%d/%m/%Y} a {end_day:%d/%m/%Y}" if start_day else "Todo o histórico"
stamp = f"{start_day or 'inicio'}_{end_day}"
d1, d2, _ = st.columns([1, 1, 2])
d1.download_button("📄 Baixar relatório (PDF)", to_pdf(records, label), f"relatorio_{stamp}.pdf", "application/pdf",
                   type="primary", width="stretch")
d2.download_button("📊 Baixar planilha (CSV)", to_csv(records), f"registros_{stamp}.csv", "text/csv", width="stretch")
