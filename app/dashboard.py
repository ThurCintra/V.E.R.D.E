"""Dashboard de monitoramento e impacto ambiental.

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paths  # noqa: E402
from src.database import Database  # noqa: E402
from src.detector import Detector, annotate, load_image  # noqa: E402
from src.hardware_controller import HardwareController, choose_action, controller_from_env  # noqa: E402
from src.impact import FACTORS, compute_impact  # noqa: E402
from src.pipeline import WastePipeline  # noqa: E402
from src.recycling import CATEGORIES  # noqa: E402
from src.report import to_csv, to_pdf, totals_from_records  # noqa: E402

TZ = ZoneInfo(os.environ.get("DISPLAY_TZ", "America/Sao_Paulo"))
COLORS = {name: c.hex for name, c in CATEGORIES.items()}

st.set_page_config(page_title="Detector de Resíduos", page_icon="♻️", layout="wide")


@st.cache_resource
def get_detector() -> Detector:
    return Detector()


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


@st.cache_resource
def get_controller() -> HardwareController:
    return controller_from_env()


def rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def detections_table(detections) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Categoria": d.category or "-", "Classe": d.label, "Confiança": f"{d.confidence:.0%}", "Peso (g)": d.weight_g}
            for d in detections
        ]
    )


def counts_row(counts) -> None:
    cols = st.columns(len(CATEGORIES))
    for col, (name, cat) in zip(cols, CATEGORIES.items()):
        col.markdown(
            f"<div style='border-left:6px solid {cat.hex};padding-left:8px'>{name}</div>", unsafe_allow_html=True
        )
        col.metric(name, counts.get(name, 0), label_visibility="collapsed")


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.title("♻️ Detector de Resíduos")
    conf = st.slider("Confiança mínima", 0.05, 0.9, 0.25, 0.05)
    detector = get_detector()
    detector.conf = conf
    db = get_db(str(paths.DB_PATH))
    controller = get_controller()
    actuate = st.toggle("Acionar comportas", value=False, help=f"Backend: {controller.backend.name}")
    st.caption(f"Modelo: `{detector.model_path.name}` ({detector.mode})")
    if detector.is_coco:
        st.caption("Sem modelo treinado: usando o mapeamento COCO → coleta seletiva. Treine com `src/train.py`.")
    st.markdown("**Cores (CONAMA 275/2001)**")
    for name, cat in CATEGORIES.items():
        st.markdown(f"<span style='color:{cat.hex}'>■</span> {name} · comando `{cat.command}`", unsafe_allow_html=True)

live_tab, analysis_tab, report_tab = st.tabs(
    ["Monitoramento ao Vivo", "Análise de Imagem/Vídeo", "Relatório de Impacto Ambiental"]
)

# ---------------------------------------------------------------- ao vivo
with live_tab:
    source = st.radio(
        "Fonte", ["Câmera do navegador", "Câmera ou stream no servidor"], horizontal=True,
        help="A câmera do servidor é a do Raspberry Pi/PC onde o sistema roda (índice 0, 1… ou URL RTSP).",
    )
    if source == "Câmera do navegador":
        shot = st.camera_input("Aponte o resíduo para a câmera")
        if shot is not None:
            image = load_image(shot.getvalue())
            detections = detector.detect(image, waste_only=True)
            st.image(rgb(annotate(image, detections)), width="stretch")
            if detections:
                db.add_detections(detections, source="ao-vivo")
                action = choose_action(detections)
                sent = controller.actuate(action.category) if actuate else False
                st.success(f"Destino: **{action.bin}** · comando `{action.command}`" + (" (enviado)" if sent else ""))
                st.dataframe(detections_table(detections), hide_index=True)
            else:
                st.info("Nenhum resíduo reconhecido.")
    else:
        cam = st.text_input("Câmera (índice ou URL)", "0")
        every_n = st.slider("Processar 1 a cada N frames", 1, 10, 2)
        running = st.toggle("Iniciar monitoramento")
        frame_slot, counts_slot = st.empty(), st.empty()
        if running:
            pipeline = WastePipeline(
                detector, db=db, controller=controller if actuate else None, source="ao-vivo"
            )
            try:
                # O loop termina quando o toggle é desligado: o Streamlit reexecuta o script
                for result in pipeline.process_video(int(cam) if cam.isdigit() else cam, every_n=every_n):
                    frame_slot.image(rgb(result.annotated), width="stretch")
                    with counts_slot.container():
                        counts_row(pipeline.counts)
            except ValueError as err:
                st.error(str(err))

# ---------------------------------------------------------------- análise
with analysis_tab:
    upload = st.file_uploader(
        "Imagem ou vídeo", type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "avi", "mov", "mkv"]
    )
    if upload is not None and upload.type.startswith("image"):
        image = load_image(upload.getvalue())
        detections = detector.detect(image, waste_only=True)
        left, right = st.columns([3, 2])
        left.image(rgb(annotate(image, detections)), width="stretch")
        with right:
            st.subheader(f"{len(detections)} resíduo(s)")
            if detections:
                st.dataframe(detections_table(detections), hide_index=True)
                action = choose_action(detections)
                st.success(f"Destino: **{action.bin}** · comando `{action.command}`")
                if st.button("Registrar no histórico"):
                    st.toast(f"{db.add_detections(detections, source='imagem')} item(ns) registrado(s)")
    elif upload is not None:
        every_n = st.slider("Processar 1 a cada N frames", 1, 10, 3, key="video_every_n")
        save = st.checkbox("Registrar itens no histórico", value=True)
        if st.button("Analisar vídeo"):
            with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix) as tmp:
                tmp.write(upload.getvalue())
                tmp.flush()
                cap = cv2.VideoCapture(tmp.name)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                cap.release()
                pipeline = WastePipeline(
                    detector, db=db if save else None, controller=controller if actuate else None, source="video"
                )
                progress, frame_slot = st.progress(0.0), st.empty()
                for result in pipeline.process_video(tmp.name, every_n=every_n):
                    progress.progress(min(1.0, (result.index + 1) * every_n / total))
                    if result.index % 5 == 0 or result.new_items:
                        frame_slot.image(rgb(result.annotated), width="stretch")
                progress.progress(1.0)
            st.subheader("Itens contados (cada objeto uma vez)")
            counts_row(pipeline.counts)

# ---------------------------------------------------------------- relatório
with report_tab:
    today = datetime.now(TZ).date()
    period = st.date_input("Período", (today - timedelta(days=30), today), max_value=today, format="DD/MM/YYYY")
    if isinstance(period, tuple) and len(period) == 2:
        start_day, end_day = period
    else:  # o usuário ainda está escolhendo a data final
        start_day = end_day = period[0] if isinstance(period, tuple) else period
    start = datetime.combine(start_day, time.min, TZ)
    end = datetime.combine(end_day + timedelta(days=1), time.min, TZ)

    records = db.records(start, end)
    impact = compute_impact(totals_from_records(records))
    total, eq = impact["total"], impact["equivalents"]

    m = st.columns(5)
    m[0].metric("Itens", total["items"])
    m[1].metric("Peso estimado", f"{total['weight_kg']:.2f} kg")
    m[2].metric("CO₂ evitado", f"{total['co2_kg']:.2f} kg", help=f"≈ {eq['car_km']} km de carro")
    m[3].metric("Energia", f"{total['energy_kwh']:.2f} kWh", help=f"≈ {eq['home_days']} dias de uma casa")
    m[4].metric("Água", f"{total['water_l']:.1f} L", help=f"≈ {eq['showers']} banhos de 8 min")

    if not records:
        st.info("Nenhuma detecção registrada no período.")
    else:
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TZ)
        df["dia"] = df["ts"].dt.strftime("%d/%m")
        df["peso_kg"] = df["weight_g"] / 1000
        df["co2_kg"] = df.apply(lambda r: FACTORS[r["category"]].co2_kg * r["peso_kg"], axis=1)

        daily = df.groupby(["dia", "category"]).size().reset_index(name="itens")
        st.plotly_chart(
            px.bar(daily, x="dia", y="itens", color="category", color_discrete_map=COLORS,
                   title="Itens por dia", labels={"dia": "Dia", "itens": "Itens", "category": "Categoria"})
            .update_xaxes(type="category"),
            width="stretch",
        )
        left, right = st.columns(2)
        by_cat = df.groupby("category")["peso_kg"].sum().reset_index()
        left.plotly_chart(
            px.pie(by_cat, names="category", values="peso_kg", color="category", color_discrete_map=COLORS,
                   title="Peso por categoria (kg)", hole=0.45),
            width="stretch",
        )
        cumulative = df.sort_values("ts").assign(co2_acumulado=lambda d: d["co2_kg"].cumsum())
        right.plotly_chart(
            px.area(cumulative, x="ts", y="co2_acumulado", title="CO₂ evitado acumulado (kg)",
                    labels={"ts": "Data", "co2_acumulado": "kg CO₂"}),
            width="stretch",
        )

        st.dataframe(
            pd.DataFrame(impact["per_category"]).T.rename(
                columns={"items": "Itens", "weight_kg": "Peso (kg)", "co2_kg": "CO₂ (kg)",
                         "energy_kwh": "Energia (kWh)", "water_l": "Água (L)"}
            ),
        )

    label = f"{start_day:%d/%m/%Y} a {end_day:%d/%m/%Y}"
    c1, c2 = st.columns(2)
    c1.download_button("Exportar CSV", to_csv(records), f"deteccoes_{start_day}_{end_day}.csv", "text/csv",
                       disabled=not records)
    c2.download_button("Exportar PDF", to_pdf(records, label), f"relatorio_{start_day}_{end_day}.pdf",
                       "application/pdf", disabled=not records)
