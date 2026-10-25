"""V.E.R.D.E.: dashboard de monitoramento e impacto ambiental.

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import replace
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
from src.review import ReviewCollector  # noqa: E402
from src.corrections import MIN_CORRECTIONS, NOT_WASTE, OPTIONS, CorrectionStore, previous_model_path, restore_previous, retrain  # noqa: E402
from src.material import crop  # noqa: E402
from src.recycling import DEFAULT_WEIGHT_G  # noqa: E402

TZ = ZoneInfo(os.environ.get("DISPLAY_TZ", "America/Sao_Paulo"))
COLORS = {name: c.hex for name, c in CATEGORIES.items()}

st.set_page_config(page_title="V.E.R.D.E.", page_icon="♻️", layout="wide")


@st.cache_resource
def get_detector() -> Detector:
    return Detector.from_env()


@st.cache_resource
def get_review(path: str) -> ReviewCollector | None:
    return ReviewCollector.from_env(path)


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


@st.cache_resource
def get_controller() -> HardwareController:
    return controller_from_env()


def rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def first_time(key: str) -> bool:
    """True só na primeira vez nesta sessão. O Streamlit reexecuta a página a cada clique: sem isso, a mesma
    foto seria registrada no histórico (e a comporta acionada) de novo a cada botão apertado."""
    done = st.session_state.setdefault("_feito", set())
    if key in done:
        return False
    done.add(key)
    return True


def corrected(image, detections, key: str):
    """Mostra cada objeto com um seletor para corrigir a categoria e devolve as detecções já corrigidas.

    Cada correção (ou confirmação de acerto) guarda o recorte em storage/corrections para o re-treino.
    """
    if not detections:
        return detections
    fixes = st.session_state.setdefault(f"fix_{key}", {})
    store = CorrectionStore(paths.CORRECTIONS_DIR)
    with st.expander("O sistema errou? Corrija aqui (isso ensina o modelo)", expanded=True):
        for i, det in enumerate(detections):
            thumb, info, choice, button = st.columns([1, 2, 2, 1.2], vertical_alignment="center")
            thumb.image(rgb(crop(image, det.box)), width=70)
            info.markdown(f"**#{i + 1}** {det.label} · {det.category} · {det.confidence:.0%}")
            current = fixes.get(i, det.category)
            picked = choice.selectbox(
                f"Categoria certa do #{i + 1}", OPTIONS, index=OPTIONS.index(current),
                key=f"sel_{key}_{i}", label_visibility="collapsed",
            )
            if i in fixes:
                button.markdown("✅ salvo")
            elif button.button("Salvar" if picked != det.category else "Certo", key=f"btn_{key}_{i}"):
                store.add(image, det.box, det.category, picked, det.label, "dashboard")
                fixes[i] = picked
                st.rerun()
        pending = [i for i in range(len(detections)) if i not in fixes]
        if pending and st.button(
            f"Salvar todas ({len(pending)}) · confira antes", key=f"all_{key}",
            help="Salva as categorias escolhidas acima. As que você não mudou contam como CERTAS: se uma estiver errada e você não corrigir, o modelo aprende o erro.",
        ):
            for i in pending:
                picked = st.session_state.get(f"sel_{key}_{i}", detections[i].category)
                store.add(image, detections[i].box, detections[i].category, picked, detections[i].label, "dashboard")
                fixes[i] = picked
            st.rerun()
    result = []
    for i, det in enumerate(detections):
        fix = fixes.get(i)
        if fix == NOT_WASTE:
            continue
        if fix and fix != det.category:
            det = replace(det, category=fix, weight_g=DEFAULT_WEIGHT_G[fix])
        result.append(det)
    return result


def image_key(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:12]


def detections_table(detections) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Categoria": d.category or "-",
                "Classe": d.label,
                "Confiança": f"{d.confidence:.0%}",
                "Material (2ª etapa)": f"{d.material} {d.material_confidence:.0%}" if d.material else "-",
                "Peso (g)": d.weight_g,
            }
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
    st.title("♻️ V.E.R.D.E.")
    st.caption("Detector de resíduos recicláveis")
    conf = st.slider("Confiança mínima", 0.05, 0.9, 0.25, 0.05)
    detector = get_detector()
    detector.conf = conf
    db = get_db(str(paths.DB_PATH))
    controller = get_controller()
    review = get_review(str(paths.REVIEW_DIR))
    actuate = st.toggle("Acionar comportas", value=False, help=f"Backend: {controller.backend.name}")
    single_item = st.toggle(
        "Modo lixeira (1 item por foto)",
        value=os.environ.get("WASTE_SINGLE_ITEM", "0") == "1",
        help="Se o detector não achar nada, a foto inteira passa pelo classificador de material.",
    )
    st.caption(f"Detector: `{detector.model_path.name}` ({detector.mode})")
    if detector.material:
        st.caption(f"Material (2ª etapa): `{detector.material.model_path.name}`")
    if detector.is_coco and not detector.material:
        st.caption("Sem modelo treinado: usando o mapeamento COCO → coleta seletiva. Treine com `src/train.py`.")
    if review:
        n_review = review.count()
        st.caption(f"Casos para revisar: **{n_review}**")
        if n_review:
            st.download_button("Baixar casos (YOLO)", review.export_zip(), "revisao.zip", "application/zip")
    st.markdown("**Cores (CONAMA 275/2001)**")
    for name, cat in CATEGORIES.items():
        st.markdown(f"<span style='color:{cat.hex}'>■</span> {name} · comando `{cat.command}`", unsafe_allow_html=True)

live_tab, analysis_tab, report_tab, train_tab = st.tabs(
    ["Monitoramento ao Vivo", "Análise de Imagem/Vídeo", "Relatório de Impacto Ambiental", "Correções e Treino"]
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
            key = image_key(shot.getvalue())
            image = load_image(shot.getvalue())
            detections = detector.detect(image, waste_only=True, single_item=single_item)
            if review and first_time(f"review_{key}"):
                review.maybe_save(image, detections, "ao-vivo", single_item)
            detections = corrected(image, detections, key)
            st.image(rgb(annotate(image, detections, list(range(1, len(detections) + 1)))), width="stretch")
            if detections:
                action = choose_action(detections)
                if actuate and first_time(f"actuate_{key}"):
                    controller.actuate(action.category)
                st.success(f"Destino: **{action.bin}** · comando `{action.command}`")
                st.dataframe(detections_table(detections), hide_index=True)
                if st.button("Registrar no histórico", key=f"reg_{key}") and first_time(f"db_{key}"):
                    st.toast(f"{db.add_detections(detections, source='ao-vivo')} item(ns) registrado(s)")
            else:
                st.info("Nenhum resíduo reconhecido.")
    else:
        cam = st.text_input("Câmera (índice ou URL)", "0")
        every_n = st.slider("Processar 1 a cada N frames", 1, 10, 2)
        running = st.toggle("Iniciar monitoramento")
        frame_slot, counts_slot = st.empty(), st.empty()
        if running:
            pipeline = WastePipeline(
                detector, db=db, controller=controller if actuate else None, source="ao-vivo", review=review
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
        key = image_key(upload.getvalue())
        image = load_image(upload.getvalue())
        detections = detector.detect(image, waste_only=True, single_item=single_item)
        left, right = st.columns([3, 2])
        with right:
            detections = corrected(image, detections, key)
            st.subheader(f"{len(detections)} resíduo(s)")
            if detections:
                st.dataframe(detections_table(detections), hide_index=True)
                action = choose_action(detections)
                st.success(f"Destino: **{action.bin}** · comando `{action.command}`")
                if st.button("Registrar no histórico", key=f"reg_{key}") and first_time(f"db_{key}"):
                    st.toast(f"{db.add_detections(detections, source='imagem')} item(ns) registrado(s)")
        left.image(rgb(annotate(image, detections, list(range(1, len(detections) + 1)))), width="stretch")
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

# ---------------------------------------------------------------- correções e treino
with train_tab:
    store = CorrectionStore(paths.CORRECTIONS_DIR)
    summary = store.summary()
    st.subheader("Ensine o sistema com as suas correções")
    st.markdown(
        "Nas abas de imagem e câmera, use **O sistema errou? Corrija aqui** em cada objeto. Cada correção (e cada "
        "**Certo**) vira um exemplo de treino. Com pelo menos "
        f"{MIN_CORRECTIONS} correções, treine aqui: o classificador é ajustado com elas e só é trocado se o novo "
        "não ficar pior."
    )
    m = st.columns(3)
    m[0].metric("Correções guardadas", summary["total"])
    m[1].metric("Erros corrigidos", summary["errors"])
    m[2].metric("Acertos confirmados", summary["total"] - summary["errors"])
    st.dataframe(
        pd.DataFrame([{"Categoria": k, "Exemplos": v} for k, v in summary["per_category"].items()]), hide_index=True
    )
    st.caption(f"Classificador em uso: `{paths.active_material_model()}`")

    usable = len(store.items())
    epochs = st.slider("Épocas de treino", 3, 20, 8, help="Mais épocas aprendem mais, mas demoram mais.")
    st.caption(
        f"No seu PC leva alguns minutos (cerca de 1 minuto por época numa CPU comum). Na primeira vez baixa o "
        f"TrashNet (42 MB). Exemplos utilizáveis: {usable}."
    )
    if st.button("Treinar com minhas correções", type="primary", disabled=usable < MIN_CORRECTIONS):
        bar = st.progress(0.0, "Preparando os dados...")
        try:
            result = retrain(
                store, epochs=epochs,
                on_epoch=lambda e, total: bar.progress(e / total, f"Treinando: época {e} de {total}"),
            )
        except Exception as err:  # mostra o erro em vez de derrubar a página
            st.error(f"O treino falhou: {err}")
        else:
            bar.progress(1.0, "Pronto")
            (st.success if result.accepted else st.warning)(result.message)
            cols = st.columns(2)
            cols[0].metric(
                "Acerto no teste do TrashNet", f"{result.new_test_accuracy:.1%}",
                f"{(result.new_test_accuracy - result.old_test_accuracy) * 100:+.1f} pontos",
            )
            if result.new_corrections_accuracy is not None:
                label = "Acerto nas correções separadas" if result.corrections_holdout else "Acerto nas suas correções"
                cols[1].metric(
                    label, f"{result.new_corrections_accuracy:.1%}",
                    f"{(result.new_corrections_accuracy - result.old_corrections_accuracy) * 100:+.1f} pontos",
                )
            if result.accepted:
                get_detector.clear()  # o próximo uso carrega o modelo novo
    if previous_model_path().is_file() and st.button("Voltar ao modelo anterior"):
        restore_previous()
        get_detector.clear()
        st.success("Modelo anterior restaurado.")
