"""Página principal: enviar ou tirar uma foto (ou vídeo) e ver em qual lixeira vai cada item."""

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

import ui
from src import paths
from src.detector import load_image
from src.live import LiveSession
from src.paths import DATA_DIR
from src.pipeline import WastePipeline
from src.recycling import CATEGORIES, label_pt

EXAMPLE = DATA_DIR / "mesa_garrafas_banana.jpg"
DISPLAY_FPS = 20  # a tela da câmera contínua; a análise roda no ritmo dela, em paralelo
SOURCES = ["📁 Enviar arquivo", "📷 Tirar foto", "🎥 Câmera contínua"]
if paths.PUBLIC:  # a câmera contínua é a do computador que roda o sistema: no site, a do servidor (que não tem)
    SOURCES = SOURCES[:2]

ui.hero("Analisar resíduos", "Envie ou tire uma foto: o V.E.R.D.E. mostra em qual lixeira vai cada item.")
cfg = ui.settings()
ui.engine_status()

top_left, top_right = st.columns([2, 1], vertical_alignment="bottom")
source = top_left.pills("Como você quer analisar?", SOURCES, default=SOURCES[0], required=True, key="fonte")
one = top_right.toggle(
    "Só um objeto por foto", value=cfg["um_objeto"],
    help="Ligue quando cada foto tiver um item só (por exemplo, numa lixeira com câmera). Fica bem mais preciso.",
)
if one != cfg["um_objeto"]:
    ui.save_setting("um_objeto", one)


def empty_state() -> None:
    st.write("")
    ui.steps(
        [
            ("Envie ou tire uma foto", "De um objeto ou de vários. Quanto mais perto e com mais luz, melhor."),
            ("Veja a lixeira certa", "Cada item ganha um número e a cor da lixeira onde deve ir."),
            ("Errou? Toque na certa", "A foto mostra a lixeira certa."
             if paths.PUBLIC else "Sua correção fica guardada e ensina o sistema a acertar mais."),
        ]
    )
    if paths.PUBLIC:
        st.caption(f"Versão online de demonstração. A câmera contínua, a calibração do fundo e o re-treino com as "
                   f"suas correções ficam no [aplicativo para Windows]({paths.DESKTOP_APP_URL}).")
    st.write("")
    st.markdown("**As 5 lixeiras da coleta seletiva**")
    ui.legend()


def analyze_video(data: bytes, name: str) -> None:
    every_n = st.select_slider(
        "Velocidade da análise", options=[1, 2, 3, 5, 10], value=3,
        format_func=lambda n: {1: "Cada quadro (lenta)", 3: "Normal", 10: "Rápida"}.get(n, f"1 a cada {n} quadros"),
    )
    save = st.toggle("Registrar os itens no meu impacto", value=True)
    if not st.button("▶️ Analisar vídeo", type="primary"):
        return
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
        pipeline = WastePipeline(ui.detector(), db=ui.db() if save else None, source="video")
        progress, frame_slot = st.progress(0.0, "Analisando..."), st.empty()
        for result in pipeline.process_video(tmp.name, every_n=every_n):
            progress.progress(min(1.0, (result.index + 1) * every_n / total), "Analisando...")
            if result.index % 5 == 0 or result.new_items:
                frame_slot.image(ui.rgb(result.annotated), width="stretch", output_format="JPEG")
        progress.progress(1.0, "Pronto")
    counts = {name: pipeline.counts.get(name, 0) for name in CATEGORIES}
    st.markdown(f"#### {sum(counts.values())} itens no vídeo (cada objeto contado uma vez)")
    ui.chips(counts)
    if save:
        st.success("Registrado! Os números aparecem em **Meu impacto**.")


def live_loop(session: LiveSession, frame_slot, counts_slot, items_slot) -> None:
    """Mostra a câmera a ~20 quadros por segundo com as caixas da última análise (que roda em paralelo) e
    atualiza a lista ao lado uma vez por segundo."""
    last_list = 0.0
    while True:
        started = time.monotonic()
        view = session.view()
        if view is not None:
            frame_slot.image(ui.rgb(view), width="stretch", output_format="JPEG")
        if session.error:
            st.error(f"A análise parou: {session.error}")
            return
        if not session.running:
            st.info("A câmera parou de enviar imagens.")
            return
        now = time.monotonic()
        if now - last_list >= 1.0:
            last_list = now
            result, counts = session.result, session.pipeline.counts
            with counts_slot.container():
                st.markdown("**Contados até agora** (cada objeto uma vez)")
                ui.chips({name: counts.get(name, 0) for name in CATEGORIES})
                st.caption(f"Analisando {session.analysis_fps:.0f} quadros por segundo")
            with items_slot.container():
                for d in result.detections if result else []:
                    st.markdown(
                        f"**#{d.tracker_id} {label_pt(d.label).capitalize()}** · {ui.EMOJI[d.category]} "
                        f"{d.category}" + ui.prob_bar(d) + ui.scan_note(d), unsafe_allow_html=True,
                    )
        time.sleep(max(0.0, 1 / DISPLAY_FPS - (time.monotonic() - started)))


def current_frame():
    """O quadro para "Zerar fundo": da câmera contínua, ou a última foto analisada."""
    if source == SOURCES[2]:
        session = st.session_state.get("sessao_ao_vivo")
        if session is not None and session.running:  # a câmera já está aberta pela análise
            return session.reader.latest()[1]
        cap = cv2.VideoCapture(int(cam) if cam.isdigit() else cam)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return st.session_state.get("ultima_imagem")


cam = st.session_state.get("camera", "0")
ui.calibration_panel(current_frame)

if source == SOURCES[0]:
    upload = st.file_uploader(
        "Escolha uma foto ou vídeo do seu computador ou celular",
        type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "avi", "mov", "mkv"],
    )
    if upload is None:
        if st.button("✨ Experimentar com uma foto de exemplo", disabled=not EXAMPLE.is_file()):
            st.session_state["exemplo"] = True
        if st.session_state.get("exemplo"):
            data = EXAMPLE.read_bytes()
            ui.show_result(load_image(data), ui.image_key(data), "imagem")
        else:
            empty_state()
    elif upload.type.startswith("image"):
        st.session_state["exemplo"] = False
        data = upload.getvalue()
        ui.show_result(load_image(data), ui.image_key(data), "imagem")
    else:
        analyze_video(upload.getvalue(), upload.name)

elif source == SOURCES[1]:
    shot = st.camera_input("Aponte a câmera para o resíduo", label_visibility="collapsed")
    if shot is None:
        st.caption("Permita o acesso à câmera no navegador, se ele pedir.")
    else:
        data = shot.getvalue()
        ui.show_result(load_image(data), ui.image_key(data), "ao-vivo")

else:
    st.markdown(
        "Analisa sem parar uma câmera ligada **ao computador onde o V.E.R.D.E. está rodando** (webcam, câmera USB "
        "ou câmera IP). Cada objeto é contado uma vez e registrado no seu impacto."
    )
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    c1.text_input("Câmera", key="camera",
                  help="0 é a primeira webcam, 1 a segunda... ou o endereço RTSP de uma câmera IP.")
    running = c2.toggle("Ligar a câmera")
    video_col, list_col = st.columns([1.4, 1], gap="large")
    frame_slot = video_col.empty()
    counts_slot, items_slot = list_col.empty(), list_col.empty()
    if running:
        pipeline = WastePipeline(ui.detector(), db=ui.db(), source="ao-vivo", review=ui.review())
        old = st.session_state.pop("sessao_ao_vivo", None)
        if old is not None:  # de uma execução anterior da página que não terminou direito
            old.stop()
        try:
            session = LiveSession(pipeline, cam)
        except ValueError:
            st.error("Não consegui abrir essa câmera. Confira se ela está ligada ou tente outro número.")
        else:
            st.session_state["sessao_ao_vivo"] = session
            try:
                live_loop(session, frame_slot, counts_slot, items_slot)
            finally:  # a chave desligada reexecuta a página e interrompe o loop: a câmera é liberada aqui
                session.stop()
                st.session_state.pop("sessao_ao_vivo", None)
    else:
        frame_slot.caption("Ligue a chave para começar. Os números na imagem acompanham cada objeto enquanto ele "
                           "estiver na frente da câmera.")
