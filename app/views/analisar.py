"""Página principal: enviar ou tirar uma foto (ou vídeo) e ver em qual lixeira vai cada item."""

import tempfile
from pathlib import Path

import cv2
import streamlit as st

import ui
from src.detector import load_image
from src.paths import DATA_DIR
from src.pipeline import WastePipeline
from src.recycling import CATEGORIES

EXAMPLE = DATA_DIR / "mesa_garrafas_banana.jpg"
SOURCES = ["📁 Enviar arquivo", "📷 Tirar foto", "🎥 Câmera contínua"]

ui.hero("Analisar resíduos", "Envie ou tire uma foto: o V.E.R.D.E. mostra em qual lixeira vai cada item.")
cfg = ui.settings()

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
            ("Errou? Toque na certa", "Sua correção fica guardada e ensina o sistema a acertar mais."),
        ]
    )
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
        controller = ui.get_controller() if cfg["comportas"] else None
        pipeline = WastePipeline(ui.detector(), db=ui.db() if save else None, controller=controller, source="video")
        progress, frame_slot = st.progress(0.0, "Analisando..."), st.empty()
        for result in pipeline.process_video(tmp.name, every_n=every_n):
            progress.progress(min(1.0, (result.index + 1) * every_n / total), "Analisando...")
            if result.index % 5 == 0 or result.new_items:
                frame_slot.image(ui.rgb(result.annotated), width="stretch")
        progress.progress(1.0, "Pronto")
    counts = {name: pipeline.counts.get(name, 0) for name in CATEGORIES}
    st.markdown(f"#### {sum(counts.values())} itens no vídeo (cada objeto contado uma vez)")
    ui.chips(counts)
    if save:
        st.success("Registrado! Os números aparecem em **Meu impacto**.")


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
    cam = c1.text_input("Câmera", "0", help="0 é a primeira webcam, 1 a segunda... ou o endereço RTSP de uma câmera IP.")
    running = c2.toggle("Ligar a câmera")
    frame_slot, counts_slot = st.empty(), st.empty()
    if running:
        controller = ui.get_controller() if cfg["comportas"] else None
        pipeline = WastePipeline(ui.detector(), db=ui.db(), controller=controller, source="ao-vivo", review=ui.review())
        try:
            # O loop termina quando a chave é desligada: o Streamlit reexecuta a página
            for result in pipeline.process_video(int(cam) if cam.isdigit() else cam, every_n=2):
                frame_slot.image(ui.rgb(result.annotated), width="stretch")
                with counts_slot.container():
                    ui.chips({name: pipeline.counts.get(name, 0) for name in CATEGORIES})
        except ValueError:
            st.error("Não consegui abrir essa câmera. Confira se ela está ligada ou tente outro número.")
    else:
        st.caption("Ligue a chave para começar.")
