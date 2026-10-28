"""Configurações: detalhes técnicos, casos duvidosos e informações do sistema."""

import streamlit as st

import ui
from src import paths

ui.hero("Configurações", "Ajustes opcionais. Para usar no dia a dia, não precisa mudar nada aqui.")
cfg = ui.settings()

with st.container(border=True):
    st.markdown("#### 🎚️ Precisão")
    st.markdown("O modo de operação (Rápido, Alta precisão, Ultra precisão), a confiança mínima, o tamanho dos "
                "objetos e a calibração do fundo ficam no **painel de calibração**: na página **Analisar**, abra a "
                "barra lateral pela setinha **»** no canto superior esquerdo.")
    st.page_link("views/analisar.py", label="Ir para Analisar", icon="📷")

with st.container(border=True):
    st.markdown("#### 🔧 Detalhes técnicos")
    tech = st.toggle("Mostrar detalhes técnicos nas análises", value=cfg["detalhes"],
                     help="Mostra a classe que o modelo viu, a confiança e o peso estimado de cada item.")
    if tech != cfg["detalhes"]:
        ui.save_setting("detalhes", tech)

with st.container(border=True):
    st.markdown("#### 🧪 Casos duvidosos")
    collector = ui.review()
    if collector is None:
        st.caption("Desligado no site público (seriam fotos dos visitantes)." if paths.PUBLIC
                   else "Desligado (WASTE_REVIEW=0).")
    else:
        n = collector.count()
        st.markdown(f"O sistema guardou **{n}** foto(s) em que ficou em dúvida, já com as marcações, para quem quiser "
                    "corrigir num rotulador (Roboflow, CVAT) e treinar o detector.")
        if n:
            st.download_button("Baixar casos duvidosos (.zip)", collector.export_zip(), "revisao.zip",
                               "application/zip")

with st.container(border=True):
    st.markdown("#### ℹ️ Sobre os modelos")
    det = ui.get_detector()
    st.markdown(
        f"- Detector de objetos: `{det.model_path.name}`\n"
        f"- Classificador de material: `{det.material.model_path.name if det.material else 'desligado'}`"
        + (" (treinado com as suas correções)" if paths.USER_MATERIAL_MODEL.is_file() else "")
    )
