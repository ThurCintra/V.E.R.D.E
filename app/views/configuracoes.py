"""Configurações: sensibilidade, detalhes técnicos, lixeira automática e informações do sistema."""

import streamlit as st

import ui
from src import paths
from src.recycling import CATEGORIES

ui.hero("Configurações", "Ajustes opcionais. Para usar no dia a dia, não precisa mudar nada aqui.")
cfg = ui.settings()

with st.container(border=True):
    st.markdown("#### 🎯 Sensibilidade")
    value = st.pills(
        "Sensibilidade", list(ui.SENSITIVITY), default=cfg["sensibilidade"], required=True,
        label_visibility="collapsed",
    )
    if value != cfg["sensibilidade"]:
        ui.save_setting("sensibilidade", value)
    st.caption("**Detectar mais**: acha mais objetos, mas pode marcar coisas que não são lixo. "
               "**Só com certeza**: mostra menos, mas erra menos.")

with st.container(border=True):
    st.markdown("#### 🔧 Detalhes técnicos")
    tech = st.toggle("Mostrar detalhes técnicos nas análises", value=cfg["detalhes"],
                     help="Mostra a classe que o modelo viu, a confiança e o peso estimado de cada item.")
    if tech != cfg["detalhes"]:
        ui.save_setting("detalhes", tech)

with st.container(border=True):
    st.markdown("#### 🗑️ Lixeira automática (opcional)")
    st.markdown(
        "Só para quem montou uma **lixeira com motores** ligada ao computador (Arduino ou ESP32). Com isto ligado, "
        "a cada análise o sistema manda o comando e a tampa da lixeira certa se abre sozinha. **Se você não tem "
        "esse equipamento, deixe desligado.**"
    )
    controller = ui.get_controller()
    on = st.toggle("Abrir a tampa da lixeira automaticamente", value=cfg["comportas"])
    if on != cfg["comportas"]:
        ui.save_setting("comportas", on)
    names = {"sim": "simulado (nenhum equipamento conectado)", "serial": "cabo USB", "mqtt": "Wi-Fi (MQTT)"}
    st.caption(f"Conexão: **{names.get(controller.backend.name, controller.backend.name)}**. Para ligar um "
               "equipamento de verdade, veja a seção *Integração IoT* do README.")
    with st.expander("Comandos enviados ao equipamento"):
        st.dataframe(
            [{"Lixeira": f"{ui.EMOJI[n]} {n}", "Cor": c.color_name, "Comando": c.command}
             for n, c in CATEGORIES.items()],
            hide_index=True,
        )

with st.container(border=True):
    st.markdown("#### 🧪 Casos duvidosos")
    collector = ui.review()
    if collector is None:
        st.caption("Desligado (WASTE_REVIEW=0).")
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
