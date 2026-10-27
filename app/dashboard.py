"""V.E.R.D.E.: interface para identificar resíduos, acompanhar o impacto e ensinar o sistema.

    streamlit run app/dashboard.py      (ou: iniciar)

Cada página fica em app/views/; as peças compartilhadas, em app/ui.py.
"""

import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
sys.path[:0] = [str(APP.parent), str(APP)]  # src/ e ui.py, de qualquer pasta

import ui  # noqa: E402

st.set_page_config(page_title="V.E.R.D.E.", page_icon="♻️", layout="wide", initial_sidebar_state="collapsed")
ui.style()
ui.start_engine()  # a IA carrega em segundo plano enquanto a página já aparece
st.logo(str(APP / "static" / "logo.svg"), size="large")

page = st.navigation(
    [
        st.Page("views/analisar.py", title="Analisar", icon="📷", default=True),
        st.Page("views/impacto.py", title="Meu impacto", icon="🌳"),
        st.Page("views/ensinar.py", title="Ensinar o sistema", icon="🎓"),
        st.Page("views/configuracoes.py", title="Configurações", icon="⚙️"),
    ],
    position="top",  # menu no topo: mais simples, e no celular vira um botão de menu
)
page.run()
