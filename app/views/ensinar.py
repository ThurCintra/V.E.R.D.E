"""Ensinar o sistema: as correções guardadas e o botão de re-treino."""

import streamlit as st

import ui
from src import paths
from src.corrections import MIN_CORRECTIONS, CorrectionStore, previous_model_path, restore_previous, retrain

ui.hero("Ensinar o sistema", "Cada correção que você faz vira um exemplo. Com elas, o V.E.R.D.E. aprende a acertar mais.")

store = CorrectionStore(paths.CORRECTIONS_DIR)
summary = store.summary()
usable = len(store.items())

ui.steps(
    [
        ("Corrija os erros", "Em Analisar, toque na categoria certa de cada item. Confirme também os acertos."),
        (f"Junte {MIN_CORRECTIONS} exemplos ou mais", "Quanto mais exemplos e mais variados, mais ele aprende."),
        ("Treine", "Leva alguns minutos. O sistema só troca o modelo se o novo for pelo menos tão bom."),
    ]
)
st.write("")

c = st.columns(3)
for col, value, label in (
    (c[0], summary["total"], "exemplos guardados"),
    (c[1], summary["errors"], "erros corrigidos"),
    (c[2], summary["total"] - summary["errors"], "acertos confirmados"),
):
    col.markdown(f"<div class='verde-card verde-impact'><p class='value'>{value}</p><p class='label'>{label}</p></div>",
                 unsafe_allow_html=True)
st.write("")
ui.chips(summary["per_category"])

with st.container(border=True):
    if usable < MIN_CORRECTIONS:
        st.progress(usable / MIN_CORRECTIONS, f"{usable} de {MIN_CORRECTIONS} exemplos para poder treinar")
    with st.expander("Opções avançadas"):
        epochs = st.slider("Rodadas de treino (épocas)", 3, 20, 8,
                           help="Mais rodadas aprendem mais, mas demoram mais (cerca de 1 minuto cada).")
        st.caption(f"Modelo em uso: `{paths.active_material_model().name}`"
                   + (" (treinado com as suas correções)" if paths.USER_MATERIAL_MODEL.is_file() else " (original)"))
    st.caption("Leva alguns minutos e usa o processador do seu computador. Na primeira vez baixa 42 MB de fotos de "
               "referência (TrashNet).")
    if st.button("🎓 Treinar com as minhas correções", type="primary", disabled=usable < MIN_CORRECTIONS,
                 width="stretch"):
        bar = st.progress(0.0, "Preparando os exemplos...")
        try:
            result = retrain(
                store, epochs=epochs,
                on_epoch=lambda e, total: bar.progress(e / total, f"Aprendendo... rodada {e} de {total}"),
            )
        except Exception as err:  # mostra o erro em vez de derrubar a página
            bar.empty()
            st.error(f"Não foi possível treinar: {err}")
        else:
            bar.progress(1.0, "Pronto")
            if result.accepted:
                st.success("Pronto! O sistema agora usa o que você ensinou.")
                ui.get_detector.clear()  # a próxima análise carrega o modelo novo
            else:
                st.warning("O modelo novo não ficou melhor que o atual, então o atual foi mantido. Corrija mais "
                           "exemplos, de objetos variados, e tente de novo.")
            m = st.columns(2)
            m[0].metric("Acerto geral (fotos de referência)", f"{result.new_test_accuracy:.1%}",
                        f"{(result.new_test_accuracy - result.old_test_accuracy) * 100:+.1f} pontos")
            if result.new_corrections_accuracy is not None:
                m[1].metric("Acerto nas suas correções", f"{result.new_corrections_accuracy:.1%}",
                            f"{(result.new_corrections_accuracy - result.old_corrections_accuracy) * 100:+.1f} pontos")

if previous_model_path().is_file():
    if st.button("↩️ Voltar ao modelo anterior", help="Desfaz o último treino."):
        restore_previous()
        ui.get_detector.clear()
        st.success("Modelo anterior restaurado.")
