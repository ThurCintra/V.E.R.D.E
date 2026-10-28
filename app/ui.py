"""Peças compartilhadas pelas páginas do dashboard: recursos em cache, configurações, estilo e componentes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import cv2
import pandas as pd
import streamlit as st

from src import paths
from src.corrections import NOT_WASTE, OPTIONS, CorrectionStore
from src.database import Database
from src.detector import DEEP_SCAN_RANGE, MODES, Detector, annotate
from src.material import crop
from src.recycling import CATEGORIES, DEFAULT_WEIGHT_G, label_pt
from src.review import ReviewCollector

EMOJI = {"Plástico": "🟥", "Papel": "🟦", "Metal": "🟨", "Vidro": "🟩", "Orgânico/Rejeito": "🟫", NOT_WASTE: "⬜"}
DEFAULTS = {
    "um_objeto": False, "detalhes": False,
    # Painel de calibração de precisão
    "modo": "alta", "confianca": 0.25, "tamanho": [0.05, 70.0], "deep_scan_todos": False, "realce": False,
}
MODE_LABELS = {"rapido": "⚡ Rápido", "alta": "🎯 Alta precisão", "ultra": "🔬 Ultra precisão"}
MODE_HELP = {
    "rapido": "Só o detector (YOLO-World). O mais rápido, bom para vídeo ao vivo num PC simples.",
    "alta": "Detector + especialista de textura (40% forma, 60% textura). Separa PET de vidro, lata de copo.",
    "ultra": "Alta precisão + Deep Scan: objetos duvidosos ganham uma segunda análise detalhada "
             "(Gemini, se houver GEMINI_API_KEY; senão, análise local de 8 variações do recorte).",
}


# ---------------------------------------------------------------- recursos e configurações
@st.cache_resource
def get_detector() -> Detector:
    return Detector.from_env()


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


@st.cache_resource
def get_review(path: str) -> ReviewCollector | None:
    return ReviewCollector.from_env(path)


def db() -> Database:
    return get_db(str(paths.DB_PATH))


def review() -> ReviewCollector | None:
    return get_review(str(paths.REVIEW_DIR))


def _settings_file():
    return paths.STORAGE_DIR / "configuracoes.json"


def settings() -> dict:
    """Configurações do usuário, guardadas em storage/configuracoes.json (valem depois de reiniciar). No site
    público, valem só para a sessão de cada visitante: o arquivo seria o mesmo para todos."""
    if "_cfg" not in st.session_state:
        cfg = dict(DEFAULTS)
        if not paths.PUBLIC:
            try:
                cfg.update(json.loads(_settings_file().read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        st.session_state["_cfg"] = cfg
    return st.session_state["_cfg"]


def save_setting(name: str, value) -> None:
    cfg = settings()
    cfg[name] = value
    if paths.PUBLIC:
        return
    _settings_file().parent.mkdir(parents=True, exist_ok=True)
    _settings_file().write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def public_notice(feature: str) -> None:
    """Aviso de recurso que só existe no aplicativo instalado (no site, ele mudaria o sistema de todos)."""
    st.info(f"{feature} Para usar, instale o [aplicativo para Windows]({paths.DESKTOP_APP_URL}).", icon="💻")


class _SessionOnlyCorrections:
    """No site público, a correção vale só para a foto na tela: o recorte não fica guardado no servidor."""

    def add(self, *args, **kwargs) -> None:
        return None

    def remove(self, path) -> None:
        pass


def correction_store():
    return _SessionOnlyCorrections() if paths.PUBLIC else CorrectionStore(paths.CORRECTIONS_DIR)


def detector() -> Detector:
    """O detector com as escolhas do painel de calibração aplicadas."""
    det, cfg = get_detector(), settings()
    det.mode = cfg["modo"] if cfg["modo"] in MODES else "alta"
    det.conf = float(cfg["confianca"])
    lo, hi = cfg["tamanho"]
    det.min_area, det.max_area = max(lo / 100, 0.0005), hi / 100
    det.deep_scan_all = bool(cfg["deep_scan_todos"])
    det.preprocess = bool(cfg["realce"])
    return det


def calibration_panel(current_frame=None) -> None:
    """Painel de Calibração de Precisão (barra lateral). ``current_frame``: função que devolve o quadro atual,
    usado para zerar o fundo."""
    cfg = settings()
    with st.sidebar:
        st.markdown("### 🎚️ Calibração de precisão")
        mode = st.radio(
            "Modo de operação", list(MODE_LABELS), format_func=MODE_LABELS.get,
            index=list(MODE_LABELS).index(cfg["modo"]) if cfg["modo"] in MODE_LABELS else 1,
        )
        st.caption(MODE_HELP[mode])
        if mode != cfg["modo"]:
            save_setting("modo", mode)
        if mode == "ultra":
            all_scan = st.toggle("Deep Scan em todos os objetos", value=cfg["deep_scan_todos"],
                                 help="Desligado: só nos duvidosos (certeza entre 35% e 60%) parados no centro.")
            if all_scan != cfg["deep_scan_todos"]:
                save_setting("deep_scan_todos", all_scan)

        conf = st.slider("Confiança mínima (confidence threshold)", 0.05, 0.90, float(cfg["confianca"]), 0.05,
                         help="Mais baixo acha mais objetos (e erra mais); mais alto mostra só o que tem certeza.")
        if conf != cfg["confianca"]:
            save_setting("confianca", conf)
        size = st.slider("Tamanho do objeto (% da imagem)", 0.0, 100.0, tuple(cfg["tamanho"]), 0.5,
                         help="Ignora o que for menor (ruído) ou maior (parede, mesa, a própria mão) que isso.")
        if list(size) != list(cfg["tamanho"]):
            save_setting("tamanho", list(size))

        det = get_detector()
        st.markdown("**Fundo e iluminação**")
        boost = st.toggle("Realce de contraste (CLAHE)", value=cfg["realce"],
                          help="Para cenas escuras ou com reflexo. Nos testes, não melhorou fotos bem iluminadas "
                               "(e piorou um pouco com vários objetos): ligue só se ajudar no seu caso.")
        if boost != cfg["realce"]:
            save_setting("realce", boost)
        if paths.PUBLIC:  # a calibração fica no detector, que é um só para todos os visitantes
            st.caption("Zerar o fundo da câmera fica só no aplicativo instalado.")
            st.caption(f"Motor: {det.description}")
            return
        st.caption("Com a câmera vendo a cena **vazia**, toque em Zerar fundo: o que fizer parte dela é ignorado e o "
                   "realce de contraste se ajusta à luz.")
        if st.button("🧹 Zerar fundo e calibrar luz", width="stretch", disabled=current_frame is None):
            frame = current_frame() if current_frame else None
            if frame is None:
                st.warning("Não há imagem da cena para calibrar.")
            else:
                det.calibrate(frame)
                st.success(f"Fundo calibrado (realce {det.calibration.clip_limit:.1f}).")
        if det.calibration is not None and st.button("Desfazer calibração", width="stretch"):
            det.calibrate(None)
            st.rerun()
        if det.calibration is not None:
            st.caption("✅ Fundo calibrado: o que é igual à cena vazia é ignorado.")
        st.caption(f"Motor: {det.description}" + (f" · ⚠️ {det.fallback_note}" if det.fallback_note else ""))


def first_time(key: str) -> bool:
    """True só na primeira vez nesta sessão. O Streamlit reexecuta a página a cada clique: sem isso, a mesma
    foto seria registrada no histórico de novo a cada botão apertado."""
    done = st.session_state.setdefault("_feito", set())
    if key in done:
        return False
    done.add(key)
    return True


def image_key(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:12]


def rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def thumbnail(image_bgr, size: int = 160):
    """Miniatura quadrada (o objeto centralizado sobre branco), para a lista de itens ficar alinhada."""
    import numpy as np

    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    small = cv2.resize(image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    return rgb(canvas)


# ---------------------------------------------------------------- visual
CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1200px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  .verde-hero h1 { font-size: 2rem; margin: 0 0 .2rem 0; }
  .verde-hero p { color: #4B5D52; font-size: 1.05rem; margin: 0 0 1.2rem 0; }
  .verde-card { background: #FFFFFF; border: 1px solid #E1EAE0; border-radius: 18px; padding: 18px 20px;
                box-shadow: 0 1px 2px rgba(20, 50, 30, .05); height: 100%; }
  .verde-step { display: flex; gap: 14px; align-items: flex-start; }
  .verde-step b { font-size: 1.05rem; }
  .verde-step .n { flex: 0 0 34px; height: 34px; border-radius: 50%; background: #1F8A4C; color: #fff;
                   display: flex; align-items: center; justify-content: center; font-weight: 700; }
  .verde-step p { margin: .15rem 0 0 0; color: #4B5D52; font-size: .95rem; }
  .verde-bin { border-radius: 18px; padding: 18px 22px; color: #fff; margin-bottom: 10px; }
  .verde-bin .small { opacity: .9; font-size: .9rem; margin: 0; }
  .verde-bin .big { font-size: 1.45rem; font-weight: 700; margin: 0; line-height: 1.25; }
  .verde-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 14px 0; }
  .verde-chip { background: #FFFFFF; border: 1px solid #E1EAE0; border-radius: 999px; padding: 5px 12px;
                font-size: .95rem; }
  .verde-muted { color: #5F7166; font-size: .92rem; }
  .verde-impact .value { font-size: 1.9rem; font-weight: 750; margin: 0; color: #1C2B22; }
  .verde-impact .label { margin: 0; color: #4B5D52; }
  .verde-impact .eq { margin: .35rem 0 0 0; color: #1F8A4C; font-size: .93rem; }
  div[data-testid="stExpander"] details { border-radius: 16px; background: #FFFFFF; }
  .verde-probbar { display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: #E8EEE6;
                   margin: 6px 0 3px 0; }
  .verde-probbar span { height: 100%; }
  .verde-problabel { font-size: .85rem; color: #4B5D52; margin: 0 0 6px 0; }
  .verde-scan { background: #F0F7F2; border-left: 4px solid #1F8A4C; border-radius: 8px; padding: 6px 10px;
                font-size: .9rem; margin: 4px 0 6px 0; }
  /* Botões de escolha quebram linha em vez de rolar para o lado (no celular, sumiam opções) */
  div[data-testid="stButtonGroup"] [role="radiogroup"] { flex-wrap: wrap !important; overflow: visible !important;
                                                          row-gap: 6px; }
</style>
"""


def style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='verde-hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for i, (col, (title, text)) in enumerate(zip(cols, items), start=1):
        col.markdown(
            f"<div class='verde-card verde-step'><div class='n'>{i}</div><div><b>{title}</b><p>{text}</p></div></div>",
            unsafe_allow_html=True,
        )


def bin_card(category: str, count: int | None = None) -> None:
    """Faixa grande na cor da lixeira: "Coloque na lixeira VERMELHA"."""
    cat = CATEGORIES[category]
    ink = "#1C2B22" if category == "Metal" else "#FFFFFF"  # texto escuro no amarelo
    what = f"{count} {'item' if count == 1 else 'itens'} de {category}" if count else category
    st.markdown(
        f"<div class='verde-bin' style='background:{cat.hex};color:{ink}'>"
        f"<p class='small'>{what}</p><p class='big'>Lixeira {cat.color_name.upper()}</p></div>",
        unsafe_allow_html=True,
    )


def chips(counts: dict[str, int], show_zero: bool = False) -> None:
    html = "".join(
        f"<span class='verde-chip'>{EMOJI[name]} {name}" + (f" <b>×{n}</b>" if n else "") + "</span>"
        for name, n in counts.items()
        if n or show_zero
    )
    if html:
        st.markdown(f"<div class='verde-chips'>{html}</div>", unsafe_allow_html=True)


def legend() -> None:
    """As 5 lixeiras e suas cores, em linha."""
    chips({name: 0 for name in CATEGORIES}, show_zero=True)


def prob_bar(det) -> str:
    """Barra de certeza por material (cores das lixeiras) + texto: "88% Plástico | 10% Vidro | 2% Metal"."""
    if not det.probs:
        return ""
    ranked = sorted(det.probs.items(), key=lambda kv: -kv[1])
    bar = "".join(
        f"<span style='width:{p * 100:.1f}%;background:{CATEGORIES[c].hex}' title='{c} {p:.0%}'></span>"
        for c, p in ranked if p >= 0.005
    )
    text = " | ".join(f"{p:.0%} {c.split('/')[0]}" for c, p in ranked[:3] if p >= 0.01)
    return f"<div class='verde-probbar'>{bar}</div><p class='verde-problabel'>{text}</p>"


def scan_note(det) -> str:
    if not det.scan:
        return ""
    who = "IA Gemini" if det.scan.source == "gemini" else "análise detalhada"
    return f"<div class='verde-scan'>🔬 <b>{det.scan.justification}</b><br><small>Deep Scan ({who})</small></div>"


# ---------------------------------------------------------------- resultado de uma foto
def _fmt(name: str) -> str:
    return f"{EMOJI[name]} {name.split('/')[0]}"


def _on_pick(store, image, det, i: int, key: str) -> None:
    """O usuário tocou numa categoria: guarda a correção (ou desfaz, se voltou à original)."""
    fixes = st.session_state.setdefault(f"fix_{key}", {})
    picked = st.session_state[f"pick_{key}_{i}_{det.category}"]
    if i in fixes:
        store.remove(fixes.pop(i)[1])
    if picked != det.category:
        fixes[i] = (picked, store.add(image, det.box, det.category, picked, det.label, "dashboard"))


def _items_with_corrections(image, detections, key: str):
    """Lista os itens, cada um com os botões de categoria. Devolve as detecções corrigidas."""
    fixes = st.session_state.setdefault(f"fix_{key}", {})
    store = correction_store()
    st.markdown("**Itens encontrados** · <span class='verde-muted'>errou? toque na categoria certa</span>",
                unsafe_allow_html=True)
    for i, det in enumerate(detections):
        with st.container(border=True):
            thumb, body = st.columns([1, 4], vertical_alignment="center")
            thumb.image(thumbnail(crop(image, det.box)), width=78)
            with body:
                name = label_pt(det.label).capitalize()
                state = ""
                if i in fixes:
                    fixed = "corrigido" if paths.PUBLIC else "correção salva"
                    state = f" · ✅ {fixed}" if fixes[i][0] != det.category else " · ✅ confirmado"
                st.markdown(f"**#{i + 1} {name}**<span class='verde-muted'>{state}</span>" + prob_bar(det)
                            + scan_note(det), unsafe_allow_html=True)
                st.pills(
                    f"Categoria do item {i + 1}", OPTIONS, default=fixes.get(i, (det.category,))[0],
                    # A categoria entra na chave: se o Deep Scan mudar a categoria, o botão marcado acompanha
                    format_func=_fmt, key=f"pick_{key}_{i}_{det.category}", required=True,
                    label_visibility="collapsed",
                    on_change=_on_pick, args=(store, image, det, i, key),
                )
    pending = [i for i in range(len(detections)) if i not in fixes]
    if pending and st.button(
        f"👍 Os outros {len(pending)} estão certos" if len(pending) < len(detections) else "👍 Está tudo certo",
        key=f"ok_{key}", help="Confirma os itens que você não corrigiu."
        + ("" if paths.PUBLIC else " Isso também ensina o sistema."),
    ):
        for i in pending:
            det = detections[i]
            fixes[i] = (det.category, store.add(image, det.box, det.category, det.category, det.label, "dashboard"))
        st.rerun()

    result = []
    for i, det in enumerate(detections):
        fix = fixes.get(i, (None,))[0]
        if fix == NOT_WASTE:
            continue
        if fix and fix != det.category:
            det = replace(det, category=fix, weight_g=DEFAULT_WEIGHT_G[fix])
        result.append(det)
    return result


def _deep_scan_photo(det, image, detections, data_key: str):
    """Modo ultra numa foto: Deep Scan nos itens duvidosos (ou em todos), guardado por foto para não repetir."""
    cache = st.session_state.setdefault(f"scan_{data_key}", {})
    out = []
    for i, d in enumerate(detections):
        confidence = max(d.probs.values()) if d.probs else d.confidence
        doubtful = DEEP_SCAN_RANGE[0] <= confidence <= DEEP_SCAN_RANGE[1]
        if i not in cache and (det.deep_scan_all or doubtful):
            with st.spinner(f"🔬 Deep Scan do item {i + 1}..."):
                cache[i] = det.deep_scan(image, d)
        out.append(cache.get(i, d))
    return out


def technical_table(detections) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "#": i + 1,
                    "Categoria": d.category,
                    "Classe": d.label,
                    "Confiança": f"{d.confidence:.0%}",
                    "Material (2ª etapa)": f"{d.material} {d.material_confidence:.0%}" if d.material else "-",
                    "Peso estimado (g)": d.weight_g,
                }
                for i, d in enumerate(detections)
            ]
        ),
        hide_index=True,
    )


def show_result(image, data_key: str, source: str) -> None:
    """Analisa a foto e mostra: imagem marcada, lixeira de cada item, correção e registro no histórico."""
    cfg = settings()
    det = detector()
    st.session_state["ultima_imagem"] = image  # para "Zerar fundo"
    detections = det.detect(image, waste_only=True, single_item=cfg["um_objeto"])
    if det.mode == "ultra" and det.scanner is not None:
        detections = _deep_scan_photo(det, image, detections, data_key)
    collector = review()
    if collector and first_time(f"review_{data_key}"):
        collector.maybe_save(image, detections, source, cfg["um_objeto"])

    left, right = st.columns([1.15, 1], gap="large")
    if not detections:
        left.image(rgb(image), width="stretch")
        with right:
            st.markdown("### Não encontrei nenhum resíduo")
            st.markdown(
                "Tente uma foto mais de perto, com o objeto inteiro aparecendo e boa luz. Se a foto tem um objeto só, "
                "ligue **Só um objeto por foto**: assim o sistema olha a foto inteira."
            )
        return

    with right:
        # A lista de itens depende da calibração: as correções ficam presas a essa combinação
        key = (f"{data_key}-{det.conf}-{int(cfg['um_objeto'])}-{det.mode}-{det.min_area}-{det.max_area}"
               f"-{int(det.deep_scan_all)}")
        detections = _items_with_corrections(image, detections, key)
    with left:
        st.image(rgb(annotate(image, detections, list(range(1, len(detections) + 1)), compact=True)), width="stretch")
        if not detections:
            st.info("Você marcou todos os itens como \"não é resíduo\".")
            return
        counts = {name: sum(d.category == name for d in detections) for name in CATEGORIES}
        present = [name for name, n in counts.items() if n]
        if len(present) == 1:
            bin_card(present[0], counts[present[0]])
        else:
            st.markdown(f"#### {len(detections)} itens, {len(present)} lixeiras diferentes")
            chips(counts)
            for name in present:
                bin_card(name, counts[name])

        if st.button("✅ Registrar no meu impacto", type="primary", key=f"reg_{data_key}", width="stretch"):
            if first_time(f"db_{data_key}"):
                db().add_detections(detections, source=source)
            st.session_state[f"registered_{data_key}"] = True
        if st.session_state.get(f"registered_{data_key}"):
            st.success("Registrado! Os números aparecem em **Meu impacto**.")
        if cfg["detalhes"]:
            with st.expander("Detalhes técnicos"):
                technical_table(detections)
