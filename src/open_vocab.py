"""Estágio 1: detector de vocabulário aberto (YOLO-World ou YOLOE) com um vocabulário de resíduos.

Em vez das 80 classes do COCO (que não têm "lata" nem "caixa de papelão"), o detector procura objetos descritos
em texto: "aluminum can", "cardboard box", "plastic bottle"... Cada descrição já aponta para uma categoria da
coleta seletiva, que vira o "voto de forma" do ensemble.

Montar o vocabulário exige o codificador de texto do modelo (CLIP ViT-B/32 para o YOLO-World, ~340 MB;
MobileCLIP para o YOLOE, ~600 MB), baixado uma vez só. O modelo com o vocabulário é salvo em
``models/verde_<tipo>.pt`` e, a partir daí, roda offline e sem o codificador.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from src.paths import MODELS_DIR
from src.weights import download

log = logging.getLogger(__name__)

# (descrição para o detector, categoria). A ordem não importa; as descrições são em inglês porque os codificadores
# de texto foram treinados em inglês.
WASTE_VOCABULARY: list[tuple[str, str]] = [
    ("plastic bottle", "Plástico"), ("pet bottle", "Plástico"), ("plastic cup", "Plástico"),
    ("plastic bag", "Plástico"), ("plastic container", "Plástico"), ("plastic lid", "Plástico"),
    ("bottle cap", "Plástico"), ("plastic wrapper", "Plástico"), ("styrofoam", "Plástico"),
    ("plastic straw", "Plástico"), ("detergent bottle", "Plástico"), ("yogurt cup", "Plástico"),
    ("cardboard box", "Papel"), ("cardboard", "Papel"), ("paper", "Papel"), ("newspaper", "Papel"),
    ("magazine", "Papel"), ("paper cup", "Papel"), ("paper bag", "Papel"), ("egg carton", "Papel"),
    ("milk carton", "Papel"), ("juice carton", "Papel"), ("book", "Papel"), ("crumpled paper", "Papel"),
    ("aluminum can", "Metal"), ("soda can", "Metal"), ("tin can", "Metal"), ("food can", "Metal"),
    ("metal lid", "Metal"), ("aerosol can", "Metal"), ("aluminum foil", "Metal"), ("metal scrap", "Metal"),
    ("glass bottle", "Vidro"), ("wine bottle", "Vidro"), ("beer bottle", "Vidro"), ("glass jar", "Vidro"),
    ("drinking glass", "Vidro"), ("broken glass", "Vidro"),
    ("food waste", "Orgânico/Rejeito"), ("banana peel", "Orgânico/Rejeito"), ("fruit", "Orgânico/Rejeito"),
    ("apple core", "Orgânico/Rejeito"), ("vegetable", "Orgânico/Rejeito"), ("leftover food", "Orgânico/Rejeito"),
    ("cigarette butt", "Orgânico/Rejeito"), ("used tissue", "Orgânico/Rejeito"), ("diaper", "Orgânico/Rejeito"),
]
# Descrições que existem só para o detector *não* confundir pessoas com resíduo: são descartadas depois.
IGNORED_VOCABULARY = ["person", "human face", "human hand"]

VOCAB_CATEGORY = dict(WASTE_VOCABULARY)
PROMPTS = [p for p, _ in WASTE_VOCABULARY] + IGNORED_VOCABULARY

# O YOLOE é o padrão: o modelo com o vocabulário já vem no repositório (models/verde_yoloe.pt), então não
# precisa baixar codificador de texto nenhum, e foi o único dos dois medido (89,5% no modo normal do teste do
# TrashNet). O YOLO-World (WASTE_DETECTOR=yolo-world) baixa o CLIP (~340 MB) na primeira vez.
KINDS = {
    "yolo-world": ("yolov8s-worldv2.pt", "verde_yoloworld.pt"),
    "yoloe": ("yoloe-11s-seg.pt", "verde_yoloe.pt"),
}


def vocabulary_model_path(kind: str) -> Path:
    return MODELS_DIR / KINDS[kind][1]


def build_vocabulary_model(kind: str, prompts: list[str] = PROMPTS) -> Path:
    """Baixa o modelo base, aplica o vocabulário de resíduos e salva em models/. Precisa de internet uma vez."""
    from ultralytics import YOLO, YOLOE, YOLOWorld

    base_name, out_name = KINDS[kind]
    base = download(MODELS_DIR / base_name, min_size=10_000_000)
    out = MODELS_DIR / out_name
    # O Ultralytics baixa o codificador de texto na pasta atual: dentro de models/, ele fica fora do Git
    with contextlib.chdir(MODELS_DIR):
        if kind == "yolo-world":
            model = YOLOWorld(str(base))
            model.set_classes(prompts)
        else:
            model = YOLOE(str(base))
            model.set_classes(prompts, model.get_text_pe(prompts))
    model.save(str(out))
    # Confere que o vocabulário sobreviveu ao salvar (o arquivo precisa rodar sem o codificador de texto)
    if list(YOLO(str(out)).names.values()) != prompts:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"O vocabulário não foi salvo em {out}")
    return out


def failure_marker(kind: str) -> Path:
    return MODELS_DIR / f".{kind}_falhou"


def load_open_vocab(kind: str, retry: bool = False):
    """O modelo com vocabulário de resíduos, montando-o se preciso. Devolve (YOLO, caminho).

    Se montar já falhou antes (sem internet, codificador bloqueado), não tenta de novo a cada abertura, o que
    custava segundos (ou mais, sem internet) toda vez: ``retry=True`` (``iniciar verificar``) tenta outra vez.
    """
    from ultralytics import YOLO

    path = vocabulary_model_path(kind)
    if path.is_file():
        model = YOLO(str(path))
        if list(model.names.values()) == PROMPTS:
            return model, path
    marker = failure_marker(kind)
    if marker.is_file() and not retry:
        raise RuntimeError(f"a montagem do {kind} falhou antes ({marker.read_text(encoding='utf-8')}); "
                           "rode 'iniciar verificar' para tentar de novo")
    log.info("Montando o vocabulário de resíduos do %s (só na primeira vez)...", kind)
    try:
        build_vocabulary_model(kind)
    except Exception as err:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{type(err).__name__}: {err}"[:300], encoding="utf-8")
        raise
    marker.unlink(missing_ok=True)
    return YOLO(str(path)), path


def is_open_vocab(names: dict[int, str]) -> bool:
    return list(names.values()) == PROMPTS
