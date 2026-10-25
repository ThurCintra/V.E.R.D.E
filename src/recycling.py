"""Categorias da coleta seletiva brasileira e mapeamento das classes detectadas para elas.

Cores conforme a Resolução CONAMA nº 275/2001: azul (papel), vermelho (plástico), amarelo (metal),
verde (vidro) e marrom (orgânico). O rejeito (não reciclável, cinza na resolução) fica junto do orgânico,
na 5ª categoria "Orgânico/Rejeito".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass



@dataclass(frozen=True)
class Category:
    name: str
    color_name: str  # cor da lixeira, no feminino ("lixeira amarela")
    hex: str
    command: str  # caractere enviado ao microcontrolador para abrir a comporta

    @property
    def bgr(self) -> tuple[int, int, int]:
        r, g, b = (int(self.hex[i : i + 2], 16) for i in (1, 3, 5))
        return (b, g, r)


PLASTICO = Category("Plástico", "vermelha", "#D62828", "P")
PAPEL = Category("Papel", "azul", "#1D4ED8", "C")  # C de celulose/cartão (P já é do plástico)
METAL = Category("Metal", "amarela", "#F2C200", "M")
VIDRO = Category("Vidro", "verde", "#2A9D3A", "V")
ORGANICO = Category("Orgânico/Rejeito", "marrom", "#7B4A1E", "O")

CATEGORIES: dict[str, Category] = {c.name: c for c in (PLASTICO, PAPEL, METAL, VIDRO, ORGANICO)}
RECYCLABLE = (PLASTICO.name, PAPEL.name, METAL.name, VIDRO.name)


@dataclass(frozen=True)
class CocoRule:
    category: Category
    weight_g: float  # peso médio estimado de um item descartado
    note: str = ""


# Classes do COCO que aparecem no lixo. Pessoas, veículos, animais etc. ficam de fora (categoria None).
COCO_RULES: dict[str, CocoRule] = {
    # Embalagens e utensílios descartáveis
    "bottle": CocoRule(PLASTICO, 30, "garrafa PET; o COCO não separa PET de vidro, o modelo treinado separa"),
    "cup": CocoRule(PLASTICO, 10, "copo descartável"),
    "fork": CocoRule(PLASTICO, 4, "talher descartável"),
    "knife": CocoRule(PLASTICO, 4, "talher descartável"),
    "spoon": CocoRule(PLASTICO, 4, "talher descartável"),
    "bowl": CocoRule(PLASTICO, 20, "pote/marmita descartável"),
    "toothbrush": CocoRule(PLASTICO, 15),
    "frisbee": CocoRule(PLASTICO, 175),
    "sports ball": CocoRule(PLASTICO, 400),
    # Vidro
    "wine glass": CocoRule(VIDRO, 150),
    "vase": CocoRule(VIDRO, 400),
    # Metal
    "scissors": CocoRule(METAL, 50),
    "umbrella": CocoRule(METAL, 350, "armação metálica"),
    # Papel
    "book": CocoRule(PAPEL, 300),
    # Orgânico
    "banana": CocoRule(ORGANICO, 120),
    "apple": CocoRule(ORGANICO, 150),
    "orange": CocoRule(ORGANICO, 150),
    "broccoli": CocoRule(ORGANICO, 150),
    "carrot": CocoRule(ORGANICO, 60),
    "sandwich": CocoRule(ORGANICO, 150),
    "hot dog": CocoRule(ORGANICO, 100),
    "pizza": CocoRule(ORGANICO, 100),
    "donut": CocoRule(ORGANICO, 60),
    "cake": CocoRule(ORGANICO, 100),
    "potted plant": CocoRule(ORGANICO, 500),
    # Rejeito / eletrônicos (vão para ponto de coleta de lixo eletrônico, não para a coleta seletiva)
    "cell phone": CocoRule(ORGANICO, 180, "lixo eletrônico: levar a ponto de coleta"),
    "remote": CocoRule(ORGANICO, 120, "lixo eletrônico: levar a ponto de coleta"),
    "mouse": CocoRule(ORGANICO, 80, "lixo eletrônico: levar a ponto de coleta"),
    "keyboard": CocoRule(ORGANICO, 600, "lixo eletrônico: levar a ponto de coleta"),
    "laptop": CocoRule(ORGANICO, 2000, "lixo eletrônico: levar a ponto de coleta"),
    "teddy bear": CocoRule(ORGANICO, 200),
    "tie": CocoRule(ORGANICO, 50),
}

# Nome em português de cada classe (COCO e TrashNet), para mostrar a quem usa o sistema
LABELS_PT = {
    "bottle": "garrafa", "cup": "copo", "bowl": "pote", "wine glass": "taça", "vase": "vaso", "fork": "garfo",
    "knife": "faca", "spoon": "colher", "scissors": "tesoura", "book": "livro", "toothbrush": "escova de dentes",
    "umbrella": "guarda-chuva", "frisbee": "frisbee", "sports ball": "bola", "banana": "banana", "apple": "maçã",
    "orange": "laranja", "broccoli": "brócolis", "carrot": "cenoura", "sandwich": "sanduíche",
    "hot dog": "cachorro-quente", "pizza": "pizza", "donut": "rosquinha", "cake": "bolo", "potted plant": "planta",
    "cell phone": "celular", "remote": "controle remoto", "mouse": "mouse", "keyboard": "teclado",
    "laptop": "notebook", "teddy bear": "bicho de pelúcia", "tie": "gravata",
    "cardboard": "papelão", "glass": "vidro", "metal": "metal", "paper": "papel", "plastic": "plástico",
    "trash": "rejeito",
    # Vocabulário do detector de vocabulário aberto (src/open_vocab.py)
    "plastic bottle": "garrafa plástica", "pet bottle": "garrafa PET", "plastic cup": "copo plástico",
    "plastic bag": "sacola plástica", "plastic container": "pote plástico", "plastic lid": "tampa plástica",
    "bottle cap": "tampinha", "plastic wrapper": "embalagem plástica", "styrofoam": "isopor",
    "plastic straw": "canudo", "detergent bottle": "frasco de detergente", "yogurt cup": "pote de iogurte",
    "cardboard box": "caixa de papelão", "newspaper": "jornal", "magazine": "revista", "paper cup": "copo de papel",
    "paper bag": "saco de papel", "egg carton": "caixa de ovos", "milk carton": "caixa de leite",
    "juice carton": "caixa de suco", "crumpled paper": "papel amassado", "aluminum can": "lata de alumínio",
    "soda can": "lata de refrigerante", "tin can": "lata de conserva", "food can": "lata de comida",
    "metal lid": "tampa de metal", "aerosol can": "lata de aerossol", "aluminum foil": "papel-alumínio",
    "metal scrap": "sucata", "glass bottle": "garrafa de vidro", "wine bottle": "garrafa de vinho",
    "beer bottle": "garrafa de cerveja", "glass jar": "pote de vidro", "drinking glass": "copo de vidro",
    "broken glass": "vidro quebrado", "food waste": "resto de comida", "banana peel": "casca de banana",
    "fruit": "fruta", "apple core": "miolo de maçã", "vegetable": "legume", "leftover food": "sobra de comida",
    "cigarette butt": "bituca", "used tissue": "lenço usado", "diaper": "fralda",
}


def label_pt(label: str) -> str:
    return LABELS_PT.get(label, label)


# Peso médio por categoria, para modelos customizados (cujas classes já são materiais).
DEFAULT_WEIGHT_G: dict[str, float] = {
    PLASTICO.name: 25,
    PAPEL.name: 60,
    METAL.name: 20,
    VIDRO.name: 250,
    ORGANICO.name: 100,
}

# Palavras-chave para nomes de classes de modelos customizados e de datasets (TACO, TrashNet, Roboflow).
# A ordem importa: "Paper straw" é papel, "Metal bottle cap" é metal, "Food can" é metal.
_KEYWORDS: list[tuple[Category, set[str]]] = [
    (VIDRO, {"glass", "vidro", "jar"}),
    (METAL, {"metal", "can", "cans", "aluminium", "aluminum", "aerosol", "pop", "tin", "steel", "lata"}),
    (PAPEL, {"paper", "papel", "cardboard", "carton", "papelao", "toilet", "magazine", "newspaper", "pizza"}),
    (
        PLASTICO,
        {
            "plastic", "plastico", "styrofoam", "foam", "straw", "lid", "bag", "wrapper", "film", "packet",
            "tub", "tupperware", "rings", "blister", "squeezable", "pet", "polypropylene", "container",
        },
    ),
    (ORGANICO, {"organic", "organico", "food", "biodegradable", "trash", "rejeito", "litter"}),
]


def _tokens(name: str) -> set[str]:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z]+", ascii_name))


def category_from_name(name: str) -> Category:
    """Categoria de uma classe de dataset/modelo de resíduos pelo nome (ex.: 'Plastic bottle' → Plástico).

    O que não bate com nenhum material (bitucas, sapatos, pilhas…) vai para Orgânico/Rejeito.
    """
    if name in CATEGORIES:
        return CATEGORIES[name]
    tokens = _tokens(name)
    for category, keywords in _KEYWORDS:
        if tokens & keywords:
            return category
    return ORGANICO

