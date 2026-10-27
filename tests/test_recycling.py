import pytest

from src.recycling import CATEGORIES, COCO_RULES, category_from_name


def test_cinco_categorias_com_cores_conama():
    assert list(CATEGORIES) == ["Plástico", "Papel", "Metal", "Vidro", "Orgânico/Rejeito"]
    assert {c.color_name for c in CATEGORIES.values()} == {"vermelha", "azul", "amarela", "verde", "marrom"}
    assert len({c.hex for c in CATEGORIES.values()}) == 5


def test_hex_vira_bgr():
    assert CATEGORIES["Plástico"].bgr == (0x28, 0x28, 0xD6)


@pytest.mark.parametrize(
    "label, category",
    [
        ("bottle", "Plástico"), ("cup", "Plástico"), ("fork", "Plástico"), ("knife", "Plástico"),
        ("spoon", "Plástico"), ("bowl", "Plástico"), ("wine glass", "Vidro"), ("vase", "Vidro"),
        ("scissors", "Metal"), ("book", "Papel"), ("banana", "Orgânico/Rejeito"), ("apple", "Orgânico/Rejeito"),
        ("cell phone", "Orgânico/Rejeito"),
    ],
)
def test_mapeamento_coco(label, category):
    assert COCO_RULES[label].category.name == category
    assert COCO_RULES[label].weight_g > 0


@pytest.mark.parametrize(
    "name, category",
    [
        # Classes do TACO
        ("Plastic bottle", "Plástico"), ("Glass bottle", "Vidro"), ("Drink can", "Metal"), ("Food Can", "Metal"),
        ("Metal bottle cap", "Metal"), ("Plastic bottle cap", "Plástico"), ("Paper straw", "Papel"),
        ("Plastic straw", "Plástico"), ("Drink carton", "Papel"), ("Corrugated carton", "Papel"),
        ("Pizza box", "Papel"), ("Glass jar", "Vidro"), ("Aerosol", "Metal"), ("Pop tab", "Metal"),
        ("Crisp packet", "Plástico"), ("Styrofoam piece", "Plástico"), ("Food waste", "Orgânico/Rejeito"),
        ("Cigarette", "Orgânico/Rejeito"), ("Shoe", "Orgânico/Rejeito"),
        # TrashNet / Roboflow / o próprio modelo treinado
        ("cardboard", "Papel"), ("glass", "Vidro"), ("metal", "Metal"), ("paper", "Papel"),
        ("plastic", "Plástico"), ("trash", "Orgânico/Rejeito"), ("Plástico", "Plástico"),
        ("Orgânico/Rejeito", "Orgânico/Rejeito"), ("PLASTICO", "Plástico"),
    ],
)
def test_categoria_pelo_nome(name, category):
    assert category_from_name(name).name == category
