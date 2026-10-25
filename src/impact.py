"""Calculadora de impacto ambiental da reciclagem.

Os fatores são estimativas por kg reciclado em vez de ir para o aterro, na ordem de grandeza de referências
como o EPA WARM (emissões) e o CEMPRE (energia e água). Variam muito com o tipo exato do material (alumínio
poupa bem mais que aço, papel branco mais que papelão); ajuste ``FACTORS`` para a realidade da sua coleta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.recycling import METAL, ORGANICO, PAPEL, PLASTICO, VIDRO


@dataclass(frozen=True)
class Factors:
    co2_kg: float  # CO2 equivalente evitado por kg
    energy_kwh: float  # energia economizada por kg
    water_l: float  # água poupada por kg


FACTORS: dict[str, Factors] = {
    PLASTICO.name: Factors(co2_kg=1.1, energy_kwh=5.3, water_l=2.0),
    PAPEL.name: Factors(co2_kg=3.5, energy_kwh=3.5, water_l=30.0),
    METAL.name: Factors(co2_kg=4.0, energy_kwh=12.0, water_l=6.0),  # mistura de latas de aço e alumínio
    VIDRO.name: Factors(co2_kg=0.3, energy_kwh=0.64, water_l=1.0),
    ORGANICO.name: Factors(co2_kg=0.2, energy_kwh=0.0, water_l=0.0),  # só o que for compostado (evita metano)
}

# Para traduzir os números em algo palpável
CAR_KG_CO2_PER_KM = 0.12  # carro popular a gasolina
HOME_KWH_PER_DAY = 5.5  # consumo médio residencial no Brasil (~165 kWh/mês)
SHOWER_LITERS = 72  # banho de 8 minutos


@dataclass(frozen=True)
class Impact:
    items: int
    weight_kg: float
    co2_kg: float
    energy_kwh: float
    water_l: float

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


def impact_of(category: str, weight_kg: float, items: int = 0) -> Impact:
    f = FACTORS.get(category, Factors(0, 0, 0))
    return Impact(items, weight_kg, f.co2_kg * weight_kg, f.energy_kwh * weight_kg, f.water_l * weight_kg)


def compute_impact(totals: dict[str, dict]) -> dict:
    """``totals`` no formato de ``Database.totals_by_category``: {categoria: {"items", "weight_kg"}}."""
    per_category = {cat: impact_of(cat, t["weight_kg"], t["items"]) for cat, t in totals.items()}
    total = Impact(
        items=sum(i.items for i in per_category.values()),
        weight_kg=sum(i.weight_kg for i in per_category.values()),
        co2_kg=sum(i.co2_kg for i in per_category.values()),
        energy_kwh=sum(i.energy_kwh for i in per_category.values()),
        water_l=sum(i.water_l for i in per_category.values()),
    )
    return {
        "per_category": {cat: i.to_dict() for cat, i in per_category.items()},
        "total": total.to_dict(),
        "equivalents": {
            "car_km": round(total.co2_kg / CAR_KG_CO2_PER_KM, 1),
            "home_days": round(total.energy_kwh / HOME_KWH_PER_DAY, 1),
            "showers": round(total.water_l / SHOWER_LITERS, 1),
        },
    }
