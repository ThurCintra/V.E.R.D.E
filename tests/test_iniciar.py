"""O iniciar.bat não roda aqui (é do Windows), mas dá para conferir a estrutura dele."""

import re

from src.paths import ROOT

BAT = (ROOT / "iniciar.bat").read_text()


def test_toda_chamada_tem_rotulo_e_argumentos():
    labels = set(re.findall(r"^:(\w+)", BAT, re.M))
    calls = re.findall(r"call :(\w+)(.*)$", BAT, re.M)
    assert calls
    for label, args in calls:
        assert label in labels
        if label in ("instalar", "liberar_porta"):
            assert args.strip(), f"call :{label} sem argumento"


def test_sub_rotinas_aparecem_uma_vez_so():
    for label in ("instalar", "liberar_porta"):
        assert len(re.findall(rf"^:{label}\b", BAT, re.M)) == 1
