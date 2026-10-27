"""Aceleração em CPU com o OpenVINO (Intel; roda também em AMD e ARM).

O detector exportado para OpenVINO dá **as mesmas detecções** do PyTorch (99 de 99 imagens no teste, em FP32) e
é mais rápido: 88 → 58 ms por quadro no servidor onde foi medido (um Xeon com AMX, onde o PyTorch já é forte;
em processadores comuns de PC o ganho costuma ser maior). O modelo do YOLOE já vem exportado em
``models/verde_yoloe_openvino_model/``.

- ``WASTE_OPENVINO=0`` desliga (volta ao PyTorch).
- ``WASTE_OV_PRECISION=bf16`` usa a precisão reduzida em processadores que a suportam (Xeon com AMX): 58 → 22 ms
  no mesmo servidor, com 98 de 99 imagens iguais. O padrão é ``f32``, que dá resultados idênticos.

Qualquer falha (pacote ausente, processador sem suporte, versão nova do Ultralytics) volta ao PyTorch.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def openvino_dir(pt_path: str | Path) -> Path:
    pt_path = Path(pt_path)
    return pt_path.with_name(f"{pt_path.stem}_openvino_model")


def enabled() -> bool:
    if os.environ.get("WASTE_OPENVINO", "1") == "0":
        return False
    try:
        import openvino  # noqa: F401
    except ImportError:
        return False
    return True


def export_openvino(pt_path: str | Path, imgsz: int = 640) -> Path:
    """Exporta um modelo .pt para OpenVINO (entrada de tamanho variável, como o PyTorch). Devolve a pasta."""
    from ultralytics import YOLO

    out = YOLO(str(pt_path)).export(format="openvino", imgsz=imgsz, half=False, dynamic=True, verbose=False)
    return Path(out)


def load_openvino(ir_dir: str | Path, task: str, expected_names: list[str] | None = None):
    """O modelo OpenVINO pronto para uso, ou None se não der (aí o chamador usa o PyTorch)."""
    ir_dir = Path(ir_dir)
    if not enabled() or not ir_dir.is_dir():
        return None
    try:
        import numpy as np
        from ultralytics import YOLO

        model = YOLO(str(ir_dir), task=task)
        if expected_names is not None and list(model.names.values()) != expected_names:
            log.warning("O modelo OpenVINO em %s tem outro vocabulário; usando o PyTorch", ir_dir)
            return None
        if os.environ.get("WASTE_OV_PRECISION", "f32").lower() != "bf16":
            _force_fp32()
        model.predict(np.zeros((64, 64, 3), np.uint8), verbose=False)  # já compila (e confere que roda)
        return model
    except Exception as err:
        log.warning("OpenVINO indisponível (%s); usando o PyTorch", err)
        return None


def _force_fp32() -> None:
    """Faz toda compilação do OpenVINO neste processo usar FP32.

    O Ultralytics só força FP32 no Windows; em processadores com AMX (Linux) o OpenVINO usaria bf16 e algumas
    detecções no limite mudariam. Configurar o modelo depois de carregado não basta: o Ultralytics recria o motor
    interno quando os parâmetros da análise mudam, e ele volta ao padrão. Por isso a regra vai na compilação.
    """
    import openvino as ov

    original = ov.Core.compile_model
    if getattr(original, "_verde_fp32", False):
        return

    def compile_model(self, *args, **kwargs):
        config = dict(kwargs.pop("config", None) or {})
        config.setdefault("INFERENCE_PRECISION_HINT", ov.Type.f32)
        return original(self, *args, config=config, **kwargs)

    compile_model._verde_fp32 = True
    ov.Core.compile_model = compile_model
