import json
import shutil

import cv2
import numpy as np
import pytest

from src import paths
from src.corrections import (
    MIN_CORRECTIONS,
    NOT_WASTE,
    CorrectionStore,
    previous_model_path,
    restore_previous,
    retrain,
    split_holdout,
)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Isola storage/ (correções, modelo do usuário, backup) numa pasta temporária."""
    monkeypatch.setattr(paths, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(paths, "USER_MATERIAL_MODEL", tmp_path / "storage" / "material_cls.pt")
    monkeypatch.setattr(paths, "CORRECTIONS_DIR", tmp_path / "storage" / "corrections")
    return tmp_path / "storage"


def test_guarda_o_recorte_na_pasta_da_categoria_certa(storage):
    store = CorrectionStore()
    image = np.zeros((200, 300, 3), np.uint8)
    path = store.add(image, (100, 50, 200, 150), "Plástico", "Metal", "cup")
    assert path.parent.name == "metal" and path.is_file()
    assert cv2.imread(str(path)).shape == (120, 120, 3)  # recorte com 10% de margem
    (entry,) = [json.loads(line) for line in (store.directory / "log.jsonl").read_text().splitlines()]
    assert entry["predicted"] == "Plástico" and entry["correct"] == "Metal" and entry["label"] == "cup"


def test_resumo_e_itens(storage):
    store = CorrectionStore()
    image = np.zeros((50, 50, 3), np.uint8)
    store.add(image, (0, 0, 50, 50), "Plástico", "Metal")
    store.add(image, (0, 0, 50, 50), "Vidro", "Vidro")  # confirmação de acerto
    store.add(image, (0, 0, 50, 50), "Papel", NOT_WASTE)
    summary = store.summary()
    assert summary["total"] == 3 and summary["errors"] == 2
    assert summary["per_category"]["Metal"] == 1 and summary["per_category"][NOT_WASTE] == 1
    assert sorted(cls for _, cls in store.items()) == ["glass", "metal"]  # "não é resíduo" não treina o classificador


def test_categoria_invalida(storage):
    with pytest.raises(ValueError):
        CorrectionStore().add(np.zeros((5, 5, 3), np.uint8), (0, 0, 5, 5), None, "Isopor")


def test_separa_1_em_5_para_validacao_so_com_10_ou_mais():
    few = [(f"{i}.jpg", "metal") for i in range(9)]
    assert split_holdout(few) == (few, [])
    many = [(f"{i}.jpg", "metal") for i in range(10)]
    train, holdout = split_holdout(many)
    assert len(train) == 8 and len(holdout) == 2 and not set(train) & set(holdout)


def test_retreino_exige_um_minimo_de_correcoes(storage):
    with pytest.raises(ValueError, match=str(MIN_CORRECTIONS)):
        retrain(CorrectionStore())


def fake_trashnet(root):
    """Mini TrashNet: as 6 amostras de data/trashnet repetidas em train/val/test."""
    for sample in sorted((paths.DATA_DIR / "trashnet").glob("*.jpg")):
        cls = sample.stem.split("_")[0]
        for split, n in (("train", 3), ("val", 1), ("test", 1)):
            (root / split / cls).mkdir(parents=True, exist_ok=True)
            for k in range(n):
                shutil.copy(sample, root / split / cls / f"{k}_{sample.name}")
    return root


def test_retreino_de_verdade_troca_o_modelo_e_permite_voltar(storage, tmp_path, material_model):
    store = CorrectionStore()
    for sample in sorted((paths.DATA_DIR / "trashnet").glob("*.jpg")):
        image = cv2.imread(str(sample))
        h, w = image.shape[:2]
        category = {"cardboard": "Papel", "glass": "Vidro", "metal": "Metal", "paper": "Papel",
                    "plastic": "Plástico", "trash": "Orgânico/Rejeito"}[sample.stem.split("_")[0]]
        store.add(image, (0, 0, w, h), None, category)

    epochs_seen = []
    result = retrain(
        store, epochs=1, imgsz=64, trashnet_dir=fake_trashnet(tmp_path / "trashnet"), work_dir=tmp_path / "work",
        on_epoch=lambda e, total: epochs_seen.append((e, total)), workers=0, batch=8,
    )
    assert epochs_seen == [(1, 1)]
    assert result.corrections_used == 6 and result.corrections_holdout == 0
    assert 0 <= result.new_test_accuracy <= 1 and result.new_corrections_accuracy is not None
    log = (storage / "retreinos.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["accepted"] == result.accepted

    if result.accepted:  # 1 época a 64 px pode piorar e ser recusada; as duas saídas são válidas
        assert paths.active_material_model() == paths.USER_MATERIAL_MODEL
        assert previous_model_path().read_bytes() == paths.MATERIAL_MODEL.read_bytes()
        assert restore_previous()
        assert paths.active_material_model() == paths.MATERIAL_MODEL  # voltou ao do repositório
    else:
        assert not paths.USER_MATERIAL_MODEL.exists()


def test_restaurar_sem_backup(storage):
    assert restore_previous() is False


def test_segundo_treino_nao_tropeca_nas_sobras_do_primeiro(tmp_path):
    """Regressão: no Windows, a pasta do treino anterior às vezes não pode ser apagada (arquivo em uso) e o
    atalho (hard link) antigo continuava lá: o 2º treino quebrava com "are the same file"."""
    from src.corrections import _fresh_work_dir, _link

    src = tmp_path / "trashnet" / "cardboard1.jpg"
    src.parent.mkdir()
    src.write_bytes(b"foto")
    first = _fresh_work_dir(tmp_path / "retreino")
    _link(src, first / "train" / "cardboard" / "cardboard1.jpg")
    _link(src, first / "train" / "cardboard" / "cardboard1.jpg")  # sobra no mesmo lugar: aproveita, sem erro
    second = _fresh_work_dir(tmp_path / "retreino")
    assert second != first and second.is_dir()
    _link(src, second / "train" / "cardboard" / "cardboard1.jpg")
    assert (second / "train" / "cardboard" / "cardboard1.jpg").read_bytes() == b"foto"
