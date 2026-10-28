import json

import cv2
import numpy as np

from src.dedup import find_duplicates, find_repeats
from src.train import MATERIAL_CLASSES, prepare_material_dataset


def photo(seed: int, size=(96, 128)) -> np.ndarray:
    """Uma "foto" com manchas suaves (ruído puro mudaria de hash ao redimensionar, foto de verdade não)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return cv2.resize(small, size[::-1], interpolation=cv2.INTER_CUBIC)


def save(path, pixels, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def test_acha_a_mesma_foto_redimensionada_e_recomprimida(tmp_path):
    original = save(tmp_path / "ref" / "a.jpg", photo(1))
    other = save(tmp_path / "ref" / "b.jpg", photo(2))
    copy = save(tmp_path / "cand" / "copia.jpg", cv2.resize(photo(1), (200, 150)), quality=60)
    different = save(tmp_path / "cand" / "outra.jpg", photo(3))

    assert find_duplicates([copy, different], [original, other]) == {copy}


def test_hash_parecido_mas_pixels_diferentes_nao_e_copia(tmp_path):
    base = photo(4)
    ref = save(tmp_path / "ref.jpg", base)
    # Mesmo desenho geral (hash quase igual), mas com um objeto no meio: a correlação dos pixels desempata
    changed = base.copy()
    cv2.circle(changed, (64, 48), 30, (255, 255, 255), -1)
    cand = save(tmp_path / "cand.jpg", changed)
    assert find_duplicates([cand], [ref], max_distance=64) == set()


def test_copias_dentro_da_mesma_lista_fica_a_primeira(tmp_path):
    a = save(tmp_path / "1.jpg", photo(5))
    b = save(tmp_path / "2.jpg", photo(5))
    c = save(tmp_path / "3.jpg", photo(6))
    assert find_repeats([a, b, c]) == {b}


def test_dataset_combinado_sem_vazamento_e_com_as_classes_do_trashnet(tmp_path):
    trashnet, trashbox = tmp_path / "trashnet", tmp_path / "trashbox"
    seed = iter(range(100, 1000))
    for split, n in (("train", 3), ("val", 1), ("test", 1)):
        for cls in ("glass", "plastic", "trash"):
            for k in range(n):
                save(trashnet / split / cls / f"{cls}{k}.jpg", photo(next(seed)))
    test_photo = photo(next(seed))
    save(trashbox / "test" / "glass" / "glass 1.jpg", test_photo)
    save(trashbox / "val" / "e-waste" / "e-waste 1.jpg", photo(next(seed)))
    # No treino do TrashBox: uma cópia da foto de teste (tem que sair), uma cópia de uma foto do treino do TrashNet
    # (cópia interna, sai), um lixo eletrônico e um hospitalar (viram "trash") e uma foto corrompida
    save(trashbox / "train" / "glass" / "glass 9.jpg", cv2.resize(test_photo, (160, 120)))
    save(trashbox / "train" / "plastic" / "plastic 9.jpg", cv2.imread(str(trashnet / "train" / "plastic" / "plastic0.jpg")))
    save(trashbox / "train" / "e-waste" / "e-waste 9.jpg", photo(next(seed)))
    save(trashbox / "train" / "medical" / "medical 9.jpg", photo(next(seed)))
    (trashbox / "train" / "metal").mkdir(parents=True)
    (trashbox / "train" / "metal" / "metal 9.jpg").write_bytes(b"not a jpeg")

    out = prepare_material_dataset(
        tmp_path / "out", trashnet, {s: trashbox / s for s in ("train", "val", "test")}, max_per_class=4,
    )
    info = json.loads((out / "info.json").read_text(encoding="utf-8"))
    assert info["treino_repete_validacao_ou_teste"] == 1
    assert info["copias_dentro_do_treino"] == 1
    assert info["ilegiveis_no_treino"] == 1

    train = {p.name: sorted(x.name for x in p.iterdir()) for p in (out / "train").iterdir()}
    assert set(train) <= set(MATERIAL_CLASSES)
    assert "trashbox_glass_glass_9.jpg" not in train["glass"]  # a cópia do teste saiu do treino
    assert train["plastic"] == [f"trashnet_plastic_plastic{k}.jpg" for k in range(3)]
    # max_per_class=4: as 3 do TrashNet entram sempre, e sobra 1 vaga para o eletrônico ou o hospitalar
    assert len(train["trash"]) == 4 and sum(n.startswith("trashnet_") for n in train["trash"]) == 3
    assert sorted(x.name for x in (out / "val" / "trash").iterdir()) == ["trashbox_e-waste_e-waste_1.jpg",
                                                                          "trashnet_trash_trash0.jpg"]
    # Os testes ficam intactos e separados por fonte
    assert (out / "test_trashbox" / "glass" / "trashbox_glass_glass_1.jpg").is_file()
    assert len(list((out / "test_trashnet").rglob("*.jpg"))) == 3
    assert info["fotos"]["test_trashbox"]["glass"] == 1
