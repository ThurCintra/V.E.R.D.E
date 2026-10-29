import zipfile

import cv2
import numpy as np

from src import big_train


def fake_realwaste(tmp_path, per_class=4):
    """Zip no formato do RealWaste: realwaste-main/RealWaste/<Classe>/<Classe>_<n>.jpg."""
    zip_path = tmp_path / "realwaste.zip"
    ok, jpg = cv2.imencode(".jpg", np.full((32, 32, 3), 128, np.uint8))
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("realwaste-main/README.md", "x")
        for cls in ("Plastic", "Food Organics", "Vegetation"):
            for i in range(per_class):
                z.writestr(f"realwaste-main/RealWaste/{cls}/{cls}_{i}.jpg", jpg.tobytes())
    return zip_path


def test_realwaste_vira_as_classes_do_trashnet(tmp_path):
    out = big_train.prepare_realwaste(tmp_path / "rw", fake_realwaste(tmp_path), val_fraction=0.25, test_fraction=0.25)

    classes = {p.name for split in ("train", "val", "test") for p in (out / split).iterdir()}
    assert classes == {"plastic", "trash"}
    # comida e vegetação vão para trash, cada uma com a sua parte no teste
    test_trash = sorted(p.name for p in (out / "test" / "trash").iterdir())
    assert any(n.startswith("rw_food_organics_") for n in test_trash)
    assert any(n.startswith("rw_vegetation_") for n in test_trash)
    assert sum(1 for _ in out.rglob("*.jpg")) == 12


def test_so_troca_se_melhorar_na_media_sem_cair_em_nenhum(monkeypatch):
    accuracy = {}
    monkeypatch.setattr(big_train, "_category_accuracy", lambda model, items: accuracy[(model, items[0])])
    tests = {"TrashNet": ["t"], "RealWaste": ["r"]}

    accuracy.update({("old", "t"): 0.95, ("new", "t"): 0.945, ("old", "r"): 0.55, ("new", "r"): 0.85})
    accepted, _, scores = big_train.compare("old", "new", tests)
    assert accepted and scores["RealWaste"]["novo"] == 0.85

    accuracy[("new", "t")] = 0.90  # caiu 5 pontos no TrashNet: recusa, mesmo melhorando muito no RealWaste
    accepted, message, _ = big_train.compare("old", "new", tests)
    assert not accepted and "TrashNet" in message
