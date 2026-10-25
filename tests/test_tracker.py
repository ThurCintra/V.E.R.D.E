from src.detector import Detection
from src.tracker import SimpleTracker, iou


def det(x, label="bottle", category="Plástico", conf=0.8, y=100):
    return Detection(label, conf, (x, y, x + 100, y + 200), category, 30)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 3) == round(50 / 150, 3)


def test_objeto_em_movimento_conta_uma_vez():
    tracker = SimpleTracker(min_hits=3)
    new_total = []
    ids = set()
    for frame in range(30):
        pairs, new = tracker.update([det(10 + frame * 5)])
        ids.update(i for _, i in pairs)
        new_total += new
    assert len(new_total) == 1
    assert ids == {1}
    assert tracker.counts == {"Plástico": 1}


def test_deteccao_de_um_frame_so_nao_conta():
    tracker = SimpleTracker(min_hits=3)
    tracker.update([det(10)])
    for _ in range(5):
        tracker.update([])
    assert tracker.counts == {}


def test_some_por_poucos_frames_e_volta_sem_contar_de_novo():
    tracker = SimpleTracker(min_hits=2, max_missed=5)
    for _ in range(3):
        tracker.update([det(10)])
    for _ in range(3):  # oclusão rápida
        tracker.update([])
    for _ in range(3):
        tracker.update([det(12)])
    assert tracker.counts == {"Plástico": 1}


def test_dois_objetos_ao_mesmo_tempo():
    tracker = SimpleTracker(min_hits=2)
    for f in range(5):
        tracker.update([det(10 + f), det(400 + f, "banana", "Orgânico/Rejeito")])
    assert tracker.counts == {"Plástico": 1, "Orgânico/Rejeito": 1}


def test_categoria_mais_votada_vence_a_hesitacao_do_modelo():
    tracker = SimpleTracker(min_hits=5)
    confirmed = []
    for f in range(5):
        category = "Vidro" if f == 1 else "Plástico"
        confirmed += tracker.update([det(10, category=category)])[1]
    assert confirmed[0].category == "Plástico"
    assert tracker.counts == {"Plástico": 1}


def test_objeto_que_sai_e_outro_que_entra_contam_separado():
    tracker = SimpleTracker(min_hits=2, max_missed=2)
    for _ in range(3):
        tracker.update([det(10)])
    for _ in range(4):
        tracker.update([])
    for _ in range(3):
        tracker.update([det(10)])
    assert tracker.counts == {"Plástico": 2}
