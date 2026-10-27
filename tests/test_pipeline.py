from src.pipeline import WastePipeline


def test_video_conta_cada_objeto_uma_vez(detector, conveyor_video, coco_samples, db):
    # Itens da foto parada, para comparar
    still = detector.detect(coco_samples["garrafas_banana"], waste_only=True)
    pipeline = WastePipeline(detector, db=db)

    frames = list(pipeline.process_video(conveyor_video))
    assert len(frames) == 30
    counted = sum(pipeline.counts.values())
    per_frame = sum(len(f.detections) for f in frames)

    # A foto atravessa a tela em 30 frames: sem rastreamento seriam ~30x os itens da foto
    assert 1 <= counted <= len(still) + 2
    assert per_frame > 5 * counted
    assert len(db.records()) == counted
    assert {r["source"] for r in db.records()} == {"video"}
    assert all(r["track_id"] is not None for r in db.records())  # os ids do ByteTrack começam em 0
    assert pipeline.counts["Plástico"] >= 1


def test_pular_frames(detector, conveyor_video):
    frames = list(WastePipeline(detector).process_video(conveyor_video, every_n=3))
    assert len(frames) == 10
    assert frames[0].annotated.shape[:2] == (640, 680)
