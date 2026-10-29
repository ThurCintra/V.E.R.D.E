"""V.E.R.D.E.: API REST para classificar resíduos.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Documentação interativa em http://localhost:8000/docs.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.database import Database
from src.detector import Detector, best_waste, load_image
from src.impact import compute_impact
from src.recycling import CATEGORIES
from src.review import ReviewCollector, review_reasons

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
_UNSET = object()


def create_app(
    detector: Detector | None = None,
    db: Database | None = None,
    review: ReviewCollector | None = _UNSET,
) -> FastAPI:
    """Os parâmetros permitem injetar dependências nos testes; em produção vêm do ambiente."""
    single_item_default = os.environ.get("WASTE_SINGLE_ITEM", "0") == "1"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = detector or Detector.from_env()
        app.state.db = db or Database()
        app.state.review = ReviewCollector.from_env() if review is _UNSET else review
        yield
        if db is None:
            app.state.db.close()

    app = FastAPI(
        title="V.E.R.D.E. - API",
        version="1.0.0",
        description="Classifica resíduos nas 5 categorias da coleta seletiva e indica a lixeira certa.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        state = request.app.state
        return {
            "status": "ok",
            "model": state.detector.model_path.name,
            "detector": state.detector.backend,
            "mode": state.detector.mode,
            "texture": state.detector.texture_name,
            "pipeline": state.detector.description,
        }

    @app.get("/categories")
    def categories():
        return [{"name": c.name, "color": c.color_name, "hex": c.hex} for c in CATEGORIES.values()]

    @app.post("/detect")
    async def detect(
        request: Request,
        file: UploadFile = File(..., description="Imagem JPEG/PNG"),
        save: bool = Query(True, description="Grava os resíduos detectados no histórico"),
        waste_only: bool = Query(True, description="Esconde o que não é resíduo (pessoas, carros…)"),
        single_item: bool = Query(
            single_item_default,
            description="Um resíduo por foto (lixeira): se o detector não achar nada, classifica a foto inteira",
        ),
    ):
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Imagem maior que 15 MB")
        try:
            image = load_image(data)
        except ValueError as err:
            raise HTTPException(400, str(err)) from err

        state = request.app.state
        detections = await run_in_threadpool(state.detector.detect, image, waste_only, single_item)
        if save:
            state.db.add_detections(detections, source="api")
        reasons = review_reasons(detections, state.review.low_conf if state.review else 0.5, single_item)
        saved = bool(state.review and reasons and state.review.maybe_save(image, detections, "api", single_item))

        best = best_waste(detections)
        action_json = None
        if best:
            cat = CATEGORIES[best.category]
            action_json = {
                "category": cat.name,
                "bin": f"{cat.name} (lixeira {cat.color_name})",
                "label": best.label,
                "confidence": round(best.confidence, 4),
            }
        h, w = image.shape[:2]
        return {
            "model": state.detector.backend,
            "mode": state.detector.mode,
            "image": {"width": w, "height": h},
            "count": len(detections),
            "detections": [d.to_dict() for d in detections],
            "action": action_json,
            "review": {"needed": bool(reasons), "reasons": reasons, "saved": saved},
        }

    @app.get("/impact")
    def impact(request: Request, days: int | None = Query(None, ge=1, description="Últimos N dias; vazio = tudo")):
        start = datetime.now(timezone.utc) - timedelta(days=days) if days else None
        return compute_impact(request.app.state.db.totals_by_category(start))

    @app.get("/review")
    def review_status(request: Request):
        """Quantos casos duvidosos estão guardados para revisão."""
        collector = request.app.state.review
        return {"enabled": collector is not None, "count": collector.count() if collector else 0}

    @app.get("/review/export")
    def review_export(request: Request):
        """Os casos para revisão num zip (imagens + rótulos YOLO), para corrigir no Roboflow/CVAT."""
        collector = request.app.state.review
        if collector is None:
            raise HTTPException(404, "Coleta de casos para revisão desligada (WASTE_REVIEW=0)")
        return Response(
            collector.export_zip(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="revisao.zip"'},
        )

    return app


app = create_app()
