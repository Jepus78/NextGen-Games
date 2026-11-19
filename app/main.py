"""API REST de Inferencia Online, Prescripción, AIaaS y Orquestación Batch — NextGen Games."""
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import pika
import redis
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.aiaas_gemini import generate_executive_brief
from src.prescriptive import build_prescription
from src.train import MODEL_PATH, download_from_gcs_if_configured, train_and_save

API_KEY = os.getenv("API_KEY", "nextgen-ep3-secret-key-2026")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
QUEUE_NAME = "inference_queue"

# Métricas Prometheus (MLOps Observabilidad)
INFERENCE_COUNT = Counter(
    "model_inference_count_total",
    "Total de solicitudes de inferencia atendidas",
    ["endpoint", "cache_status"],
)
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Latencia de inferencia del modelo en segundos",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.4, 0.5, 1.0, 2.5),
)
MODEL_LOADED_GAUGE = Gauge(
    "model_loaded_status",
    "Indica si el modelo .pkl está cargado en memoria RAM (1=sí, 0=no)",
)

# Estado en memoria (Singleton + fallback para pruebas unitarias sin Redis/RabbitMQ)
STATE = {
    "bundle": None,
    "redis": None,
    "memory_cache": {},
    "memory_jobs": {},
}


def init_redis():
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=1)
        client.ping()
        return client
    except Exception:
        return None


def load_model_singleton():
    download_from_gcs_if_configured(MODEL_PATH, "models/model_bundle.pkl")
    if not MODEL_PATH.exists():
        train_and_save()
    bundle = joblib.load(MODEL_PATH)
    STATE["bundle"] = bundle
    MODEL_LOADED_GAUGE.set(1)
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_singleton()
    STATE["redis"] = init_redis()
    yield


app = FastAPI(
    title="NextGen Games — Video Game Sales Predictor API (EP3)",
    description=(
        "API REST de inferencia online, análisis prescriptivo, integración AIaaS (Google Gemini API) "
        "y procesamiento asíncrono por lotes (RabbitMQ + Redis) sobre Google Cloud Platform."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# --- Seguridad: Autenticación por API Key ---
def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv("API_KEY", API_KEY)
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida o ausente en header X-API-Key.",
        )
    return x_api_key


# --- Esquemas Pydantic (Guardrails de Entrada) ---
class GamePredictionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120, examples=["Cyber Odyssey 2026"])
    platform: str = Field(..., min_length=2, max_length=20, examples=["PS4"])
    year_of_release: Optional[float] = Field(default=2018.0, ge=1980, le=2030)
    genre: str = Field(..., min_length=2, max_length=40, examples=["Action"])
    publisher: str = Field(default="Ubisoft", min_length=1, max_length=80)
    rating: Optional[str] = Field(default="M", max_length=10)
    critic_score: Optional[float] = Field(default=None, ge=0, le=100, description="Opcional: KNNImputer lo imputa si es nulo")
    critic_count: Optional[float] = Field(default=None, ge=0, le=500)
    user_score: Optional[float] = Field(default=None, ge=0, le=10, description="Opcional: KNNImputer lo imputa si es nulo")
    user_count: Optional[float] = Field(default=None, ge=0, le=20000)
    include_ai_brief: bool = Field(default=False, description="Si es true, invoca Gemini API (AIaaS) para brief ejecutivo")


class BatchPredictionRequest(BaseModel):
    games: List[GamePredictionRequest] = Field(..., min_length=1, max_length=200)


def run_single_inference(req: GamePredictionRequest) -> dict:
    bundle = STATE["bundle"] or load_model_singleton()
    metadata = bundle["metadata"]
    top_publishers = metadata["top_publishers"]
    pub_clean = req.publisher if req.publisher in top_publishers else "Other"

    row = pd.DataFrame(
        [
            {
                "Year_of_Release": req.year_of_release if req.year_of_release is not None else np.nan,
                "Critic_Score": req.critic_score if req.critic_score is not None else np.nan,
                "Critic_Count": req.critic_count if req.critic_count is not None else np.nan,
                "User_Score": req.user_score if req.user_score is not None else np.nan,
                "User_Count": req.user_count if req.user_count is not None else np.nan,
                "Platform": req.platform,
                "Genre": req.genre,
                "Publisher": pub_clean,
                "Rating": req.rating or "Unknown",
            }
        ]
    )

    reg = bundle["regressor"]
    clf = bundle["classifier"]
    iso = bundle["anomaly_detector"]

    pred_log = float(reg.predict(row)[0])
    predicted_sales = round(float(np.expm1(max(0.0, pred_log))), 3)

    tier = str(clf.predict(row)[0])
    probs = clf.predict_proba(row)[0]
    classes = list(clf.named_steps["model"].classes_)
    tier_probabilities = {c: round(float(p), 4) for c, p in zip(classes, probs)}

    X_trans = reg.named_steps["preprocessor"].transform(row)
    is_anomaly = bool(iso.predict(X_trans)[0] == -1)

    genre_avg = metadata["genre_benchmarks"].get(req.genre, 0.50)
    platform_avg = metadata["platform_benchmarks"].get(req.platform, 0.50)

    prescription = build_prescription(
        predicted_sales_m=predicted_sales,
        commercial_tier=tier,
        critic_score=req.critic_score,
        user_score=req.user_score,
        platform=req.platform,
        genre=req.genre,
        is_anomaly=is_anomaly,
    )

    imputed_fields = [
        f
        for f, v in {
            "Year_of_Release": req.year_of_release,
            "Critic_Score": req.critic_score,
            "Critic_Count": req.critic_count,
            "User_Score": req.user_score,
            "User_Count": req.user_count,
        }.items()
        if v is None
    ]

    return {
        "game_name": req.name,
        "platform": req.platform,
        "genre": req.genre,
        "publisher": req.publisher,
        "predicted_global_sales_millions": predicted_sales,
        "commercial_tier": tier,
        "tier_probabilities": tier_probabilities,
        "anomaly_detected": is_anomaly,
        "imputed_with_knn": imputed_fields,
        "interpretability": {
            "genre_historical_avg_millions": genre_avg,
            "platform_historical_avg_millions": platform_avg,
            "vs_genre_ratio": round(predicted_sales / genre_avg, 2) if genre_avg > 0 else 1.0,
        },
        "prescription": prescription,
    }


def cache_key_for(req: GamePredictionRequest) -> str:
    raw = req.model_dump_json(exclude={"include_ai_brief"})
    return "pred:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


@app.get("/health", tags=["Observabilidad"])
def health_check():
    r = STATE["redis"] or init_redis()
    STATE["redis"] = r
    return {
        "status": "healthy",
        "service": "nextgen-sales-predictor-api",
        "model_loaded": STATE["bundle"] is not None,
        "redis_connected": r is not None,
        "aiaas_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "cloud_provider": "Google Cloud Platform (GCP)",
    }


@app.get("/metrics", tags=["Observabilidad"])
def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["Modelo ML"])
def model_info():
    bundle = STATE["bundle"] or load_model_singleton()
    return bundle["metadata"]


@app.post("/predict", tags=["Inferencia Online"])
async def predict_sales(
    req: GamePredictionRequest,
    _: str = Depends(verify_api_key),
):
    start = time.perf_counter()
    key = cache_key_for(req)
    cached = None

    if STATE["redis"] is not None:
        try:
            raw_val = STATE["redis"].get(key)
            if raw_val:
                cached = json.loads(raw_val)
        except Exception:
            pass
    elif key in STATE["memory_cache"]:
        cached = dict(STATE["memory_cache"][key])

    if cached is not None:
        result = cached
        cache_status = "hit"
    else:
        result = run_single_inference(req)
        cache_status = "miss"
        if STATE["redis"] is not None:
            try:
                STATE["redis"].setex(key, 3600, json.dumps(result))
            except Exception:
                pass
        else:
            STATE["memory_cache"][key] = dict(result)

    if req.include_ai_brief:
        result = dict(result)
        result["aiaas_executive_brief"] = await generate_executive_brief(
            game_name=req.name,
            platform=req.platform,
            genre=req.genre,
            publisher=req.publisher,
            predicted_sales_m=result["predicted_global_sales_millions"],
            commercial_tier=result["commercial_tier"],
            prescription=result["prescription"],
        )

    elapsed = time.perf_counter() - start
    INFERENCE_LATENCY.observe(elapsed)
    INFERENCE_COUNT.labels(endpoint="/predict", cache_status=cache_status).inc()
    result["cache_status"] = cache_status
    result["latency_ms"] = round(elapsed * 1000, 2)
    return result


@app.post("/batch", status_code=status.HTTP_202_ACCEPTED, tags=["Inferencia Asíncrona (EDA)"])
def create_batch_job(
    payload: BatchPredictionRequest,
    _: str = Depends(verify_api_key),
):
    job_id = str(uuid.uuid4())
    job_record = {
        "job_id": job_id,
        "status": "PENDING",
        "total_items": len(payload.games),
        "results": [],
    }

    # Intentar publicar en RabbitMQ + Redis; si no está disponible (ej. test local), procesar con fallback
    try:
        creds = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=creds,
            connection_attempts=1,
            socket_timeout=1.5,
        )
        conn = pika.BlockingConnection(params)
        channel = conn.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)

        if STATE["redis"] is not None:
            STATE["redis"].setex(f"job:{job_id}", 3600, json.dumps(job_record))
        else:
            STATE["memory_jobs"][job_id] = job_record

        message = {
            "job_id": job_id,
            "games": [g.model_dump() for g in payload.games],
        }
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        conn.close()
        return {
            "job_id": job_id,
            "status": "PENDING",
            "mode": "async_rabbitmq",
            "poll_url": f"/job/{job_id}",
        }
    except Exception:
        # Fallback local inmediato cuando RabbitMQ no está corriendo (tests/dev)
        results = [run_single_inference(g) for g in payload.games]
        job_record["status"] = "COMPLETED"
        job_record["results"] = results
        if STATE["redis"] is not None:
            STATE["redis"].setex(f"job:{job_id}", 3600, json.dumps(job_record))
        else:
            STATE["memory_jobs"][job_id] = job_record
        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "mode": "local_fallback",
            "poll_url": f"/job/{job_id}",
        }


@app.get("/job/{job_id}", tags=["Inferencia Asíncrona (EDA)"])
def get_batch_job(job_id: str, _: str = Depends(verify_api_key)):
    if STATE["redis"] is not None:
        raw = STATE["redis"].get(f"job:{job_id}")
        if raw:
            return json.loads(raw)
    if job_id in STATE["memory_jobs"]:
        return STATE["memory_jobs"][job_id]
    raise HTTPException(status_code=404, detail=f"Trabajo '{job_id}' no encontrado.")
