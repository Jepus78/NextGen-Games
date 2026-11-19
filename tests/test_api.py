"""Suite de 12 pruebas unitarias y de integración para NextGen Games EP3 (IE10)."""
import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("API_KEY", "nextgen-ep3-secret-key-2026")
from app.main import app  # noqa: E402
from src.prescriptive import build_prescription  # noqa: E402
from src.preprocess import assign_commercial_tier, load_unified_dataset  # noqa: E402

HEADERS = {"X-API-Key": os.environ["API_KEY"]}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# 1. Prueba unitaria: unificación exacta de los 3 CSV (17.766 filas x 16 columnas)
def test_01_dataset_unification_shape():
    df = load_unified_dataset("data")
    assert df.shape == (17766, 16)
    assert "Global_Sales" in df.columns


# 2. Prueba unitaria: clasificación de tiers comerciales
def test_02_commercial_tier_thresholds():
    assert assign_commercial_tier(0.05) == "Bajo"
    assert assign_commercial_tier(0.35) == "Medio"
    assert assign_commercial_tier(1.25) == "Alto"


# 3. Prueba unitaria: motor prescriptivo para título Tier A
def test_03_prescriptive_engine_high_tier():
    p = build_prescription(2.5, "Alto", 88.0, 8.5, "PS4", "Action", False)
    assert "PRIORITARIA" in p["license_decision"]
    assert p["suggested_marketing_budget_kusd"] > 0


# 4. Prueba unitaria: alerta de anomalía en motor prescriptivo
def test_04_prescriptive_engine_anomaly_alert():
    p = build_prescription(0.1, "Bajo", 40.0, 4.0, "PC", "Strategy", True)
    assert any("Anomalía detectada" in a for a in p["recommended_actions"])


# 5. Integración: endpoint /health responde 200 y modelo cargado
def test_05_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


# 6. Integración: /model/info expone métricas de regresión y clasificación
def test_06_model_info_metrics(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "rmse_millions" in data["regression_metrics"]
    assert "f1_weighted" in data["classification_metrics"]


# 7. Seguridad: /predict sin API Key rechaza con 401
def test_07_predict_unauthorized_without_key(client):
    r = client.post("/predict", json={"name": "Test", "platform": "PS4", "genre": "Action"})
    assert r.status_code == 401


# 8. Integración: /predict válido retorna predicción, tier y prescripción
def test_08_predict_valid_game(client):
    payload = {
        "name": "Grand Action Test",
        "platform": "PS4",
        "year_of_release": 2018,
        "genre": "Action",
        "publisher": "Rockstar Games",
        "rating": "M",
        "critic_score": 92.0,
        "critic_count": 60,
        "user_score": 8.4,
        "user_count": 900,
        "include_ai_brief": False,
    }
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_global_sales_millions"] >= 0
    assert data["commercial_tier"] in {"Bajo", "Medio", "Alto"}
    assert "license_decision" in data["prescription"]


# 9. Integración: /predict con nulos imputa automáticamente vía KNNImputer
def test_09_predict_with_nulls_knn_imputation(client):
    payload = {
        "name": "Mystery Title",
        "platform": "XOne",
        "genre": "Shooter",
        "publisher": "Activision",
        "critic_score": None,
        "user_score": None,
    }
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "Critic_Score" in data["imputed_with_knn"]
    assert "User_Score" in data["imputed_with_knn"]


# 10. Guardrail de entrada: Critic_Score > 100 rechazado con 422
def test_10_guardrail_invalid_score_range(client):
    payload = {
        "name": "Invalid Game",
        "platform": "PS4",
        "genre": "Sports",
        "critic_score": 150.0,
    }
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 422


# 11. Integración AIaaS: /predict con include_ai_brief=True responde con Gemini o fallback
def test_11_predict_with_aiaas_brief(client):
    payload = {
        "name": "AIaaS Brief Test",
        "platform": "PS4",
        "genre": "Role-Playing",
        "publisher": "Square Enix",
        "critic_score": 85.0,
        "user_score": 8.0,
        "include_ai_brief": True,
    }
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 200
    assert "aiaas_executive_brief" in r.json()
    assert len(r.json()["aiaas_executive_brief"]["brief"]) > 20


# 12. Integración Batch + Observabilidad: /batch, /job/{id} y /metrics
def test_12_batch_and_prometheus_metrics(client):
    batch_payload = {
        "games": [
            {"name": "Game A", "platform": "PS4", "genre": "Sports", "publisher": "Electronic Arts"},
            {"name": "Game B", "platform": "XOne", "genre": "Racing", "publisher": "Microsoft Studios"},
        ]
    }
    rb = client.post("/batch", json=batch_payload, headers=HEADERS)
    assert rb.status_code == 202
    job_id = rb.json()["job_id"]

    rj = client.get(f"/job/{job_id}", headers=HEADERS)
    assert rj.status_code == 200
    assert rj.json()["total_items"] == 2

    rm = client.get("/metrics")
    assert rm.status_code == 200
    assert "model_inference_count_total" in rm.text
