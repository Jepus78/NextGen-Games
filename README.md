# 🎮 NextGen Games — Sistema Predictor y Prescriptor de Ventas de Videojuegos (EP3)

**Asignatura:** ITY1102 — Arquitectura de Sistemas de Inteligencia Artificial (Duoc UC)  
**Equipo 3:** Emerson Lemunao • Jesús Muñoz • Joaquín Acuña  
**Infraestructura Cloud:** Google Cloud Platform (GCP Free Tier — Compute Engine + Cloud Storage) + Google AI Studio (Gemini API como AIaaS)

## Arquitectura Implementada
- **Preprocesamiento & ML (`src/train.py`):** Unificación de 3 datasets históricos (17.766 registros x 16 columnas), tratamiento de outliers (P99), imputación matemática con `KNNImputer`, modelo regresor (`HistGradientBoostingRegressor`), clasificador de tier comercial (`HistGradientBoostingClassifier`) y detector de anomalías (`IsolationForest`), serializados en `artifacts/model_bundle.pkl`.
- **API REST (`app/main.py`):** FastAPI con documentación automática Swagger/OpenAPI (`/docs`), autenticación mediante header `X-API-Key`, caché de inferencias en **Redis**, análisis prescriptivo comercial y exportación de métricas Prometheus (`/metrics`).
- **Plataforma AIaaS (`src/aiaas_gemini.py`):** Integración con **Google AI Studio (Gemini API gratuita)** aplicando patrón AI Gateway, reintento exponencial y fallback automático para generar briefs ejecutivos interpretables.
- **Comunicación Asíncrona (`worker/worker.py`):** Arquitectura dirigida por eventos (EDA) con **RabbitMQ** (`inference_queue`, entrega garantizada ACK) y **Redis** para procesamiento por lotes (`POST /batch` → `GET /job/{id}`).
- **API Gateway & Dashboard (`nginx/`, `dashboard/`):** Balanceo `least_conn` + rate limiting (`10 req/s`) en Nginx y dashboard interactivo en **Streamlit** para los 50 analistas comerciales.
- **Observabilidad y Testing (`monitoring/`, `tests/`, `locustfile.py`):** Prometheus + Grafana, 12 pruebas unitarias/integración con `pytest` y pruebas de carga con `Locust`.

## Ejecución Rápida (Local o GCP VM)
```bash
cp .env.example .env
# Opcional: agrega tu GEMINI_API_KEY gratuita de https://aistudio.google.com/apikey en .env

# Modo completo (incluye Prometheus y Grafana — recomendado en PC o GCP e2-small):
docker compose up --build -d

# Modo ligero (recomendado en GCP Free Tier e2-micro de 1 GB RAM):
docker compose -f docker-compose.lite.yml up --build -d
```

## Endpoints Principales
- **Dashboard Streamlit:** `http://<IP>:8080/`
- **Swagger / OpenAPI:** `http://<IP>:8000/docs` (o `http://<IP>:8080/api/docs`)
- **Salud (`/health`):** `http://<IP>:8000/health`
- **Métricas Prometheus:** `http://<IP>:8000/metrics`
- **RabbitMQ Management:** `http://<IP>:15672` (`guest` / `guest`)
- **Grafana:** `http://<IP>:3000` (`admin` / `admin`)

## Pruebas Unitarias, Integración y Carga (IE10)
```bash
# Ejecutar los 12 casos de prueba dentro del contenedor API:
docker exec -it nextgen_api pytest -v

# Ejecutar prueba de carga Locust (50 usuarios concurrentes, 1 minuto, exportando CSV):
docker exec -it nextgen_api locust -f locustfile.py --headless -u 50 -r 5 -t 60s --host http://127.0.0.1:8000 --csv=locust_report
```
