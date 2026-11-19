FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY worker/ ./worker/
COPY data/ ./data/
COPY tests/ ./tests/
COPY locustfile.py .

RUN mkdir -p /app/artifacts && useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://127.0.0.1:8000/health', timeout=3); assert r.status_code == 200"

CMD ["sh", "-c", "if [ ! -f /app/artifacts/model_bundle.pkl ]; then python -m src.train; fi && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
