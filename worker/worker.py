"""Worker asíncrono Consumer de RabbitMQ para procesamiento por lotes de NextGen Games."""
import json
import os
import time

import pika
import redis

from app.main import GamePredictionRequest, load_model_singleton, run_single_inference

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
QUEUE_NAME = "inference_queue"


def main():
    print("[WORKER] Cargando modelo .pkl en memoria...")
    load_model_singleton()
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    while True:
        try:
            creds = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            params = pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=creds)
            conn = pika.BlockingConnection(params)
            channel = conn.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_qos(prefetch_count=1)

            def callback(ch, method, properties, body):
                payload = json.loads(body.decode("utf-8"))
                job_id = payload["job_id"]
                games = payload["games"]
                print(f"[WORKER] Procesando lote {job_id} ({len(games)} juegos)...")

                results = []
                for g in games:
                    req = GamePredictionRequest(**g)
                    results.append(run_single_inference(req))

                record = {
                    "job_id": job_id,
                    "status": "COMPLETED",
                    "total_items": len(results),
                    "results": results,
                }
                r.setex(f"job:{job_id}", 3600, json.dumps(record))
                r.publish("job_notifications", json.dumps({"job_id": job_id, "status": "COMPLETED"}))
                ch.basic_ack(delivery_tag=method.delivery_tag)
                print(f"[WORKER] Lote {job_id} completado y ACK enviado.")

            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
            print("[WORKER] Esperando mensajes en inference_queue...")
            channel.start_consuming()
        except Exception as exc:
            print(f"[WORKER] Esperando conexión a RabbitMQ ({exc}). Reintentando en 5s...")
            time.sleep(5)


if __name__ == "__main__":
    main()
