"""Pruebas de carga con Locust para validar SLA < 500 ms y picos de 100 req/min (IE10)."""
import os
import random
from locust import HttpUser, between, task

API_KEY = os.getenv("API_KEY", "nextgen-ep3-secret-key-2026")
PLATFORMS = ["PS4", "XOne", "PC", "Wii", "PS3", "3DS"]
GENRES = ["Action", "Shooter", "Sports", "Role-Playing", "Racing", "Adventure"]
PUBLISHERS = ["Ubisoft", "Electronic Arts", "Activision", "Nintendo", "Square Enix", "Other"]


class NextGenAnalystUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def predict_single(self):
        payload = {
            "name": f"LoadTest Title {random.randint(1, 50)}",
            "platform": random.choice(PLATFORMS),
            "year_of_release": random.randint(2014, 2020),
            "genre": random.choice(GENRES),
            "publisher": random.choice(PUBLISHERS),
            "rating": random.choice(["E", "T", "M"]),
            "critic_score": float(random.randint(55, 95)),
            "critic_count": float(random.randint(10, 80)),
            "user_score": round(random.uniform(5.5, 9.2), 1),
            "user_count": float(random.randint(50, 1500)),
            "include_ai_brief": False,
        }
        self.client.post(
            "/predict",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/predict",
        )

    @task(1)
    def check_health(self):
        self.client.get("/health", name="/health")
