"""Integración AIaaS con Google AI Studio (Gemini API gratuita) + Reintento Exponencial y Fallback."""
import asyncio
import os
from typing import Dict, Any

import httpx

GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


async def generate_executive_brief(
    game_name: str,
    platform: str,
    genre: str,
    publisher: str,
    predicted_sales_m: float,
    commercial_tier: str,
    prescription: Dict[str, Any],
) -> Dict[str, Any]:
    """Invoca Gemini API (AIaaS GCP) con backoff exponencial (t = 0.5 * 2^i) y fallback local."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    fallback_text = (
        f"Resumen Ejecutivo (Fallback Local): Para '{game_name}' ({platform} / {genre}, {publisher}), "
        f"el modelo estima {predicted_sales_m:.2f}M copias globales (Tier {commercial_tier}). "
        f"Decisión sugerida: {prescription['license_decision']} con presupuesto de marketing estimado en "
        f"USD {prescription['suggested_marketing_budget_kusd']:,.1f}K."
    )

    if not api_key:
        return {
            "provider": "fallback-local",
            "model": "deterministic-template",
            "status": "fallback_no_api_key",
            "brief": fallback_text,
        }

    prompt = (
        "Eres un analista comercial senior de NextGen Games (distribuidora de videojuegos en LATAM y Europa). "
        "Redacta un brief ejecutivo en español de máximo 80 palabras, claro y accionable, interpretando esta predicción:\n"
        f"- Título: {game_name}\n"
        f"- Plataforma: {platform} | Género: {genre} | Publisher: {publisher}\n"
        f"- Ventas globales estimadas: {predicted_sales_m:.2f} millones de copias\n"
        f"- Nivel comercial: {commercial_tier}\n"
        f"- Decisión prescriptiva: {prescription['license_decision']}\n"
        f"- Presupuesto marketing sugerido: USD {prescription['suggested_marketing_budget_kusd']:,.1f}K\n"
        "Incluye el principal riesgo comercial y una recomendación concreta."
    )

    url = GEMINI_ENDPOINT_TEMPLATE.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 180},
    }

    max_retries = 3
    base_delay = 0.5

    async with httpx.AsyncClient(timeout=8.0) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(url, params={"key": api_key}, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    text = (
                        data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                        .strip()
                    )
                    if text:
                        return {
                            "provider": "Google AI Studio (Gemini API - AIaaS)",
                            "model": model,
                            "status": "success",
                            "brief": text,
                        }
                if resp.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(base_delay * (2**attempt))
                    continue
                break
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2**attempt))
                    continue
                break

    return {
        "provider": "fallback-local",
        "model": "deterministic-template",
        "status": "fallback_after_retries",
        "brief": fallback_text,
    }
