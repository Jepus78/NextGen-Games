"""Motor de Análisis Prescriptivo para la toma de decisiones en NextGen Games."""
from typing import Dict, Any


def build_prescription(
    predicted_sales_m: float,
    commercial_tier: str,
    critic_score: float | None,
    user_score: float | None,
    platform: str,
    genre: str,
    is_anomaly: bool,
) -> Dict[str, Any]:
    """Genera recomendaciones accionables de licencia, marketing y mejora."""
    # Estimación de presupuesto sugerido de marketing proporcional al retorno esperado
    # Supuesto comercial NextGen Games: ingreso neto estimado USD 25/copia, marketing ~12%
    estimated_revenue_musd = round(predicted_sales_m * 25.0, 2)
    suggested_marketing_kusd = round(max(15.0, min(estimated_revenue_musd * 1000 * 0.12, 2500.0)), 1)

    actions = []
    if commercial_tier == "Alto":
        decision = "ADQUIRIR LICENCIA PRIORITARIA (Tier A)"
        actions.append(
            f"Aprobar adquisición prioritaria para {platform} ({genre}) con presupuesto de marketing sugerido de USD {suggested_marketing_kusd:,.1f}K."
        )
        actions.append("Planificar campaña multicanal en Latinoamérica y Europa 60 días antes del lanzamiento.")
    elif commercial_tier == "Medio":
        decision = "ADQUIRIR CON PRESUPUESTO CONTROLADO (Tier B)"
        actions.append(
            f"Negociar licencia con costo acotado y campaña digital segmentada (USD {suggested_marketing_kusd:,.1f}K)."
        )
        actions.append("Priorizar preventa digital para reducir riesgo de sobrestock físico.")
    else:
        decision = "REVISAR / NO PRIORIZAR LICENCIA (Tier C)"
        actions.append(
            f"Proyección conservadora ({predicted_sales_m:.2f}M copias). Evitar compra de stock físico masivo y limitar marketing a USD {suggested_marketing_kusd:,.1f}K."
        )
        actions.append("Evaluar distribución únicamente digital o negociación en paquete (bundle).")

    if critic_score is not None and critic_score < 70:
        actions.append(
            f"Alerta de calidad: Critic_Score proyectado ({critic_score:.1f}/100) bajo el umbral competitivo (75). Recomendar parche pre-lanzamiento o ajuste de precio."
        )
    if user_score is not None and user_score < 6.5:
        actions.append(
            f"Riesgo de recepción: User_Score ({user_score:.1f}/10) sugiere reforzar pruebas beta y comunicación con la comunidad."
        )
    if is_anomaly:
        actions.append(
            "Anomalía detectada (IsolationForest): la combinación de atributos se desvía del histórico. Validar manualmente antes de aprobar presupuesto."
        )

    return {
        "license_decision": decision,
        "estimated_gross_revenue_musd": estimated_revenue_musd,
        "suggested_marketing_budget_kusd": suggested_marketing_kusd,
        "recommended_actions": actions,
    }
