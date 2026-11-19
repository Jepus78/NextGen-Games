"""Dashboard Interactivo Comercial — NextGen Games (EP3)."""
import os
import time
import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "nextgen-ep3-secret-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(
    page_title="NextGen Games — Predictor de Ventas IA",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 NextGen Games — Sistema Predictor y Prescriptor de Ventas IA")
st.caption("Arquitectura MLOps en Google Cloud Platform • FastAPI + Scikit-learn (KNNImputer) + RabbitMQ + Redis + Gemini AIaaS")


@st.cache_data(ttl=60)
def fetch_model_info():
    try:
        r = httpx.get(f"{API_BASE_URL}/model/info", timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


info = fetch_model_info()
platforms = info["valid_platforms"] if info else ["PS4", "XOne", "PC", "Wii", "PS3", "X360", "3DS"]
genres = info["valid_genres"] if info else ["Action", "Shooter", "Sports", "Role-Playing", "Racing", "Adventure"]
publishers = info["top_publishers"] if info else ["Ubisoft", "Electronic Arts", "Activision", "Nintendo", "Sony Computer Entertainment"]
ratings = info["valid_ratings"] if info else ["E", "E10+", "T", "M", "Unknown"]

tab1, tab2, tab3 = st.tabs([
    "🔮 Predicción Individual & AIaaS",
    "📦 Evaluación de Catálogo (Batch RabbitMQ)",
    "📊 Métricas del Modelo & Salud",
])

with tab1:
    col_form, col_res = st.columns([1, 1.3], gap="large")
    with col_form:
        st.subheader("Parámetros del Título Candidato")
        name = st.text_input("Nombre del Videojuego", value="Cyber Odyssey 2026")
        c1, c2 = st.columns(2)
        platform = c1.selectbox("Plataforma", platforms, index=platforms.index("PS4") if "PS4" in platforms else 0)
        genre = c2.selectbox("Género", genres, index=genres.index("Action") if "Action" in genres else 0)
        c3, c4 = st.columns(2)
        publisher = c3.selectbox("Publisher", publishers + ["Other"])
        rating = c4.selectbox("Clasificación ESRB", ratings, index=ratings.index("M") if "M" in ratings else 0)
        year = st.slider("Año de Lanzamiento", 1995, 2026, 2018)

        use_knn_scores = st.checkbox("Simular datos faltantes (aplicar imputación KNNImputer)", value=False)
        if not use_knn_scores:
            c5, c6 = st.columns(2)
            critic_score = c5.slider("Critic Score (0-100)", 0.0, 100.0, 82.0)
            critic_count = c5.number_input("Critic Count", 0, 200, 45)
            user_score = c6.slider("User Score (0-10)", 0.0, 10.0, 8.1)
            user_count = c6.number_input("User Count", 0, 10000, 420)
        else:
            critic_score = critic_count = user_score = user_count = None

        include_ai_brief = st.checkbox("Generar Brief Ejecutivo con Gemini API (AIaaS)", value=True)
        run_btn = st.button("🚀 Predecir y Prescribir", type="primary", use_container_width=True)

    with col_res:
        st.subheader("Resultados e Interpretabilidad Comercial")
        if run_btn:
            payload = {
                "name": name,
                "platform": platform,
                "year_of_release": float(year),
                "genre": genre,
                "publisher": publisher,
                "rating": rating,
                "critic_score": critic_score,
                "critic_count": float(critic_count) if critic_count is not None else None,
                "user_score": user_score,
                "user_count": float(user_count) if user_count is not None else None,
                "include_ai_brief": include_ai_brief,
            }
            try:
                r = httpx.post(f"{API_BASE_URL}/predict", json=payload, headers=HEADERS, timeout=15.0)
                r.raise_for_status()
                res = r.json()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ventas Globales", f"{res['predicted_global_sales_millions']:.2f} M")
                m2.metric("Tier Comercial", res["commercial_tier"])
                m3.metric("Presupuesto Mkt", f"USD {res['prescription']['suggested_marketing_budget_kusd']:,.0f}K")
                m4.metric("Latencia API", f"{res['latency_ms']:.1f} ms", delta=f"Cache: {res['cache_status']}")

                st.success(f"**Decisión Prescriptiva:** {res['prescription']['license_decision']}")
                for act in res["prescription"]["recommended_actions"]:
                    st.write(f"- {act}")

                if res.get("imputed_with_knn"):
                    st.info(f"Campos imputados automáticamente con KNNImputer: {', '.join(res['imputed_with_knn'])}")

                if "aiaas_executive_brief" in res:
                    ai = res["aiaas_executive_brief"]
                    st.markdown(f"#### 🤖 Brief Ejecutivo AIaaS (`{ai['provider']} — {ai['model']}`)")
                    st.write(ai["brief"])

                # Gráfico comparativo de interpretabilidad
                bench_df = pd.DataFrame(
                    {
                        "Referencia": ["Título Evaluado", f"Promedio Género ({genre})", f"Promedio Plataforma ({platform})"],
                        "Ventas (Millones)": [
                            res["predicted_global_sales_millions"],
                            res["interpretability"]["genre_historical_avg_millions"],
                            res["interpretability"]["platform_historical_avg_millions"],
                        ],
                    }
                )
                fig = px.bar(bench_df, x="Referencia", y="Ventas (Millones)", color="Referencia", text_auto=".2f")
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Error al consultar la API: {exc}")
        else:
            st.info("Completa los parámetros a la izquierda y presiona **Predecir y Prescribir**.")

with tab2:
    st.subheader("Procesamiento Asíncrono por Lotes (RabbitMQ + Worker + Redis)")
    sample_games = pd.DataFrame([
        {"name": "FIFA Next 2026", "platform": "PS4", "year_of_release": 2018, "genre": "Sports", "publisher": "Electronic Arts", "rating": "E", "critic_score": 84.0, "user_score": 7.4},
        {"name": "Space Raiders X", "platform": "XOne", "year_of_release": 2018, "genre": "Shooter", "publisher": "Activision", "rating": "M", "critic_score": 79.0, "user_score": 7.8},
        {"name": "Indie Quest", "platform": "PC", "year_of_release": 2017, "genre": "Adventure", "publisher": "Other", "rating": "T", "critic_score": 65.0, "user_score": 6.9},
    ])
    edited_df = st.data_editor(sample_games, num_rows="dynamic", use_container_width=True)
    if st.button("📨 Enviar Lote a Cola RabbitMQ (POST /batch)", type="primary"):
        records = edited_df.to_dict(orient="records")
        try:
            r = httpx.post(f"{API_BASE_URL}/batch", json={"games": records}, headers=HEADERS, timeout=10.0)
            r.raise_for_status()
            job = r.json()
            st.info(f"Lote encolado: `job_id={job['job_id']}` | Modo: `{job['mode']}`")
            # Polling hasta completar
            for _ in range(15):
                jr = httpx.get(f"{API_BASE_URL}/job/{job['job_id']}", headers=HEADERS, timeout=5.0)
                jdata = jr.json()
                if jdata.get("status") == "COMPLETED":
                    rows = [
                        {
                            "Título": item["game_name"],
                            "Plataforma": item["platform"],
                            "Ventas Est. (M)": item["predicted_global_sales_millions"],
                            "Tier": item["commercial_tier"],
                            "Decisión": item["prescription"]["license_decision"],
                            "Mkt Sugerido (K USD)": item["prescription"]["suggested_marketing_budget_kusd"],
                        }
                        for item in jdata["results"]
                    ]
                    st.success("Lote procesado correctamente por el Worker.")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    break
                time.sleep(0.5)
        except Exception as exc:
            st.error(f"Error procesando lote: {exc}")

with tab3:
    st.subheader("Salud del Sistema y Métricas Reales del Modelo")
    try:
        h = httpx.get(f"{API_BASE_URL}/health", timeout=5.0).json()
        st.json(h)
    except Exception as exc:
        st.error(f"API no disponible: {exc}")

    if info:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Métricas de Regresión (`Global_Sales`)")
            st.json(info["regression_metrics"])
        with c2:
            st.markdown("#### Métricas de Clasificación (`Commercial_Tier`)")
            st.json(info["classification_metrics"])
        st.markdown("#### Hiperparámetros y Dataset")
        st.json({
            "dataset_total_rows": info["dataset_total_rows"],
            "dataset_model_rows": info["dataset_model_rows"],
            "train_rows": info["train_rows"],
            "test_rows": info["test_rows"],
            "hyperparameters": info["hyperparameters"],
        })
