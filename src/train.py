"""Pipeline de entrenamiento offline: KNNImputer + Regresor + Clasificador + IsolationForest."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
sklearn.set_config(working_memory=64)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.preprocess import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    assign_commercial_tier,
    load_unified_dataset,
    reduce_cardinality,
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model_bundle.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def upload_to_gcs_if_configured(local_path: Path, blob_name: str) -> str | None:
    bucket_name = os.getenv("GCS_BUCKET_NAME", "").strip()
    if not bucket_name:
        return None
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        uri = f"gs://{bucket_name}/{blob_name}"
        print(f"[GCS] Artefacto subido correctamente a {uri}")
        return uri
    except Exception as exc:
        print(f"[GCS] Aviso: no se pudo subir a GCS ({exc}). Se mantiene copia local.")
        return None


def download_from_gcs_if_configured(local_path: Path, blob_name: str) -> bool:
    bucket_name = os.getenv("GCS_BUCKET_NAME", "").strip()
    if not bucket_name or local_path.exists():
        return False
    try:
        from google.cloud import storage

        local_path.parent.mkdir(parents=True, exist_ok=True)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if blob.exists():
            blob.download_to_filename(str(local_path))
            print(f"[GCS] Artefacto descargado desde gs://{bucket_name}/{blob_name}")
            return True
    except Exception as exc:
        print(f"[GCS] Aviso: no se pudo descargar desde GCS ({exc}).")
    return False


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn_imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def train_and_save(data_dir: str | Path = "data") -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_unified_dataset(data_dir)

    # Filtrar registros sin ventas válidas y tratar outliers extremos (> P99)
    df = df.dropna(subset=["Global_Sales"]).copy()
    p99 = float(df["Global_Sales"].quantile(0.99))
    df_model = df[df["Global_Sales"] <= p99].copy()

    df_model["Publisher"], top_publishers = reduce_cardinality(df_model["Publisher"], top_n=40)
    df_model["Commercial_Tier"] = df_model["Global_Sales"].apply(assign_commercial_tier)

    X = df_model[ALL_FEATURES].copy()
    y_reg = np.log1p(df_model["Global_Sales"].values)
    y_clf = df_model["Commercial_Tier"].values

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    # 1. Pipeline Regresor (Ventas Globales en millones, escala log1p)
    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.06,
                    max_depth=8,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
    reg_pipeline.fit(X_train, y_reg_train)
    y_reg_pred_log = reg_pipeline.predict(X_test)
    y_reg_pred = np.expm1(np.clip(y_reg_pred_log, 0, None))
    y_reg_true = np.expm1(y_reg_test)

    reg_metrics = {
        "rmse_millions": round(float(root_mean_squared_error(y_reg_true, y_reg_pred)), 4),
        "mae_millions": round(float(mean_absolute_error(y_reg_true, y_reg_pred)), 4),
        "r2_score": round(float(r2_score(y_reg_true, y_reg_pred)), 4),
        "r2_log_scale": round(float(r2_score(y_reg_test, y_reg_pred_log)), 4),
    }

    # 2. Pipeline Clasificador (Tier Comercial: Bajo / Medio / Alto)
    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.06,
                    max_depth=8,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
    clf_pipeline.fit(X_train, y_clf_train)
    y_clf_pred = clf_pipeline.predict(X_test)

    clf_metrics = {
        "accuracy": round(float(accuracy_score(y_clf_test, y_clf_pred)), 4),
        "precision_weighted": round(float(precision_score(y_clf_test, y_clf_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_clf_test, y_clf_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_clf_test, y_clf_pred, average="weighted", zero_division=0)), 4),
        "classes": list(clf_pipeline.named_steps["model"].classes_),
    }

    # 3. Detector de Anomalías (IsolationForest — Sesión 3.4.3)
    X_train_transformed = reg_pipeline.named_steps["preprocessor"].transform(X_train)
    anomaly_detector = IsolationForest(
        n_estimators=100,
        contamination=0.03,
        random_state=42,
        n_jobs=1,
    )
    anomaly_detector.fit(X_train_transformed)

    # Estadísticas por género para interpretabilidad en el Dashboard
    genre_benchmarks = (
        df_model.groupby("Genre")["Global_Sales"].mean().round(3).to_dict()
    )
    platform_benchmarks = (
        df_model.groupby("Platform")["Global_Sales"].mean().round(3).to_dict()
    )

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_total_rows": int(len(df)),
        "dataset_model_rows": int(len(df_model)),
        "outlier_cap_p99_millions": round(p99, 3),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": ALL_FEATURES,
        "hyperparameters": {
            "imputation": "KNNImputer(n_neighbors=5, weights='distance')",
            "regressor": "HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=8, l2_regularization=1.0)",
            "classifier": "HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=8, l2_regularization=1.0)",
            "anomaly_detector": "IsolationForest(n_estimators=100, contamination=0.03)",
        },
        "regression_metrics": reg_metrics,
        "classification_metrics": clf_metrics,
        "top_publishers": top_publishers,
        "valid_platforms": sorted(df_model["Platform"].unique().tolist()),
        "valid_genres": sorted(df_model["Genre"].unique().tolist()),
        "valid_ratings": sorted(df_model["Rating"].unique().tolist()),
        "genre_benchmarks": genre_benchmarks,
        "platform_benchmarks": platform_benchmarks,
    }

    bundle = {
        "regressor": reg_pipeline,
        "classifier": clf_pipeline,
        "anomaly_detector": anomaly_detector,
        "metadata": metadata,
    }

    joblib.dump(bundle, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    gcs_uri = upload_to_gcs_if_configured(MODEL_PATH, "models/model_bundle.pkl")
    if gcs_uri:
        metadata["gcs_model_uri"] = gcs_uri
        METRICS_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TRAIN] Modelo guardado en {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"[TRAIN] Regresión: {reg_metrics}")
    print(f"[TRAIN] Clasificación: {clf_metrics}")
    return metadata


if __name__ == "__main__":
    train_and_save()