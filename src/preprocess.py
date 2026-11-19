"""Módulo de carga, unificación y preprocesamiento de datasets de NextGen Games."""
from pathlib import Path
import numpy as np
import pandas as pd

MAPEO_COLUMNAS = {
    "Game": "Name",
    "Year": "Year_of_Release",
    "North America": "NA_Sales",
    "Europe": "EU_Sales",
    "Japan": "JP_Sales",
    "Rest of World": "Other_Sales",
    "Global": "Global_Sales",
}

NUMERIC_FEATURES = [
    "Year_of_Release",
    "Critic_Score",
    "Critic_Count",
    "User_Score",
    "User_Count",
]

CATEGORICAL_FEATURES = [
    "Platform",
    "Genre",
    "Publisher",
    "Rating",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_unified_dataset(data_dir: str | Path = "data") -> pd.DataFrame:
    """Unifica los 3 CSV históricos (General, PS4 y Xbox One) replicando EP1."""
    data_path = Path(data_dir)
    df_general = pd.read_csv(data_path / "Video_Games_Sales_as_at_22_Dec_2016.csv")
    df_ps4 = pd.read_csv(data_path / "PS4_GamesSales.csv", encoding="latin-1")
    df_xbox = pd.read_csv(data_path / "XboxOne_GameSales.csv", encoding="latin-1")

    df_ps4 = df_ps4.rename(columns=MAPEO_COLUMNAS)
    df_ps4["Platform"] = "PS4"

    df_xbox = df_xbox.rename(columns=MAPEO_COLUMNAS)
    if "Pos" in df_xbox.columns:
        df_xbox = df_xbox.drop(columns=["Pos"])
    df_xbox["Platform"] = "XOne"

    df_total = pd.concat([df_general, df_ps4, df_xbox], ignore_index=True)
    df = df_total.drop_duplicates(subset=["Name", "Platform", "Year_of_Release"]).copy()

    # Normalizar tipos numéricos (maneja 'tbd' o 'N/A' si existen)
    for col in NUMERIC_FEATURES + ["Global_Sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Limpieza de categóricas
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col] == "", col] = "Unknown"

    return df


def assign_commercial_tier(sales: float) -> str:
    """Clasifica las ventas globales (en millones) en 3 niveles comerciales."""
    if sales < 0.15:
        return "Bajo"
    if sales < 0.60:
        return "Medio"
    return "Alto"


def reduce_cardinality(series: pd.Series, top_n: int = 35) -> tuple[pd.Series, list[str]]:
    """Agrupa publishers poco frecuentes en 'Other' para evitar alta dimensionalidad."""
    top_values = series.value_counts().head(top_n).index.tolist()
    cleaned = series.where(series.isin(top_values), "Other")
    return cleaned, top_values
