# pipeline/limpieza_inicial.py
import pandas as pd
from pipeline.utils import log

def limpiar_datos(dfs: dict) -> dict:
    """
    Limpieza básica de todas las tablas Olist
    """
    dfs_clean = {}
    for name, df in dfs.items():
        df = df.copy()
        df = df.drop_duplicates()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
        # fechas
        for col in df.columns:
            if "date" in col or "timestamp" in col:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        # nulos
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("unknown")
        dfs_clean[name] = df
        log(f"Limpieza completada para {name}")
    return dfs_clean
