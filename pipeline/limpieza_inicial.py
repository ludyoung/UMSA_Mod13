# pipeline/limpieza_inicial.py
import pandas as pd
from pipeline.utils import log

def limpiar_datos(dfs: dict) -> dict:
    """
    Limpieza básica de todas las tablas Olist con logging detallado.
    """
    dfs_clean = {}
    log("=== INICIO Limpieza de tablas ===")

    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame):
            log(f"[WARN] {name} no es DataFrame, se salta")
            continue

        # log(f"Limpieza iniciada para {name}, shape original: {df.shape}")
        df = df.copy()

        # ==========================
        # 1. Eliminar duplicados
        # ==========================
        df = df.drop_duplicates()
        # log(f"{name}: duplicados eliminados, {df.shape[0]} filas restantes")

        # ==========================
        # 2. Normalización de textos
        # ==========================
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in obj_cols:
            df[col] = df[col].astype(str).str.strip()
        if obj_cols:
           log(f"{name}: columnas de texto normalizadas -> {obj_cols}")

        # ==========================
        # 3. Conversión de fechas
        # ==========================
        fecha_cols = [c for c in df.columns if "date" in c or "timestamp" in c]
        for col in fecha_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        if fecha_cols:
           log(f"{name}: columnas de fecha convertidas -> {fecha_cols}")

        # ==========================
        # 4. Manejo de nulos
        # ==========================
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
            else:
                df[col] = df[col].fillna("unknown")
        # log(f"{name}: nulos tratados para todas las columnas")

        dfs_clean[name] = df
        # log(f"Limpieza completada para {name}, shape final: {df.shape}")

    log("=== FIN Limpieza de tablas ===")
    return dfs_clean
