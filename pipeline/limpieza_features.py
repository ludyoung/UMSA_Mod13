# pipeline/limpieza_features.py
import pandas as pd
from pipeline.utils import log

def limpiar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza de DataFrame de features agregados por customer_id
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("limpiar_features solo acepta DataFrame")

    log("=== INICIO Limpieza de features ===")
    df = df.copy()

    # Nulos
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log(f"{col}: nulos rellenados con mediana={median_val}")
        else:
            df[col] = df[col].fillna("unknown")
            log(f"{col}: nulos rellenados con 'unknown'")

    log("=== FIN Limpieza de features ===")
    return df
