# pipeline/metrics_satisfaction.py
import pandas as pd
from pipeline.utils import log


def calcular_metricas_satisfaccion(df_master: pd.DataFrame) -> dict:
    """
    Calcula métricas clave para análisis de satisfacción del cliente.
    Recibe directamente la master_table ya preprocesada.
    """

    log("=== INICIO Cálculo de métricas de satisfacción ===")

    if not isinstance(df_master, pd.DataFrame):
        raise TypeError("El input para calcular_metricas_satisfaccion debe ser un DataFrame")

    # Validación mínima
    required_cols = ["cust_review_mean"]
    for col in required_cols:
        if col not in df_master.columns:
            raise ValueError(f"El DataFrame no contiene la columna requerida: {col}")

    # === MÉTRICAS GLOBALES ===
    total_clientes = len(df_master)
    promedio_satisfaccion = df_master["cust_review_mean"].mean()
    std_satisfaccion = df_master["cust_review_mean"].std()

    # Clasificación por tipo de cliente según satisfacción
    df_master["satisfied"] = (df_master["cust_review_mean"] >= 4).astype(int)
    df_master["detractor"] = (df_master["cust_review_mean"] <= 2).astype(int)
    df_master["neutral"] = (
        (df_master["cust_review_mean"] > 2) &
        (df_master["cust_review_mean"] < 4)
    ).astype(int)

    # === CSAT ===
    csat = df_master["satisfied"].mean()

    # === NPS ===
    promoters = df_master["satisfied"].mean()
    detractors = df_master["detractor"].mean()
    nps = (promoters - detractors) * 100

    # === Porcentaje por tipo de reseña ===
    pct_bad = detractors
    pct_neutral = df_master["neutral"].mean()
    pct_good = promoters

    # === Correlación retrasos – satisfacción ===
    if "cust_late_ratio" in df_master.columns:
        correlacion_retraso = df_master["cust_review_mean"].corr(df_master["cust_late_ratio"])
    else:
        correlacion_retraso = None

    # === Correlación precio – satisfacción ===
    if "cust_avg_price" in df_master.columns:
        correlacion_precio = df_master["cust_review_mean"].corr(df_master["cust_avg_price"])
    else:
        correlacion_precio = None

    # === Correlación frecuencia – satisfacción ===
    if "cust_total_orders" in df_master.columns:
        correlacion_frecuencia = df_master["cust_review_mean"].corr(df_master["cust_total_orders"])
    else:
        correlacion_frecuencia = None

    log("=== FIN Cálculo de métricas de satisfacción ===")

    return {
        "total_clientes": total_clientes,
        "promedio_satisfaccion": promedio_satisfaccion,
        "std_satisfaccion": std_satisfaccion,

        "csat": csat,
        "nps": nps,

        "pct_reviews_negativas": pct_bad,
        "pct_reviews_neutral": pct_neutral,
        "pct_reviews_positivas": pct_good,

        "correlacion_retraso_review": correlacion_retraso,
        "correlacion_precio_review": correlacion_precio,
        "correlacion_frecuencia_review": correlacion_frecuencia,
    }
