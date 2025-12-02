# pipeline/metrics_satisfaction.py
import pandas as pd
import time
from pipeline.utils import log


def calcular_metricas_satisfaccion(df_master: pd.DataFrame) -> dict:
    """
    Calcula métricas clave de satisfacción del cliente con validaciones,
    logging detallado y checkpoints de ejecución.
    """

    log("=== INICIO Cálculo de métricas de satisfacción ===")

    # ---------------------------------
    # VALIDACIONES INICIALES
    # ---------------------------------
    try:
        if not isinstance(df_master, pd.DataFrame):
            log("ERROR: Input no es un DataFrame")
            raise TypeError("El input para calcular_metricas_satisfaccion debe ser un DataFrame")

        if df_master.empty:
            log("ERROR: DataFrame vacío")
            raise ValueError("El DataFrame está vacío, no se pueden calcular métricas")

        required_cols = ["cust_review_mean"]
        for col in required_cols:
            if col not in df_master.columns:
                log(f"ERROR: Falta columna requerida: {col}")
                raise ValueError(f"El DataFrame no contiene la columna requerida: {col}")

        log("Validaciones iniciales: OK")

    except Exception as e:
        log(f"EXCEPCIÓN durante validaciones: {str(e)}")
        raise e

    # CHECKPOINT 1
    log("Checkpoint 1: Validaciones completas, iniciando cálculos...")

    # Simulación de latencia para verificar seguimiento en logs
    time.sleep(0.3)

    # ---------------------------------
    # MÉTRICAS GLOBALES
    # ---------------------------------
    try:
        total_clientes = len(df_master)
        promedio_satisfaccion = df_master["cust_review_mean"].mean()
        std_satisfaccion = df_master["cust_review_mean"].std()

        log(f"Cálculo global: total_clientes={total_clientes}, "
            f"mean={promedio_satisfaccion:.4f}, std={std_satisfaccion:.4f}")

    except Exception as e:
        log(f"ERROR en cálculos globales: {str(e)}")
        raise e

    # CHECKPOINT 2
    log("Checkpoint 2: Cálculos globales completados.")

    time.sleep(0.3)

    # ---------------------------------
    # CLASIFICACIÓN DE CLIENTES
    # ---------------------------------
    try:
        df_master["satisfied"] = (df_master["cust_review_mean"] >= 4).astype(int)
        df_master["detractor"] = (df_master["cust_review_mean"] <= 2).astype(int)
        df_master["neutral"] = (
            (df_master["cust_review_mean"] > 2) &
            (df_master["cust_review_mean"] < 4)
        ).astype(int)

        log("Clasificación de clientes aplicada correctamente.")

    except Exception as e:
        log(f"ERROR en clasificación de clientes: {str(e)}")
        raise e

    # CHECKPOINT 3
    log("Checkpoint 3: Clasificación completada.")

    time.sleep(0.3)

    # ---------------------------------
    # CSAT y NPS
    # ---------------------------------
    try:
        csat = df_master["satisfied"].mean()
        promoters = df_master["satisfied"].mean()
        detractors = df_master["detractor"].mean()
        nps = (promoters - detractors) * 100

        log(f"CSAT={csat:.4f}, NPS={nps:.2f}")

    except Exception as e:
        log(f"ERROR en el cálculo de CSAT/NPS: {str(e)}")
        raise e

    # CHECKPOINT 4
    log("Checkpoint 4: Métricas CSAT & NPS completadas.")

    time.sleep(0.3)

    # ---------------------------------
    # CORRELACIONES
    # ---------------------------------
    def safe_corr(col1, col2):
        if col2 not in df_master.columns:
            log(f"Aviso: No existe columna {col2} para correlación con {col1}")
            return None
        try:
            return df_master[col1].corr(df_master[col2])
        except Exception:
            log(f"Error al calcular correlación entre {col1} y {col2}")
            return None

    correlacion_retraso = safe_corr("cust_review_mean", "cust_late_ratio")
    correlacion_precio = safe_corr("cust_review_mean", "cust_avg_price")
    correlacion_frecuencia = safe_corr("cust_review_mean", "cust_total_orders")

    log("Cálculo de correlaciones completado.")

    # CHECKPOINT 5 — última etapa antes de retorno
    log("Checkpoint 5: Todas las métricas calculadas correctamente.")

    time.sleep(0.3)

    # ---------------------------------
    # FIN
    # ---------------------------------
    log("=== FIN Cálculo de métricas de satisfacción ===")

    return {
        "total_clientes": total_clientes,
        "promedio_satisfaccion": promedio_satisfaccion,
        "std_satisfaccion": std_satisfaccion,

        "csat": csat,
        "nps": nps,

        "pct_reviews_negativas": detractors,
        "pct_reviews_neutral": df_master["neutral"].mean(),
        "pct_reviews_positivas": promoters,

        "correlacion_retraso_review": correlacion_retraso,
        "correlacion_precio_review": correlacion_precio,
        "correlacion_frecuencia_review": correlacion_frecuencia,
    }
