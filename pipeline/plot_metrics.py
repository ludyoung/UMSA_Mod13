# pipeline/plot_metrics.py

import os
import matplotlib.pyplot as plt
from pipeline.utils import log

def plot_metricas_satisfaccion(metrics: dict, output_path: str = "./Images/RESULTADOS/metricas_barras.png"):
    """
    Genera un gráfico de barras con las métricas clave de satisfacción
    y guarda la imagen en ./Images/RESULTADOS/
    """

    log("=== INICIO Gráfico de métricas de satisfacción ===")

    if not isinstance(metrics, dict):
        raise TypeError("metrics debe ser un dict generado por calcular_metricas_satisfaccion()")

    # ------------------------------------------------------
    # Selección de métricas relevantes
    # ------------------------------------------------------
    labels = [
        "CSAT",
        "NPS",
        "% Reviews Positivas",
        "% Reviews Neutrales",
        "% Reviews Negativas"
    ]

    # Normalizar NPS de 0–100 → 0–1 para que todas las barras estén en escala comparable
    nps_norm = metrics.get("nps", 0) / 100

    values = [
        metrics.get("csat", 0),
        nps_norm,
        metrics.get("pct_reviews_positivas", 0),
        metrics.get("pct_reviews_neutral", 0),
        metrics.get("pct_reviews_negativas", 0),
    ]

    # Reemplazar valores None o NaN por 0
    values = [0 if (v is None) else v for v in values]

    # ------------------------------------------------------
    # Crear carpeta si no existe
    # ------------------------------------------------------
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------
    # Crear gráfico de barras
    # ------------------------------------------------------
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)

    plt.title("Indicadores de Satisfacción del Cliente", fontsize=14)
    plt.ylabel("Valor (Escala Normalizada)", fontsize=12)
    plt.xticks(rotation=15)

    # Etiquetas arriba de cada barra
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.2f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()

    # Guardar imagen
    plt.savefig(output_path, dpi=300)
    plt.close()

    log(f"Imagen guardada en {output_path}")
    log("=== FIN Gráfico de métricas de satisfacción ===")
