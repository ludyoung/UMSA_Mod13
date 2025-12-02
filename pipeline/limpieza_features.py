# pipeline/limpieza_features.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pipeline.utils import log

# -----------------------------
# Ruta donde se guardarán los gráficos
# -----------------------------
IMG_DF_DIR = Path("./Images/DATAFRAME")
IMG_DF_DIR.mkdir(parents=True, exist_ok=True)


def limpiar_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("limpiar_features solo acepta DataFrame")

    log("=== INICIO Limpieza de features ===")
    df = df.copy()

    # -----------------------------------------------------
    # Rellenar valores nulos por mediana o 'unknown'
    # -----------------------------------------------------
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna("unknown")

    log("Valores nulos reemplazados correctamente")

    # -----------------------------------------------------
    # Selección de columnas numéricas para gráficos
    # -----------------------------------------------------
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    if numeric_df.empty:
        log("No hay columnas numéricas para graficar en limpieza_features.")
        log("=== FIN Limpieza de features ===")
        return df

    # -----------------------------------------------------
    # 1. Boxplot general de todas las variables numéricas
    # -----------------------------------------------------
    try:
        plt.figure(figsize=(25, 10))
        numeric_df.boxplot(rot=90)
        plt.title("Distribución de Variables Numéricas Después de la Limpieza")
        plt.tight_layout()

        boxplot_path = IMG_DF_DIR / "boxplot_variables_numericas.png"
        plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        plt.close()

        log(f"Boxplot guardado en: {boxplot_path}")

    except Exception as e:
        log(f"ERROR al crear boxplot general: {e}")

    # -----------------------------------------------------
    # 3. Heatmap de correlación
    # -----------------------------------------------------
    try:
        plt.figure(figsize=(20, 14))
        corr = numeric_df.corr()

        sns.heatmap(
            corr,
            cmap="viridis",
            annot=False,
            square=True,
            cbar=True
        )

        plt.title("Heatmap de Correlación — Variables Numéricas")
        plt.tight_layout()

        heatmap_path = IMG_DF_DIR / "heatmap_correlacion.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()

        log(f"Heatmap de correlación guardado en: {heatmap_path}")

    except Exception as e:
        log(f"ERROR al generar heatmap de correlación: {e}")

    # -----------------------------------------------------
    # FIN
    # -----------------------------------------------------
    log("=== FIN Limpieza de features ===")
    return df
