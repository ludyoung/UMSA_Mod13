# pipeline/preproceso.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_master_table(csv_path: str, target_col: str):
    """
    Carga master_table.csv, separa target, crea preprocesador,
    genera gráfico de correlación de features relacionados con la satisfacción
    y devuelve: X, y, preprocessor, df_master
    """

    # 1. Cargar CSV
    df = pd.read_csv(csv_path)

    # 2. Separar variable objetivo
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en master_table.csv")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # ---------------------------
    # 3. Extraer columnas que tengan que ver con satisfacción del cliente
    # ---------------------------

    keywords = [
        "satisfaction", "satisfecho", "insatisfecho",
        "review", "score", "sentiment",
        "espera", "wait", "delay", "delivery"
    ]

    satisfaction_cols = [
        col for col in df.columns
        if any(key in col.lower() for key in keywords)
    ]

    # Asegurar que solo sean numéricas para correlación
    satisfaction_numeric = df[satisfaction_cols].select_dtypes(include=["int64", "float64"])

    # Crear directorio si no existe
    output_dir = Path("./Images/DATAFRAME/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 4. Graficar correlación de variables de satisfacción
    # ---------------------------
    if not satisfaction_numeric.empty:

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            satisfaction_numeric.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5
        )

        plt.title("Correlación entre variables de satisfacción del cliente", fontsize=16)
        plt.tight_layout()

        save_path = output_dir / "correlacion_satisfaccion.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    # ---------------------------
    # 5. Identificar columnas numéricas y categóricas
    # ---------------------------
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 6. Preprocesadores
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    # 7. Ensamblado de transformador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return X, y, preprocessor, df
