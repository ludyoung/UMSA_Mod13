# pipeline/preproceso.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def preprocess_master_table(csv_path: str, target_col: str):
    """
    Carga master_table.csv, separa target, crea preprocesador y devuelve:
    X, y, preprocessor, df_master
    """

    # 1. Cargar CSV
    df = pd.read_csv(csv_path)

    # 2. Separar variable objetivo
    if target_col not in df.columns:
        raise ValueError(f"❌ La columna objetivo '{target_col}' no existe en master_table.csv")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. Identificar columnas numéricas y categóricas
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 4. Definir transformaciones
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    # 5. Ensamblar transformador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return X, y, preprocessor, df
