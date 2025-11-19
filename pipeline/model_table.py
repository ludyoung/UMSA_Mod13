import pandas as pd
from pipeline.utils import log
from pipeline.feature_eng import generar_features

def crear_master_table(dfs: dict, save_path: str = None) -> pd.DataFrame:
    """
    Crea la master table lista para modelamiento.
    Retorna el DataFrame de la master table.
    
    Parámetros:
    - dfs: dict con DataFrames limpios
    - save_path: ruta opcional para guardar la master table en CSV
    """

    log("=== INICIO Creación Master Table ===")

    # Generar features por cliente y master table
    customer_features, master_table = generar_features(dfs)
    
    log(f"Master table generada con shape: {master_table.shape}")

    # Guardar a CSV si se proporciona ruta
    if save_path:
        master_table.to_csv(save_path, index=False)
        log(f"Master table guardada en: {save_path}")

    log("=== FIN Creación Master Table ===")
    return master_table