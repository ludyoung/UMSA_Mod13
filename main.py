# main.py
from pipeline.utils import log
from pipeline.carga_datos import ejecutar_carga
from pipeline.limpieza_inicial import limpiar_datos
from pipeline.feature_eng import generar_features

def main():
    log("=== INICIO PIPELINE ===")
    dfs, rows_map = ejecutar_carga()
    if dfs is None:
        log("Carga fallida, se aborta pipeline")
        return

    dfs_clean = limpiar_datos(dfs)
    df_features = generar_features(dfs_clean)

    log("Features generados:")
    print(df_features.head())
    log("=== PIPELINE FINALIZADO ===")

if __name__ == "__main__":
    main()
