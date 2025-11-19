# main.py
from pipeline.utils import log
from pipeline.carga_datos import ejecutar_carga
from pipeline.limpieza_inicial import limpiar_datos
from pipeline.feature_eng import generar_features
from pipeline.limpieza_features import limpiar_features
from pipeline.preproceso import preprocess_master_table
from pipeline.metricas import calcular_metricas_satisfaccion

def main():
    log("=== INICIO PIPELINE ===")
    dfs, rows_map = ejecutar_carga()
    if dfs is None:
        log("Carga fallida, se aborta pipeline")
        return

    dfs_clean = limpiar_datos(dfs)
    df_features = generar_features(dfs_clean)
    features_raw = generar_features(dfs_clean)
    features_clean = limpiar_features(features_raw)
 
    log("Creando master table...")
    master_table = features_clean.copy()  # aquí podrías agregar joins adicionales si necesitas
    master_table.to_csv("output/master_table.csv", index=False)
    log(f"Master table creada con {master_table.shape[0]} filas y {master_table.shape[1]} columnas")
    
    
    
    X, y, preprocessor, df_master = preprocess_master_table(
    csv_path="./output/master_table.csv",
    target_col="cust_review_mean"
)
    
    
    print(X.head(), y.head())
    
    # master_table viene del feature engineering + limpieza final
    
    metricas = calcular_metricas_satisfaccion(df_master)
    print(metricas)    
    
    print(features_clean.head())
    log("=== PIPELINE FINALIZADO ===")

if __name__ == "__main__":
    main()
