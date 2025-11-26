# main.py
from pipeline.utils import log
from pipeline.carga_datos import ejecutar_carga
from pipeline.limpieza_inicial import limpiar_datos
from pipeline.feature_eng import generar_features
from pipeline.limpieza_features import limpiar_features
from pipeline.preproceso import preprocess_master_table
from pipeline.metricas import calcular_metricas_satisfaccion
from pipeline.model_training import run_training_pipeline

def main():
    log("======================================= INICIO PIPELINE =================================")
    dfs, rows_map = ejecutar_carga()
    if dfs is None:
        log("Carga fallida, se aborta pipeline")
        return
    
    log("======================================= LIMPIEZA =================================")
    dfs_clean = limpiar_datos(dfs)
    log("======================================= FEATURE ENGINEERING =================================")
    df_features = generar_features(dfs_clean)
    features_raw = generar_features(dfs_clean)
    log("======================================= LIMPIEZA FEATURES =================================")
    features_clean = limpiar_features(features_raw)
 
    log("======================================= MASTER TABLE =================================")
    master_table = features_clean.copy()  # aquí podrías agregar joins adicionales si necesitas
    log("======================================= EXPORTACION DE MASTER TABLE =================================") 
    master_table.to_csv("output/master_table.csv", index=False)
    log(f"Master table creada con {master_table.shape[0]} filas y {master_table.shape[1]} columnas")
    log("======================================= PREPROCESO MASTER TABLE =================================")    
    X, y, preprocessor, df_master = preprocess_master_table(
    csv_path="./output/master_table.csv",
    target_col="cust_review_mean"
)   
    log("======================================= CALCULO DE METRICAS =================================")       
    metricas = calcular_metricas_satisfaccion(df_master)
    log("======================================= ENTRENAMIENTO DEL MODELO =================================")       
    log("Entrenando modelo de predicción de satisfacción...")
    run_training_pipeline()
    
    log("=== PIPELINE FINALIZADO ===")

if __name__ == "__main__":
    main()
