from pipeline.utils import log, update_stage

def ejecutar_transform(dfs: dict):
    update_stage("Transformación", "running", "Iniciando transformaciones")
    try:
        # Placeholder: ejemplo simple - normalizar nombres de columnas a minúsculas
        dfs_out = {}
        for name, df in dfs.items():
            df.columns = [c.lower() for c in df.columns]
            dfs_out[name] = df
        update_stage("Transformación", "success", "Transformación completada")
        return dfs_out
    except Exception as e:
        log(f"ERROR Transform: {e}")
        update_stage("Transformación", "error", str(e))
        raise
