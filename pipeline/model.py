from pipeline.utils import log, update_stage

def entrenar_modelo(dfs: dict):
    update_stage("Modelo", "running", "Entrenando modelo (placeholder)")
    try:
        # Placeholder: no hay entrenamiento real aquí
        log("Entrenamiento simulado completado")
        update_stage("Modelo", "success", "Modelo simulado entrenado")
        return True
    except Exception as e:
        log(f"ERROR Modelo: {e}")
        update_stage("Modelo", "error", str(e))
        raise
