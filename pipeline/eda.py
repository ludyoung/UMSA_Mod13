import os
from datetime import datetime
from pipeline.utils import log, update_stage

EDA_FOLDER = os.path.join("pipeline", "eda_results")
os.makedirs(EDA_FOLDER, exist_ok=True)

def ejecutar_eda(dfs: dict):
    update_stage("EDA", "running", "Iniciando EDA")
    resumen = {}
    for name, df in dfs.items():
        try:
            filas, cols = df.shape
            nulos = df.isnull().sum().to_dict()
            resumen[name] = {"filas": int(filas), "columnas": int(cols), "nulos": nulos}
            # Guardar texto resumen
            out = os.path.join(EDA_FOLDER, f"eda_resumen_{name}.txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write(f"EDA BASICO {name}\n")
                f.write(f"filas: {filas}\ncolumnas: {cols}\n\n")
                f.write("NULOS POR COLUMNA\n")
                for c, v in nulos.items():
                    f.write(f"{c}: {v}\n")
            log(f"EDA guardado para {name} -> {out}")
        except Exception as e:
            log(f"ERROR EDA {name}: {e}")
            update_stage("EDA", "error", f"Error en {name}: {e}")
            raise
    update_stage("EDA", "success", "EDA completado")
    return resumen
