import subprocess
import sys
from pipeline.utils import log, read_state, write_state, update_stage
from pipeline.carga_datos import ejecutar_carga
from pipeline.eda import ejecutar_eda
from pipeline.transform import ejecutar_transform
from pipeline.model import entrenar_modelo
from datetime import datetime
import json, os
def main():
    log("=== INICIO PIPELINE ===")
    state = read_state()
    # reset statuses to running flow begins
    state["status"] = {k: "idle" for k in state["status"].keys()}
    write_state(state)
    try:
        # 1. Carga
        dfs, rows_map = ejecutar_carga()
        if dfs is None:
            log("Carga fallida - se aborta pipeline")
            return
        # update files_rows in state
        s = read_state()
        s["files_rows"] = {k: int(v) for k,v in rows_map.items()}
        s["last_run"] = datetime.utcnow().isoformat() + "Z"
        write_state(s)
        # 2. EDA
        eda_summary = ejecutar_eda(dfs)
        # 3. Transform
        dfs2 = ejecutar_transform(dfs)
        # 4. Model
        entrenar_modelo(dfs2)
        log("=== PIPELINE FINALIZADO ===")
    except Exception as e:
        log(f"EXCEPCION EN PIPELINE: {e}")
        # mark any running stage as error
        st = read_state()
        for k,v in st["status"].items():
            if v == "running":
                st["status"][k] = "error"
        st["last_message"] = f"Error: {str(e)}"
        write_state(st)
        raise
if __name__ == "__main__":
    main()
