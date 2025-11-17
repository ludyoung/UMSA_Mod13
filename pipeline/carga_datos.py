import os
import pandas as pd
import hashlib
from pipeline.utils import log, update_stage, read_hashes, write_hashes, read_state

DATA_FOLDER = "dataset"

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def ejecutar_carga():
    """
    Carga todos los CSV en ./dataset y detecta cambios por hash.
    Retorna dict {filename: dataframe} y dict rows {filename: rows}.
    """
    update_stage("Carga de datos", "running", "Iniciando carga de archivos")
    if not os.path.exists(DATA_FOLDER):
        update_stage("Carga de datos", "error", f"Carpeta {DATA_FOLDER} no encontrada")
        return None, {}

    hashes_prev = read_hashes()
    hashes_now = {}
    dataframes = {}
    rows = {}
    changed_any = False

    for fname in sorted(os.listdir(DATA_FOLDER)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(DATA_FOLDER, fname)
        try:
            h = file_hash(path)
            hashes_now[fname] = h
            df = pd.read_csv(path)
            dataframes[fname] = df
            rows[fname] = int(df.shape[0])
            if hashes_prev.get(fname) != h:
                changed_any = True
                log(f"CAMBIO detectado en {fname} (filas={rows[fname]})")
        except Exception as e:
            log(f"ERROR cargando {fname}: {e}")

    # Save current hashes
    write_hashes(hashes_now)

    if not dataframes:
        update_stage("Carga de datos", "error", "No se cargó ningún CSV")
        return None, {}

    update_stage("Carga de datos", "success", f"Cargados {len(dataframes)} archivos")
    return dataframes, rows
