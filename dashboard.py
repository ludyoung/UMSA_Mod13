import os
import json
import subprocess
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, RerunData

# -------------------------
# Config paths & constantes
# -------------------------
ROOT = Path(".").resolve()
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "pipeline.log"
DATASET_DIR = ROOT / "dataset"
IMAGES_EDA_DIR = ROOT / "Images/EDA"
STATE_PATH = ROOT / "state.json"

# -------------------------
# Helpers: schedule + state
# -------------------------
def load_schedule():
    cfg_path = ROOT / "config" / "schedule.json"
    if not cfg_path.exists():
        return {"enabled": False, "frequency": "manual", "hour": 0, "minute": 0, "weekday": 0, "day": 1}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except:
        return {"enabled": False, "frequency": "manual", "hour": 0, "minute": 0, "weekday": 0, "day": 1}

def save_schedule(cfg):
    cfg_path = ROOT / "config" / "schedule.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")

def read_state():
    if not STATE_PATH.exists():
        return {"status": {}, "last_run": None, "files_rows": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except:
        return {"status": {}, "last_run": None, "files_rows": {}}

# -------------------------
# Utility: tail log file
# -------------------------
def tail_lines(path: Path, n: int = 200):
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        size = 1024
        data = b""
        while end > 0 and len(data.splitlines()) <= n:
            start = max(0, end - size)
            f.seek(start)
            chunk = f.read(end - start)
            data = chunk + data
            end = start
            size *= 2
        try:
            return data.decode(errors="replace").splitlines()[-n:]
        except:
            return []

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Pipeline Monitor", layout="wide")
st.title("Pipeline Monitor — Dashboard")

# -------------------------
# Crear pestañas
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ejecución / Configuración",
    "Feature Engineering",
    "Correlación de Features",
    "Metricas de satisfacción",
    "Comparación de REAL vs PREDICCION"
])

# -------------------------
# PESTAÑA 1: Principal
# -------------------------
with tab1:
    left, right = st.columns([1, 2])

    # --- LEFT: Scheduler y acciones
    with left:
        st.header("Scheduler")
        cfg = load_schedule()

        enabled = st.checkbox("Habilitar scheduler", value=cfg.get("enabled", False))

        frequency = st.selectbox(
            "Frecuencia",
            ["manual","hourly","daily","weekly","monthly"],
            index=["manual","hourly","daily","weekly","monthly"].index(cfg.get("frequency","manual"))
        )

        hour = st.number_input("Hora (0-23)", min_value=0, max_value=23, value=cfg.get("hour",0))
        minute = st.number_input("Minuto (0-59)", min_value=0, max_value=59, value=cfg.get("minute",0))
        weekday = st.selectbox(
            "Día semana (weekly)",
            ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            index=cfg.get("weekday",0)
        )

        day_month = st.number_input(
            "Día del mes (monthly)", min_value=1, max_value=31,
            value=cfg.get("day",1)
        ) if frequency=="monthly" else 1

        if st.button("Guardar programación"):
            new_cfg = {
                "enabled": bool(enabled),
                "frequency": frequency,
                "hour": int(hour),
                "minute": int(minute),
                "weekday": int(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(weekday)),
                "day": int(day_month) if frequency=="monthly" else 1
            }
            save_schedule(new_cfg)
            st.success("Programación guardada correctamente.")

        st.markdown("---")
        st.header("Acciones por realizar")

        # --- BOTÓN: Ejecutar pipeline completo ---
        if st.button("Ejecutar pipeline completo."):
            py = os.sys.executable
            try:
                LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(LOG_PATH, "a") as f:
                    subprocess.Popen([py, str(ROOT / "main.py")], stdout=f, stderr=f)
                st.info("Pipeline lanzado en background. Revisar logs en 'logs/pipeline.log'.")
            except Exception as e:
                st.error(f"Error al lanzar pipeline: {e}")

        st.markdown("---")
        st.header("Archivos del dataset")
        files = sorted([p.name for p in DATASET_DIR.glob("*.csv")]) if DATASET_DIR.exists() else []

        if files:
            rows = {}
            for f in files:
                try:
                    file_path = DATASET_DIR / f
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                        rc = sum(1 for _ in fh) - 1
                    rows[f] = rc
                except:
                    rows[f] = None
            df_files = pd.DataFrame.from_dict(rows, orient="index", columns=["rows"])
            st.dataframe(df_files)
        else:
            st.info("No hay archivos CSV en dataset/")

    # --- RIGHT: Logs en tiempo real ---
    with right:
        st.subheader("Logs pipeline")
        if LOG_PATH.exists():
            n_lines = 1080
            lines = tail_lines(LOG_PATH, n=n_lines)
            styled = []
            for line in lines:
                safe = (
                    line.replace("&","&amp;")
                        .replace("<","&lt;")
                        .replace(">","&gt;")
                )
                if "ERROR" in line or "CRITICAL" in line:
                    styled.append(f"<span style='color:red'>{safe}</span>")
                elif "WARNING" in line:
                    styled.append(f"<span style='color:orange'>{safe}</span>")
                elif "INFO" in line:
                    styled.append(f"<span style='color:lightblue'>{safe}</span>")
                else:
                    styled.append(f"<span style='color:white'>{safe}</span>")

            st.markdown(
                f"""
                <div style='height:1080px; overflow-y:scroll; background-color:#0b0b0b; padding:10px; border-radius:6px; font-family:monospace; white-space:pre; font-size:10px;'>
                {'<br>'.join(styled)}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Aún no se ha generado pipeline.log")

# -------------------------
# PESTAÑA 2: Features
# -------------------------
with tab2:
    feature_img_path = Path("./Images/FEATURES/resumen_features_hierarchy.png")
    if feature_img_path.exists():
        st.markdown("### Jerarquía de Features Generados")
        st.image(str(feature_img_path))
    else:
        st.warning("No se encontró la imagen jerárquica de features.")

# -------------------------
# PESTAÑA 3: 
# -------------------------
with tab3:
    from pathlib import Path

    IMAGES_FEATURES_DIR = Path("./Images/DATAFRAME/")
    st.subheader("LIMPIEZA Y CORRELACIÓN DE FEATURES")

    if IMAGES_FEATURES_DIR.exists():
        imgs = sorted([f for f in IMAGES_FEATURES_DIR.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

        if imgs:
            for img in imgs:
                st.markdown(f"### {img.name}")
                st.image(str(img))
        else:
            st.info("No se encontraron imágenes en Images/DATAFRAME/")
    else:
        st.warning("La carpeta Images/DATAFRAME no existe.")

# -------------------------
# PESTAÑA 4: 
# -------------------------
with tab4:
    from pathlib import Path

    IMAGES_FEATURES_DIR = Path("./Images/RESULTADOS/")
    st.subheader("Métricas de satisfacción")

    if IMAGES_FEATURES_DIR.exists():
        imgs = sorted([f for f in IMAGES_FEATURES_DIR.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

        if imgs:
            for img in imgs:
                st.markdown(f"### {img.name}")
                st.image(str(img))
        else:
            st.info("No se encontraron imágenes en Images/DATAFRAME/")
    else:
        st.warning("La carpeta Images/DATAFRAME no existe.")

# -------------------------
# PESTAÑA 5: 
# -------------------------
with tab5:
    from pathlib import Path

    IMAGES_FEATURES_DIR = Path("./Images/PREDICCIONES/")
    st.subheader("Valores REALES vs PREDECIDOS")

    if IMAGES_FEATURES_DIR.exists():
        imgs = sorted([f for f in IMAGES_FEATURES_DIR.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

        if imgs:
            for img in imgs:
                st.markdown(f"### {img.name}")
                st.image(str(img))
        else:
            st.info("No se encontraron imágenes en Images/DATAFRAME/")
    else:
        st.warning("La carpeta Images/DATAFRAME no existe.")
# -------------------------
# Auto-refresh
# -------------------------
refresh_ms = st.sidebar.number_input("Refresh (ms)", min_value=1000, max_value=15000, value=3000, step=500)
auto = st.sidebar.checkbox("Auto-refresh dashboard", value=True)
if auto:
    time.sleep(refresh_ms / 1000)
    raise RerunException(RerunData())
