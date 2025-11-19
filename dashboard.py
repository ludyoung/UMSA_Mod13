import streamlit as st
import os
import json
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px
import subprocess
from pipeline.utils import read_state, log

# Setup
st.set_page_config(page_title="Pipeline Scheduler & Monitor", layout="wide")
st.title("Pipeline Scheduler & Monitor")

os.makedirs("temp", exist_ok=True)
HTML_PATH = os.path.join("temp", "pipeline.html")
SCHEDULE_PATH = os.path.join("config", "schedule.json")

# Helpers
def load_schedule():
    if not os.path.exists(SCHEDULE_PATH):
        return {"enabled": False, "frequency": "manual", "hour": 0, "minute":0, "weekday":0, "day":1}
    with open(SCHEDULE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_schedule(cfg):
    os.makedirs(os.path.dirname(SCHEDULE_PATH), exist_ok=True)
    with open(SCHEDULE_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

def draw_dag(state):
    net = Network(height="350px", width="100%", directed=True)
    # base nodes fixed
    nodes = ["Carga de datos","EDA","Transformación","Modelo"]
    color_map = {"idle":"#B0C4DE", "running":"orange", "success":"lightgreen", "error":"red"}
    # add nodes once with color according to state
    for n in nodes:
        stt = state["status"].get(n, "idle")
        color = color_map.get(stt, "#B0C4DE")
        net.add_node(n, label=n, color=color)
    # edges
    edges = [("Carga de datos","EDA"), ("EDA","Transformación"), ("Transformación","Modelo")]
    for s,t in edges:
        net.add_edge(s,t)
    net.write_html(HTML_PATH, notebook=False, local=True)
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        components.html(f.read(), height=380)

# Layout: left (controls), right (status + logs)
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Scheduler")
    cfg = load_schedule()
    enabled = st.checkbox("Habilitar scheduling (ejecución automática via run_scheduler)", value=cfg.get("enabled", False))
    freq = st.selectbox("Frecuencia", ["manual","hourly","daily","weekly","monthly"], index=["manual","hourly","daily","weekly","monthly"].index(cfg.get("frequency","manual")))
    hour = st.number_input("Hora (0-23)", min_value=0, max_value=23, value=cfg.get("hour",0))
    minute = st.number_input("Minuto (0-59)", min_value=0, max_value=59, value=cfg.get("minute",0))
    weekday = st.selectbox("Día semana (para weekly)", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], index=cfg.get("weekday",0))
    day_month = st.number_input("Día del mes (1-28/31) (para monthly)", min_value=1, max_value=31, value=cfg.get("day",1)) if freq=="monthly" else st.empty()

    if st.button("Guardar programación"):
        new_cfg = {
            "enabled": bool(enabled),
            "frequency": freq,
            "hour": int(hour),
            "minute": int(minute),
            "weekday": int(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(weekday)),
            "day": int(day_month) if freq=="monthly" else 1
        }
        save_schedule(new_cfg)
        st.success("Programación guardada en config/schedule.json")

    st.markdown("---")
    st.subheader("Ejecutar ahora")
    if st.button("Ejecutar Pipeline ahora"):
        # Launch main.py in background (non-blocking)
        py = os.sys.executable
        # Use subprocess.Popen to not block the Streamlit process
        subprocess.Popen([py, "main.py"])
        st.info("Pipeline lanzado en background; revisa logs y estado.")

with col2:
    st.subheader("Estado actual del pipeline")
    state = read_state()
    # Draw DAG
    

    
   
    st.markdown("---")
    st.subheader("Gráfica: Filas anteriores vs actuales")
    # state['files_rows'] contains latest rows from last main run
    files_rows = state.get("files_rows", {})
    # keep history in session to compare
    if "prev_files_rows" not in st.session_state:
        st.session_state.prev_files_rows = {}

    # When new run detected (last_run changed), update prev <- old
    last_run = state.get("last_run")
    if "last_run_recorded" not in st.session_state:
        st.session_state.last_run_recorded = None

    if last_run and st.session_state.last_run_recorded != last_run:
        # new run; move prev to prev_files_rows and set last_run_recorded
        st.session_state.prev_files_rows = st.session_state.get("last_files_rows_snapshot", {})
        st.session_state.last_files_rows_snapshot = files_rows.copy()
        st.session_state.last_run_recorded = last_run

    # build DataFrame for plotting
    import pandas as pd
    names = sorted(set(list(st.session_state.prev_files_rows.keys()) + list(files_rows.keys())))
    prev_vals = [st.session_state.prev_files_rows.get(n, 0) for n in names]
    curr_vals = [files_rows.get(n, 0) for n in names]
    if names:
        df_plot = pd.DataFrame({
            "Archivo": names,
            "Previas": prev_vals,
            "Actuales": curr_vals
        })
        fig = px.bar(df_plot, x="Archivo", y=["Previas","Actuales"], barmode="group",
                     title="Comparativa: filas previas vs filas actuales")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aún no hay datos cargados por el pipeline.")

    st.markdown("---")
    st.subheader("Logs (últimas líneas)")
    logpath = os.path.join("logs","pipeline.log")
    if os.path.exists(logpath):
        with open(logpath, "r", encoding="utf-8") as f:
            lines = f.readlines()[-200:]
            st.code("".join(lines[-200:]))
    else:
        st.info("Aún no hay logs.")



