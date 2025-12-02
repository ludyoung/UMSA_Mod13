import os
import time
import json
import datetime
from pathlib import Path

# Rutas
ROOT = Path(".").resolve()
SCHEDULE_PATH = ROOT / "config" / "schedule.json"

def load_schedule():
    """Leer la configuración del scheduler desde JSON."""
    if not SCHEDULE_PATH.exists():
        # Configuración por defecto
        return {"enabled": False, "frequency": "manual", "hour": 0, "minute": 0, "weekday": 0, "day": 1}
    try:
        return json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error leyendo config: {e}")
        return {"enabled": False, "frequency": "manual", "hour": 0, "minute": 0, "weekday": 0, "day": 1}

def should_run(cfg, now):
    """Verifica si el pipeline debe ejecutarse en el momento actual."""
    if not cfg.get("enabled", False):
        return False
    
    freq = cfg.get("frequency", "manual")
    hour = cfg.get("hour", 0)
    minute = cfg.get("minute", 0)
    
    if freq == "manual":
        return False
    elif freq == "hourly":
        return now.minute == minute
    elif freq == "daily":
        return now.hour == hour and now.minute == minute
    elif freq == "weekly":
        return now.weekday() == cfg.get("weekday", 0) and now.hour == hour and now.minute == minute
    elif freq == "monthly":
        return now.day == cfg.get("day", 1) and now.hour == hour and now.minute == minute
    return False

def run_pipeline():
    """Lanza main.py en background."""
    py = os.sys.executable
    try:
        os.system(f'"{py}" main.py')  # ejecuta main.py con Python
        print(f"[{datetime.datetime.now()}] Pipeline ejecutado.")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Error ejecutando pipeline: {e}")

def main():
    last_run_time = None
    while True:
        cfg = load_schedule()
        now = datetime.datetime.now()
        # Ejecutar solo si no lo hemos corrido este minuto
        if should_run(cfg, now):
            if last_run_time != (now.year, now.month, now.day, now.hour, now.minute):
                run_pipeline()
                last_run_time = (now.year, now.month, now.day, now.hour, now.minute)
        time.sleep(30)  # revisa cada 30 segundos

if __name__ == "__main__":
    print("Scheduler iniciado...")
    main()