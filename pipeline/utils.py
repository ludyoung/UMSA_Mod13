import os
import json
from datetime import datetime
from typing import Dict, Any

LOG_PATH = os.path.join("logs", "pipeline.log")
STATE_PATH = os.path.join("logs", "pipeline_state.json")
HASHES_PATH = os.path.join("config", "hashes.json")
CONFIG_DIR = "config"

os.makedirs("logs", exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def log(msg: str):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {msg}\n")

def read_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        base = {
            "last_run": None,
            "status": {
                "Carga de datos": "idle",
                "EDA": "idle",
                "Transformación": "idle",
                "Modelo": "idle"
            },
            "files_rows": {},   # {filename: last_row_count}
            "last_message": ""
        }
        write_state(base)
        return base
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def write_state(state: Dict[str, Any]):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

def update_stage(stage_name: str, stage_status: str, message: str = None):
    s = read_state()
    s["status"][stage_name] = stage_status
    if message:
        s["last_message"] = f"{now_iso()} - {message}"
    if stage_status in ("success", "idle", "error"):
        s["last_run"] = now_iso()
    write_state(s)
    log(f"STAGE {stage_name}: {stage_status} - {message or ''}")

def read_hashes() -> dict:
    if not os.path.exists(HASHES_PATH):
        return {}
    with open(HASHES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def write_hashes(hashes: dict):
    with open(HASHES_PATH, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=4)
