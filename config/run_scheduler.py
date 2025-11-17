"""
Script a ejecutar periódicamente (ej.: cada 5 min desde Task Scheduler o cron).
Lee config/schedule.json y decide si debe ejecutar main.py.
"""
import json, os, subprocess
from datetime import datetime, timedelta
from pipeline.utils import read_state, log

CFG_PATH = os.path.join("config", "schedule.json")
CHECK_INTERVAL_MIN = 5  # if you schedule this script every 5 minutes

def load_cfg():
    if not os.path.exists(CFG_PATH):
        return {"enabled": False, "frequency": "manual"}
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def should_run(cfg, last_run_iso):
    if not cfg.get("enabled", False):
        return False
    freq = cfg.get("frequency", "manual")
    now = datetime.utcnow()

    last_run = None
    if last_run_iso:
        try:
            last_run = datetime.fromisoformat(last_run_iso.replace("Z",""))
        except:
            last_run = None

    if freq == "hourly":
        if not last_run:
            return True
        return (now - last_run) >= timedelta(hours=1)
    if freq == "daily":
        # check hour/minute
        hr = cfg.get("hour", 0)
        mn = cfg.get("minute", 0)
        target = datetime.utcnow().replace(hour=hr, minute=mn, second=0, microsecond=0)
        if last_run and last_run >= target:
            # already ran today at/after target
            return False
        return now >= target
    if freq == "weekly":
        # weekday 0..6
        wd = cfg.get("weekday", 0)
        hr = cfg.get("hour", 0)
        mn = cfg.get("minute", 0)
        # find this week's target
        today = datetime.utcnow()
        # compute most recent target
        days_ago = (today.weekday() - wd) % 7
        target = (today - timedelta(days=days_ago)).replace(hour=hr, minute=mn, second=0, microsecond=0)
        if last_run and last_run >= target:
            return False
        return now >= target
    if freq == "monthly":
        hr = cfg.get("hour", 0)
        mn = cfg.get("minute", 0)
        day = cfg.get("day", 1)
        today = datetime.utcnow()
        try:
            target = today.replace(day=day, hour=hr, minute=mn, second=0, microsecond=0)
        except ValueError:
            # invalid day (e.g., 31 in feb) -> pick last day of month
            from calendar import monthrange
            lastday = monthrange(today.year, today.month)[1]
            target = today.replace(day=lastday, hour=hr, minute=mn, second=0, microsecond=0)
        if last_run and last_run >= target:
            return False
        return now >= target
    return False

def run_main():
    # call main.py with the same python interpreter
    import sys
    py = sys.executable
    log("Scheduler invoking main.py")
    subprocess.run([py, "main.py"])

def main():
    cfg = load_cfg()
    st = read_state()
    last_run = st.get("last_run")
    if should_run(cfg, last_run):
        run_main()
    else:
        log("Scheduler checked - not due")

if __name__ == "__main__":
    main()
