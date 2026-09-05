"""Simple stdlib scheduler. Prefer OS Task Scheduler/cron for production."""
from __future__ import annotations
import os,time,subprocess
from datetime import datetime
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while True:
    now=datetime.now()
    if now.hour==10 and now.minute==0:
        subprocess.run([os.path.join(ROOT,'.venv','Scripts','python.exe'),os.path.join(ROOT,'scripts','daily_job.py')],cwd=ROOT,check=False)
        time.sleep(65)
    time.sleep(20)
