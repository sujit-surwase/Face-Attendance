import os
import csv
from datetime import datetime

DATASET_DIR = "dataset"
ATTENDANCE_FILE = "Attendance.csv"

def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    existing = set()
    try:
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    existing.add((row[0], row[1]))
    except FileNotFoundError:
        pass

    if (name, date_str) not in existing:
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        print(f"[✓] Attendance marked: {name} at {time_str}")
        return True
    else:
        print(f"[!] {name} already marked today.")
        return False

def get_registered_names():
    if not os.path.exists(DATASET_DIR):
        return []
    return [d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))]