<div align="center">

<img src="https://img.icons8.com/fluency/96/face-id.png" width="100"/>

# 🎓 Face Attendance System

### Real-time Automatic Attendance using Deep Learning Face Recognition

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-FaceNet-purple?style=for-the-badge&logo=tensorflow&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-orange?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**Made by [Sujit Surwase](https://linkedin.com/in/sujit-surwase)**
📍 Pune, Maharashtra, India

</div>

---

## 📌 Table of Contents

- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [How to Use](#-how-to-use)
- [All Source Code](#-all-source-code)
- [Configuration](#-configuration)
- [Attendance CSV Format](#-attendance-csv-format)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 Face Registration | Register any person with 100 auto-captured images inside the GUI |
| ✅ Real-time Attendance | Live camera feed embedded directly inside the app — no popups |
| 🎯 Strict Matching | FaceNet + Cosine distance for exact face verification |
| 🚫 Anti-Duplicate | Same person cannot be marked twice on the same day |
| 📋 Attendance Logs | Records saved to CSV with Name, Date, Time |
| 🖥️ Full Screen GUI | Beautiful dark-themed full screen Tkinter interface |
| 📊 View Records | Browse complete history in-app |
| ⚡ Optimized | Background threading — smooth camera at low CPU usage |
| 🗑️ Clear Today | Reset today's records with one click |
| 📈 Progress Bar | Live progress bar during face registration |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Face Detection | OpenCV Haar Cascade |
| Face Recognition | DeepFace (FaceNet model) |
| Deep Learning Backend | TensorFlow 2.x + tf-keras |
| GUI Framework | Tkinter + ttk |
| Image Processing | OpenCV + Pillow |
| Data Storage | CSV |
| Threading | Python `threading` module |

---


## 📁 Project Structure

```
Face-Attendance/
│
├── main.py                  ← Main GUI Application (Full Screen)
├── register.py              ← Face Registration Logic  
├── attendance.py            ← Face Recognition & Attendance Marking
├── utils.py                 ← Helper Functions (CSV, Paths)
├── requirements.txt         ← All dependencies
├── README.md                ← This file
│
├── dataset/                 ← Stores face images per person
│   ├── Alice/
│   │   ├── Alice_0.jpg
│   │   └── Alice_99.jpg
│   └── Bob/
│       ├── Bob_0.jpg
│       └── Bob_99.jpg
│
└── Attendance.csv           ← Auto-created attendance log
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.12 installed
- Webcam connected
- Windows 10/11 (tested)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/face-attendance-system.git
cd face-attendance-system
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\Activate.ps1

# Activate on Mac/Linux
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the App

```bash
python main.py
```

> ⚠️ **First Run:** DeepFace auto-downloads FaceNet model (~90MB). Wait for it once — never again.

---

## 🚀 How to Use

### 📷 Register a Face
1. Launch app → click **Register Face**
2. Type the person's full name → click OK
3. Camera opens inside the app — look at it
4. Move face slightly **left → right → up → down** for better accuracy
5. Progress bar fills to 100 → registration auto-completes ✅

### ✅ Mark Attendance
1. Click **Start Attendance**
2. Live camera feed appears inside the app
3. Recognized face → name + time logged automatically
4. Today's attendance updates live in the left table
5. Click **Stop Camera** when done

### 📋 View / Export Records
- **Today's records** — shown in left panel live
- **All records** — click **View All Records**
- **Excel export** — open `Attendance.csv` directly in Excel

---

## 💻 All Source Code

### `requirements.txt`
opencv-python
deepface
tf-keras
pandas
numpy
pillow


---

### `utils.py`

```python
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
                    existing.add((row, row))[1]
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
```

---

### `register.py`

```python
import cv2
import os
from utils import DATASET_DIR, ensure_dirs

def register_face(name: str, num_images: int = 100):
    ensure_dirs()
    if not name.strip():
        print("[!] Name cannot be empty.")
        return False
    person_dir = os.path.join(DATASET_DIR, name.strip())
    os.makedirs(person_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    count = 0
    print(f"[*] Registering '{name}'. Capturing {num_images} images...")
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            if w > 100 and h > 100:
                face_img = frame[y:y+h, x:x+w]
                img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if count > 0:
        print(f"[✓] Registered '{name}' with {count} images.")
        return True
    else:
        print("[!] No face detected.")
        return False
```

---

### `attendance.py`

```python
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
from deepface import DeepFace
from utils import DATASET_DIR, mark_attendance, get_registered_names

MATCH_THRESHOLD = 0.40

def recognize_face(face_img, idx):
    temp_path = f"temp_face_{idx}.jpg"
    cv2.imwrite(temp_path, face_img)
    try:
        result = DeepFace.find(
            img_path=temp_path,
            db_path=DATASET_DIR,
            model_name="Facenet",
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False,
            silent=True
        )
        if result and len(result) > 0:
            top_match = result.iloc
            distance = top_match["distance"]
            if distance < MATCH_THRESHOLD:
                identity_path = top_match["identity"]
                name = os.path.normpath(identity_path).split(os.sep)[-2]
                mark_attendance(name)
                return (f"{name} ✓ ({distance:.2f})", (0, 255, 0))
            else:
                return (f"Unknown ({distance:.2f})", (0, 0, 255))
        else:
            return ("Unknown", (0, 0, 255))
    except Exception as e:
        print(f"[Error] {e}")
        return ("Error", (0, 165, 255))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def run_attendance():
    import threading
    registered = get_registered_names()
    if not registered:
        print("[!] No registered faces found.")
        return
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    frame_count = 0
    recognition_result = {}
    is_recognizing = False

    def _bg_recognize(face_img, idx):
        nonlocal is_recognizing
        label, color = recognize_face(face_img, idx)
        recognition_result[idx] = (label, color)
        is_recognizing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        if frame_count % 30 == 0 and len(faces) > 0 and not is_recognizing:
            is_recognizing = True
            x, y, w, h = faces
            threading.Thread(
                target=_bg_recognize,
                args=(frame[y:y+h, x:x+w].copy(), 0),
                daemon=True
            ).start()
        for idx, (x, y, w, h) in enumerate(faces):
            label, color = recognition_result.get(idx, ("Detecting...", (255, 200, 0)))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-28), (x+w, y), color, -1)
            cv2.putText(frame, label, (x+4, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attendance()
```

---

### `main.py`

```python
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import csv
import threading
import cv2
from PIL import Image, ImageTk
from utils import ensure_dirs, ATTENDANCE_FILE, get_registered_names
from attendance import recognize_face, MATCH_THRESHOLD

ensure_dirs()

BG     = "#1e1e2e"
PANEL  = "#2a2a3e"
ACCENT = "#7c3aed"
BTN_FG = "#ffffff"
TEXT   = "#e0e0e0"
GREEN  = "#22c55e"
RED    = "#ef4444"
ORANGE = "#f97316"
BLUE   = "#0ea5e9"
YELLOW = "#eab308"

class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Face Attendance System")
        self.root.configure(bg=BG)
        self.root.state("zoomed")
        self.root.update()
        self.cap = None
        self.camera_running = False
        self.camera_mode = None
        self.frame_count = 0
        self.is_recognizing = False
        self.recognition_result = {}
        self.reg_name = ""
        self.reg_count = 0
        self.reg_total = 100
        self.reg_person_dir = ""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg=ACCENT, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="🎓 Face Attendance System — Strict Mode",
                 font=("Segoe UI", 20, "bold"),
                 bg=ACCENT, fg=BTN_FG).pack(expand=True)

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=15, pady=10)

        left = tk.Frame(content, bg=PANEL, width=290,
                        highlightbackground=ACCENT, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(left, text="Controls",
                 font=("Segoe UI", 13, "bold"),
                 bg=PANEL, fg=ACCENT).pack(pady=(18, 8))

        self._btn(left, "📷  Register Face",   GREEN,  self.register_action)
        self._btn(left, "✅  Start Attendance", ACCENT, self.start_attendance)
        self._btn(left, "⏹️  Stop Camera",      RED,    self.stop_camera)
        self._btn(left, "📋  View All Records", ORANGE, self.view_records)
        self._btn(left, "🔄  Refresh Table",    BLUE,   self.refresh_table)
        self._btn(left, "🗑️  Clear Today",      RED,    self.clear_today)

        self.reg_var = tk.StringVar()
        self._update_reg_count()
        tk.Label(left, textvariable=self.reg_var,
                 font=("Segoe UI", 10), bg=PANEL, fg=TEXT).pack(pady=(10, 2))

        self.status_var = tk.StringVar(value="Ready ✓")
        tk.Label(left, textvariable=self.status_var,
                 font=("Segoe UI", 9), bg=PANEL, fg=TEXT,
                 wraplength=260).pack(pady=4)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(left, variable=self.progress_var,
                                         maximum=100, length=250,
                                         mode="determinate")
        self.progress.pack(pady=4, padx=12)
        self.progress_label = tk.Label(left, text="",
                                        font=("Segoe UI", 9),
                                        bg=PANEL, fg=YELLOW)
        self.progress_label.pack()

        tk.Frame(left, bg=ACCENT, height=1).pack(fill="x", padx=10, pady=10)

        tk.Label(left, text="Today's Attendance",
                 font=("Segoe UI", 11, "bold"),
                 bg=PANEL, fg=TEXT).pack(pady=(0, 6))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=PANEL, foreground=TEXT,
                        rowheight=26, fieldbackground=PANEL,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading",
                        background=ACCENT, foreground=BTN_FG,
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])

        tree_frame = tk.Frame(left, bg=PANEL)
        tree_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.tree = ttk.Treeview(tree_frame, columns=("Name", "Time"),
                                  show="headings")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Time", text="Time")
        self.tree.column("Name", anchor="center", width=145)
        self.tree.column("Time", anchor="center", width=95)
        sb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.refresh_table()

        right = tk.Frame(content, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self.cam_title = tk.Label(right, text="📹 Camera Feed",
                                   font=("Segoe UI", 12, "bold"),
                                   bg=BG, fg=TEXT)
        self.cam_title.pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(right, bg="#0a0a14",
                                 highlightbackground=ACCENT,
                                 highlightthickness=2)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()
        self._show_placeholder()

    def _show_placeholder(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.create_text(cw // 2, ch // 2,
                                 text="📷 Click 'Register Face' or 'Start Attendance'",
                                 fill="#555577", font=("Segoe UI", 15))

    def _btn(self, parent, text, color, cmd):
        tk.Button(parent, text=text,
                  font=("Segoe UI", 10, "bold"),
                  bg=color, fg=BTN_FG, activebackground=color,
                  relief="flat", cursor="hand2",
                  width=28, height=1,
                  command=cmd).pack(pady=5, padx=12)

    def _update_reg_count(self):
        self.reg_var.set(f"👤 Registered: {len(get_registered_names())} person(s)")

    def register_action(self):
        if self.camera_running:
            messagebox.showwarning("Camera Busy", "Stop the current camera first.")
            return
        name = simpledialog.askstring("Register Face",
                                      "Enter the person's full name:",
                                      parent=self.root)
        if not name or not name.strip():
            return
        self.reg_name = name.strip()
        self.reg_count = 0
        self.reg_total = 100
        from utils import DATASET_DIR
        self.reg_person_dir = os.path.join(DATASET_DIR, self.reg_name)
        os.makedirs(self.reg_person_dir, exist_ok=True)
        self.camera_mode = "register"
        self.camera_running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cam_title.config(text=f"📷 Registering: {self.reg_name}")
        self.status_var.set(f"Registering '{self.reg_name}'... (0/{self.reg_total})")
        self.progress_var.set(0)
        self.progress_label.config(text=f"0 / {self.reg_total} images captured")
        self._update_frame()

    def start_attendance(self):
        if self.camera_running:
            messagebox.showwarning("Camera Busy", "Stop the current camera first.")
            return
        if not get_registered_names():
            messagebox.showwarning("No Faces", "No registered faces! Please register first.")
            return
        self.camera_mode = "attendance"
        self.frame_count = 0
        self.is_recognizing = False
        self.recognition_result = {}
        self.camera_running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cam_title.config(text="✅ Attendance Mode — Live Feed")
        self.status_var.set("🎥 Attendance running...")
        self.progress_label.config(text="")
        self.progress_var.set(0)
        self._update_frame()

    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_mode = None
        self.cam_title.config(text="📹 Camera Feed")
        self.status_var.set("Camera stopped ✓")
        self.progress_var.set(0)
        self.progress_label.config(text="")
        self._show_placeholder()
        self.refresh_table()
        self._update_reg_count()

    def _update_frame(self):
        if not self.camera_running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        if self.camera_mode == "register":
            for (x, y, w, h) in faces:
                if w > 100 and h > 100 and self.reg_count < self.reg_total:
                    cv2.imwrite(
                        os.path.join(self.reg_person_dir,
                                     f"{self.reg_name}_{self.reg_count}.jpg"),
                        frame[y:y+h, x:x+w]
                    )
                    self.reg_count += 1
                    pct = (self.reg_count / self.reg_total) * 100
                    self.progress_var.set(pct)
                    self.progress_label.config(
                        text=f"{self.reg_count} / {self.reg_total} images captured")
                    self.status_var.set(
                        f"Registering '{self.reg_name}'... ({self.reg_count}/{self.reg_total})")
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0) if w > 100 else (0, 165, 255), 2)
            cv2.putText(frame,
                        f"Registering: {self.reg_name}  {self.reg_count}/{self.reg_total}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
            cv2.putText(frame, "Look: straight → left → right → up → down",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            if self.reg_count >= self.reg_total:
                self.stop_camera()
                self.progress_var.set(100)
                self.progress_label.config(text="✅ Registration complete!")
                self.status_var.set(f"'{self.reg_name}' registered ✓")
                messagebox.showinfo("Success",
                                    f"✅ '{self.reg_name}' registered successfully!")
                return

        elif self.camera_mode == "attendance":
            self.frame_count += 1
            if self.frame_count % 30 == 0 and len(faces) > 0 and not self.is_recognizing:
                self.is_recognizing = True
                x, y, w, h = faces
                threading.Thread(target=self._bg_recognize,
                                  args=(frame[y:y+h, x:x+w].copy(), 0),
                                  daemon=True).start()
            for idx, (x, y, w, h) in enumerate(faces):
                label, color = self.recognition_result.get(
                    idx, ("Detecting...", (255, 200, 0)))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.rectangle(frame, (x, y-32), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+6, y-9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status = "Processing..." if self.is_recognizing else "Watching..."
            cv2.putText(frame, f"Status: {status}  |  Threshold: {MATCH_THRESHOLD}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((cw, ch), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.root.after(30, self._update_frame)

    def _bg_recognize(self, face_img, idx):
        label, color = recognize_face(face_img, idx)
        self.recognition_result[idx] = (label, color)
        self.is_recognizing = False
        self.root.after(0, self.refresh_table)

    def refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        try:
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            with open(ATTENDANCE_FILE, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 3 and row == today:[1]
                        self.tree.insert("", "end", values=(row, row))[2]
        except FileNotFoundError:
            pass

    def view_records(self):
        win = tk.Toplevel(self.root)
        win.title("All Attendance Records")
        win.geometry("700x500")
        win.configure(bg=BG)
        tk.Label(win, text="📋 All Attendance Records",
                 font=("Segoe UI", 14, "bold"),
                 bg=BG, fg=TEXT).pack(pady=12)
        cols = ("Name", "Date", "Time")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=210)
        tree.pack(fill="both", expand=True, padx=12, pady=6)
        sb2 = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscroll=sb2.set)
        sb2.pack(side="right", fill="y")
        try:
            with open(ATTENDANCE_FILE, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        tree.insert("", "end", values=row)
        except FileNotFoundError:
            messagebox.showinfo("Info", "No attendance records yet.")

    def clear_today(self):
        if messagebox.askyesno("Confirm", "Clear today's attendance records?"):
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            rows = []
            try:
                with open(ATTENDANCE_FILE, "r") as f:
                    reader = csv.reader(f)
                    rows = [row for row in reader
                            if row == "Name" or row != today][1]
            except FileNotFoundError:
                return
            with open(ATTENDANCE_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            self.refresh_table()
            self.status_var.set("Today's records cleared ✓")

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
```

---

## 🎯 Face Matching Configuration

Edit `MATCH_THRESHOLD` in `attendance.py`:

```python
MATCH_THRESHOLD = 0.40
```

| Threshold | Strictness | Use Case |
|-----------|------------|----------|
| `0.20` | 🔴 Very Strict | Twin/lookalike prevention |
| `0.30` | 🟡 Strict | Office / School |
| `0.40` | 🟢 Normal | General purpose |
| `0.50` | ⚪ Loose | Low-light environments |

---

## 📊 Attendance CSV Format

```csv
Name,Date,Time
Alice,2026-04-24,09:05:32
Bob,2026-04-24,09:07:15
Alice,2026-04-25,08:58:44
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `tf-keras not found` | `pip install tf-keras` |
| `No module named PIL` | `pip install pillow` |
| Camera not opening | Check if webcam is used by another app |
| Face not detected | Better lighting, move closer |
| Always "Unknown" | Lower threshold or re-register |
| High CPU usage | Already optimized — runs every 30 frames |
| First run slow | DeepFace downloads model once (~90MB) |

---

## 🔮 Future Improvements

- [ ] 📧 Auto email daily attendance report
- [ ] 📱 WhatsApp/SMS alerts via Twilio
- [ ] 🌐 Web dashboard using Flask/Django
- [ ] 🕐 Late arrival detection
- [ ] 🔐 Admin login with password
- [ ] 📸 Anti-spoofing (reject photos/screens)
- [ ] 📊 Power BI / Excel analytics integration
- [ ] 🤖 Multi-face detection simultaneously

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m "Add AmazingFeature"`
4. Push: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 👨‍💻 Author

**Sujit Surwase**
- 🐙 GitHub: [@sujit-surwase](https://github.com/sujit-surwase)
- 💼 LinkedIn: [Sujit Surwase](https://linkedin.com/in/sujit-surwase)
- 📍 Pune, Maharashtra, India

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

Made with ❤️ using Python, DeepFace & OpenCV

</div>