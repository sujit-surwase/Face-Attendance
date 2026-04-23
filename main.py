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

        # ✅ Full screen
        self.root.state("zoomed")              # Windows full screen
        self.root.update()

        # Get actual screen size after zoom
        self.screen_w = self.root.winfo_width()
        self.screen_h = self.root.winfo_height()

        # Camera canvas size (right panel)
        self.cam_w = self.screen_w - 320
        self.cam_h = self.screen_h - 120

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
        # ── Header ─────────────────────────────────────────────
        header = tk.Frame(self.root, bg=ACCENT, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header,
                 text="🎓 Face Attendance System — Strict Mode",
                 font=("Segoe UI", 20, "bold"),
                 bg=ACCENT, fg=BTN_FG).pack(expand=True)

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=15, pady=10)

        # ── Left Panel ─────────────────────────────────────────
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

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(left, variable=self.progress_var,
                                         maximum=100, length=250,
                                         mode="determinate")
        self.progress.pack(pady=4, padx=12)
        self.progress_label = tk.Label(left, text="",
                                        font=("Segoe UI", 9),
                                        bg=PANEL, fg=YELLOW)
        self.progress_label.pack()

        # Separator
        tk.Frame(left, bg=ACCENT, height=1).pack(fill="x", padx=10, pady=10)

        # Today's attendance table
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

        cols = ("Name", "Time")
        self.tree = ttk.Treeview(tree_frame, columns=cols,
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

        # ── Right Panel (Camera) ────────────────────────────────
        right = tk.Frame(content, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self.cam_title = tk.Label(right,
                                   text="📹 Camera Feed",
                                   font=("Segoe UI", 12, "bold"),
                                   bg=BG, fg=TEXT)
        self.cam_title.pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(right,
                                 bg="#0a0a14",
                                 highlightbackground=ACCENT,
                                 highlightthickness=2)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()
        self._show_placeholder()

    def _show_placeholder(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.create_text(
            cw // 2, ch // 2,
            text="📷 Click 'Register Face' or 'Start Attendance'",
            fill="#555577",
            font=("Segoe UI", 15)
        )

    def _btn(self, parent, text, color, cmd):
        tk.Button(parent, text=text,
                  font=("Segoe UI", 10, "bold"),
                  bg=color, fg=BTN_FG,
                  activebackground=color,
                  relief="flat", cursor="hand2",
                  width=28, height=1,
                  command=cmd).pack(pady=5, padx=12)

    def _update_reg_count(self):
        names = get_registered_names()
        self.reg_var.set(f"👤 Registered: {len(names)} person(s)")

    # ── REGISTRATION ────────────────────────────────────────────
    def register_action(self):
        if self.camera_running:
            messagebox.showwarning("Camera Busy",
                                   "Please stop the current camera first.")
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

        self.cam_title.config(
            text=f"📷 Registering: {self.reg_name} — Look at the camera"
        )
        self.status_var.set(f"Registering '{self.reg_name}'... (0/{self.reg_total})")
        self.progress_var.set(0)
        self.progress_label.config(text=f"0 / {self.reg_total} images captured")
        self._update_frame()

    # ── ATTENDANCE ───────────────────────────────────────────────
    def start_attendance(self):
        if self.camera_running:
            messagebox.showwarning("Camera Busy",
                                   "Please stop the current camera first.")
            return

        registered = get_registered_names()
        if not registered:
            messagebox.showwarning("No Faces",
                                   "No registered faces!\nPlease register first.")
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

    # ── STOP ─────────────────────────────────────────────────────
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

    # ── MAIN FRAME LOOP ──────────────────────────────────────────
    def _update_frame(self):
        if not self.camera_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        # Get current canvas size for dynamic scaling
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(80, 80)
        )

        # ── Registration mode ──────────────────────────────────
        if self.camera_mode == "register":
            for (x, y, w, h) in faces:
                if w > 100 and h > 100 and self.reg_count < self.reg_total:
                    face_img = frame[y:y+h, x:x+w]
                    img_path = os.path.join(
                        self.reg_person_dir,
                        f"{self.reg_name}_{self.reg_count}.jpg"
                    )
                    cv2.imwrite(img_path, face_img)
                    self.reg_count += 1

                    pct = (self.reg_count / self.reg_total) * 100
                    self.progress_var.set(pct)
                    self.progress_label.config(
                        text=f"{self.reg_count} / {self.reg_total} images captured"
                    )
                    self.status_var.set(
                        f"Registering '{self.reg_name}'... ({self.reg_count}/{self.reg_total})"
                    )

                color = (0, 255, 0) if w > 100 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.putText(frame,
                        f"Registering: {self.reg_name}  |  {self.reg_count}/{self.reg_total}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
            cv2.putText(frame,
                        "Look straight → left → right → up → down",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            if self.reg_count >= self.reg_total:
                self.stop_camera()
                self.progress_var.set(100)
                self.progress_label.config(text="✅ Registration complete!")
                self.status_var.set(f"'{self.reg_name}' registered ✓")
                messagebox.showinfo("Success",
                                    f"✅ '{self.reg_name}' registered successfully!")
                return

        # ── Attendance mode ────────────────────────────────────
        elif self.camera_mode == "attendance":
            self.frame_count += 1

            if (self.frame_count % 30 == 0 and
                    len(faces) > 0 and not self.is_recognizing):
                self.is_recognizing = True
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w].copy()
                threading.Thread(
                    target=self._bg_recognize,
                    args=(face_roi, 0),
                    daemon=True
                ).start()

            for idx, (x, y, w, h) in enumerate(faces):
                if idx in self.recognition_result:
                    label, color = self.recognition_result[idx]
                else:
                    label, color = "Detecting...", (255, 200, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.rectangle(frame, (x, y-32), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+6, y-9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            status = "Processing..." if self.is_recognizing else "Watching..."
            cv2.putText(frame,
                        f"Status: {status}  |  Threshold: {MATCH_THRESHOLD}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # ── Render to canvas (dynamic size) ────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((cw, ch), Image.LANCZOS)   # ✅ Fills full canvas
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)

        self.root.after(30, self._update_frame)

    def _bg_recognize(self, face_img, idx):
        label, color = recognize_face(face_img, idx)
        self.recognition_result[idx] = (label, color)
        self.is_recognizing = False
        self.root.after(0, self.refresh_table)

    # ── Records ──────────────────────────────────────────────────
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
                    if len(row) >= 3 and row[1] == today:
                        self.tree.insert("", "end", values=(row[0], row[2]))
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
                            if row[0] == "Name" or row[1] != today]
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