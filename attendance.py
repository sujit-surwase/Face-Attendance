import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
from deepface import DeepFace
from utils import DATASET_DIR, mark_attendance, get_registered_names

# ✅ Matching threshold — lower = stricter
MATCH_THRESHOLD = 0.40

def recognize_face(face_img, idx):
    """
    Called from main.py background thread.
    Takes a face crop image, runs DeepFace, returns (label, color).
    """
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

        if result and len(result[0]) > 0:
            top_match = result[0].iloc[0]
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
    """
    Fallback CLI mode — only used if running attendance.py directly.
    Normal usage: camera is embedded in the GUI via main.py
    """
    import threading

    registered = get_registered_names()
    if not registered:
        print("[!] No registered faces found. Please register first.")
        return

    print(f"[*] Loaded {len(registered)} person(s): {registered}")
    print("[*] Starting webcam... Press 'Q' to stop.")

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
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w].copy()
            threading.Thread(
                target=_bg_recognize,
                args=(face_roi, 0),
                daemon=True
            ).start()

        for idx, (x, y, w, h) in enumerate(faces):
            if idx in recognition_result:
                label, color = recognition_result[idx]
            else:
                label, color = "Detecting...", (255, 200, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-28), (x+w, y), color, -1)
            cv2.putText(frame, label, (x+4, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        status_text = "Processing..." if is_recognizing else "Watching..."
        cv2.putText(frame, f"Status: {status_text}  |  Threshold: {MATCH_THRESHOLD}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Press Q to Quit",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Face Attendance System", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Attendance session ended.")


if __name__ == "__main__":
    run_attendance()