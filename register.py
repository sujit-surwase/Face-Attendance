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
    print(f"[*] Registering '{name}'. Look at camera. Capturing {num_images} images...")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # ✅ Only save large, clear faces (exact match quality)
            if w > 100 and h > 100:
                face_img = frame[y:y+h, x:x+w]
                img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured: {count}/{num_images}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        cv2.putText(frame, "Press Q to quit early",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        cv2.imshow(f"Registering: {name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"[✓] Registered '{name}' with {count} images → '{person_dir}'")
        return True
    else:
        print("[!] No face detected. Try better lighting or move closer.")
        return False