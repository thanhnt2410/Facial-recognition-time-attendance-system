import cv2
import numpy as np
from keras_facenet import FaceNet
from numpy.linalg import norm
import time

prev_time = 0
frame_count = 0
last_names = []
last_faces = []

# Load database
known_embeddings = np.load("embeddings.npy")
known_labels = np.load("labels.npy")

# Load FaceNet
embedder = FaceNet()

# Haar detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def recognize(face_embedding, threshold=0.8):
    best_score = -1
    best_label = "Unknown"

    for i, known_embedding in enumerate(known_embeddings):
        score = cosine_similarity(face_embedding, known_embedding)

        if score > best_score:
            best_score = score
            best_label = known_labels[i]

    if best_score < threshold:
        return "Unknown"
    
    return best_label

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    frame_small = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    frame_count += 1

    # ===== Chỉ chạy FaceNet mỗi 3 frame =====
    if frame_count % 3 == 0:
        last_names = []
        last_faces = faces

        for (x, y, w, h) in faces:
            padding = 25
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, frame_small.shape[1])
            y2 = min(y + h + padding, frame_small.shape[0])

            face_crop = frame_small[y1:y2, x1:x2]
            face_resize = cv2.resize(face_crop, (160, 160))
            face_rgb = cv2.cvtColor(face_resize, cv2.COLOR_BGR2RGB)

            embedding = embedder.embeddings([face_rgb])[0]
            name = recognize(embedding)

            last_names.append(name)

    # ===== Vẽ lại mỗi frame (không cần chạy FaceNet) =====
    for i, (x, y, w, h) in enumerate(faces):
        name = "Unknown"

        if i < len(last_names):
            name = last_names[i]

        cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame_small, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # FPS
    cv2.putText(frame_small, f"FPS: {int(fps)}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 2)

    cv2.imshow("Face Recognition", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
