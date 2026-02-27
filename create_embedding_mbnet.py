import os
import cv2
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm

# ===== Load ONNX model =====
session = ort.InferenceSession(
    "w600k_mbf.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

def get_embedding(face_img):
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5

    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    embedding = session.run(None, {input_name: face})[0][0]
    return embedding


# ===== Load Haar =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

dataset_path = "dataset"  # thư mục chứa ảnh
embeddings = []
labels = []

for person_name in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face_crop = image[y:y+h, x:x+w]
            embedding = get_embedding(face_crop)

            embeddings.append(embedding)
            labels.append(person_name)

print("Tổng số embeddings:", len(embeddings))

np.save("embeddings.npy", np.array(embeddings))
np.save("labels.npy", np.array(labels))

print("Đã lưu embeddings.npy và labels.npy")