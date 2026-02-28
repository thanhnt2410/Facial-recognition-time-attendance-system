import os
import cv2
import numpy as np
import onnxruntime as ort
from pymongo import MongoClient
from datetime import datetime

# =============================
# MongoDB
# =============================
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["faces"]

# Xóa DB cũ nếu muốn
collection.delete_many({})

# =============================
# Load model
# =============================
session = ort.InferenceSession("w600k_mbf.onnx")
input_name = session.get_inputs()[0].name

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_embedding(face_img):
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5

    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    embedding = session.run(None, {input_name: face})[0][0]
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# =============================
# Tạo embedding
# =============================
dataset_path = "dataset"

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    person_embeddings = []

    for image_name in os.listdir(person_folder):

        img_path = os.path.join(person_folder, image_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(f"{image_name} - detected faces:", len(faces))

        for (x, y, w, h) in faces:
            face_crop = img[y:y+h, x:x+w]
            embedding = get_embedding(face_crop)

            person_embeddings.append(embedding.tolist())

    if len(person_embeddings) == 0:
        continue

    collection.insert_one({
        "IDCard": person_name,
        "embeddings": person_embeddings,  # ⭐ mảng 2 chiều
        "created_at": datetime.utcnow()
    })

    print(f"Saved {len(person_embeddings)} embeddings for {person_name}")

print("Done!")