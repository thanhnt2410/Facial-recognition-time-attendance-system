import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

facenet = load_model("facenet_keras.h5")

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32")
    face_img = (face_img - 127.5) / 128.0
    face_img = np.expand_dims(face_img, axis=0)
    return facenet.predict(face_img)[0]

embeddings = []
labels = []

dataset_path = "dataset"

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)
        embedding = get_embedding(img)

        embeddings.append(embedding)
        labels.append(person_name)

np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)

