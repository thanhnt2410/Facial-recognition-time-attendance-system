import cv2
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm
from pymongo import MongoClient


class FaceEngine:
    def __init__(self):

        # ===== Load ONNX model =====
        self.session = ort.InferenceSession(
            "w600k_mbf.onnx",
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # ===== Connect MongoDB =====
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["face_recognition"]
        self.collection = self.db["faces"]

        # ===== Load embeddings từ DB =====
        self.known_embeddings = []
        self.known_labels = []

        self.load_embeddings_from_db()

        # Convert sang numpy array để tăng tốc
        if len(self.known_embeddings) > 0:
            self.known_embeddings = np.array(self.known_embeddings)

        # ===== Haar detector =====
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        self.frame_count = 0
        self.last_names = []
        self.last_faces = []

    # ============================
    def load_embeddings_from_db(self):

        self.known_embeddings = []
        self.known_labels = []

        for doc in self.collection.find():

            name = doc["Name"]

            # embeddings dạng 2D (ví dụ 5x512)
            embeddings_2d = np.array(doc["embeddings"], dtype=np.float32)

            for emb in embeddings_2d:
                self.known_embeddings.append(emb)
                self.known_labels.append(name)

        if len(self.known_embeddings) > 0:
            self.known_embeddings = np.array(self.known_embeddings)

        print("Loaded total embeddings:", len(self.known_embeddings))

    # ============================
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    # ============================
    def recognize(self, embedding, threshold=0.6):

        if len(self.known_embeddings) == 0:
            return "Unknown"

        # cosine similarity vector hóa
        similarities = np.dot(self.known_embeddings, embedding)

        best_index = np.argmax(similarities)
        best_score = similarities[best_index]

        print("Best score:", best_score)

        if best_score < threshold:
            return "Unknown"

        return self.known_labels[best_index]

    # ============================
    def get_embedding(self, face_img):

        face = cv2.resize(face_img, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5  # normalize [-1,1]

        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        embedding = self.session.run(
            None, {self.input_name: face}
        )[0][0]

        # Normalize embedding (QUAN TRỌNG)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    # ============================
    def process_frame(self, frame):

        frame_small = cv2.resize(frame, (480, 360))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        self.frame_count += 1

        if self.frame_count % 1 == 0:
            self.last_names = []
            self.last_faces = faces

            for (x, y, w, h) in faces:

                padding = 20
                x1 = max(x - padding, 0)
                y1 = max(y - padding, 0)
                x2 = min(x + w + padding, frame_small.shape[1])
                y2 = min(y + h + padding, frame_small.shape[0])

                face_crop = frame_small[y1:y2, x1:x2]

                embedding = self.get_embedding(face_crop)
                name = self.recognize(embedding)

                self.last_names.append(name)

        # Draw bounding box
        for i, (x, y, w, h) in enumerate(faces):

            name = "Unknown"
            if i < len(self.last_names):
                name = self.last_names[i]

            cv2.rectangle(
                frame_small,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

        return frame_small, (
            self.last_names[0] if self.last_names else ""
        )