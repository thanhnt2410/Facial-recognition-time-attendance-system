import os
import cv2
import numpy as np
import onnxruntime as ort
from pymongo import MongoClient
import base64


class EmbeddingGenerator:
    def __init__(self, mongo_uri: str, db_name: str = "faceid", collection_name: str = "students"):
        # =============================
        # MongoDB
        # =============================
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # =============================
        # Load model
        # =============================
        self.session = ort.InferenceSession("w600k_mbf.onnx")
        self.input_name = self.session.get_inputs()[0].name

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # =============================
    # Helper functions
    # =============================
    def _base64_to_cv2(self, b64_str: str):
        """Chuyển base64 string sang ảnh OpenCV (BGR)."""
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def _get_embedding(self, face_img) -> np.ndarray:
        """Trích xuất embedding từ ảnh khuôn mặt đã crop."""
        face = cv2.resize(face_img, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        embedding = self.session.run(None, {self.input_name: face})[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _extract_embeddings_from_images(self, images_b64: list, label: str = "") -> list:
        """Nhận vào list base64 ảnh, detect face và trả về list embeddings (tối đa 5 ảnh)."""
        embeddings = []
        for idx, b64 in enumerate(images_b64[:5]):
            img = self._base64_to_cv2(b64)
            if img is None:
                print(f"  [{label}] Ảnh {idx+1}: Không đọc được, bỏ qua.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            print(f"  [{label}] Ảnh {idx+1}: phát hiện {len(faces)} khuôn mặt.")

            for (x, y, w, h) in faces:
                face_crop = img[y:y+h, x:x+w]
                emb = self._get_embedding(face_crop)
                embeddings.append(emb.tolist())

        return embeddings

    # =============================
    # Hàm chính: tạo embedding
    # =============================
    def generate_embeddings(self, force: bool = False) -> dict:
        """
        Duyệt toàn bộ students trong DB, chỉ tạo embedding nếu:
        - Trường embeddings rỗng / không tồn tại, HOẶC
        - force=True (bắt buộc tạo lại tất cả)

        Trả về dict: {"updated": int, "skipped": int, "failed": int}
        """
        students = list(self.collection.find({}))
        print(f"\n[EmbeddingGenerator] Tổng số sinh viên: {len(students)}")

        updated = 0
        skipped = 0
        failed  = 0

        for student in students:
            name    = student.get("Name", "Unknown")
            id_card = student.get("IDCard", str(student["_id"]))
            images  = student.get("images", [])
            existing_embeddings = student.get("embeddings", [])

            # ⭐ Bỏ qua nếu đã có embedding và không force
            if existing_embeddings and not force:
                print(f"[SKIP] {name} (ID: {id_card}) - đã có {len(existing_embeddings)} embeddings.")
                skipped += 1
                continue

            if not images:
                print(f"[SKIP] {name} (ID: {id_card}) - không có trường 'images'.")
                skipped += 1
                continue

            print(f"[PROCESS] {name} (ID: {id_card}) - {len(images)} ảnh ...")

            embeddings = self._extract_embeddings_from_images(images, label=name)

            if not embeddings:
                print(f"  => Không trích xuất được embedding nào, bỏ qua.\n")
                failed += 1
                continue

            self.collection.update_one(
                {"_id": student["_id"]},
                {"$set": {"embeddings": embeddings}}
            )
            print(f"  => Đã lưu {len(embeddings)} embeddings.\n")
            updated += 1

        result = {"updated": updated, "skipped": skipped, "failed": failed}
        print(f"[EmbeddingGenerator] Hoàn tất! Cập nhật: {updated} | Bỏ qua: {skipped} | Lỗi: {failed}\n")
        return result


# =============================
# Chạy độc lập (test)
# =============================
if __name__ == "__main__":
    MONGO_URI = "mongodb+srv://nguyentuandatntd2k2:Datnguyen37@shop.l7tkp.mongodb.net/faceid"
    gen = EmbeddingGenerator(MONGO_URI)
    gen.generate_embeddings(force=False)
