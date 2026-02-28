from pymongo import MongoClient

# ===== Kết nối MongoDB =====
client = MongoClient("mongodb://localhost:27017/")

db = client["face_recognition"]      # Tên database
collection = db["faces"]          # Tên collection

# ===== Danh sách cần update =====
updates = [
    {"IDCard": "SV001", "Name": "Nguyễn Tuấn Đạt"},
    {"IDCard": "SV002", "Name": "Nguyễn Tuấn Thành"}
]

# ===== Thực hiện update =====
for item in updates:
    result = collection.update_one(
        {"IDCard": item["IDCard"]},
        {"$set": {"Name": item["Name"]}}
    )

    if result.matched_count > 0:
        print(f"Đã cập nhật {item['IDCard']} thành công.")
    else:
        print(f"Không tìm thấy {item['IDCard']} trong database.")

print("Hoàn thành cập nhật.")