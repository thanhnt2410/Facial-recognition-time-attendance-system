from pymongo import MongoClient
from datetime import datetime


class MongoDB:

    def __init__(self):

        self.client = MongoClient(
            "mongodb+srv://nguyentuandatntd2k2:Datnguyen37@shop.l7tkp.mongodb.net/faceid"
        )

        self.db = self.client["faceid"]
        self.collection = self.db["students"]
        self.attendances = self.db["attendances"]
        # chống ghi nhiều lần liên tiếp
        self.last_attendance = {}

    def get_all_embeddings(self):

        return list(self.collection.find())

    def get_user_by_idcard(self, idcard):

        return self.collection.find_one({"IDCard": idcard})
    def load_users(self):

        users = {}

        for u in self.collection.find({}, {"Name":1, "IDCard":1, "RFID":1}):

            users[u["IDCard"]] = {
                "Name": u["Name"],
                "RFID": u["RFID"]
            }

        return users
    def save_attendance(self, idcard):

        now = datetime.utcnow()

        # chống spam (10 giây)
        if idcard in self.last_attendance:

            diff = (now - self.last_attendance[idcard]).total_seconds()

            if diff < 10:
                return

        self.last_attendance[idcard] = now

        attendance = {
            "IDCard": idcard,
            "timestamps": now
        }

        self.attendances.insert_one(attendance)

        print("Attendance saved:", attendance)