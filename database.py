from pymongo import MongoClient


class MongoDB:

    def __init__(self):

        self.client = MongoClient(
            "mongodb+srv://nguyentuandatntd2k2:Datnguyen37@shop.l7tkp.mongodb.net/faceid"
        )

        self.db = self.client["faceid"]
        self.collection = self.db["students"]

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