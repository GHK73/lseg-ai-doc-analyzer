from bson import ObjectId
from datetime import datetime


class DocumentRepository:
    def __init__(self, db):
        self.collection = db["documents"]
        self.users = db["users"]

    def create_document(self, user_id, doc_data):
        doc_data["user_id"] = ObjectId(user_id)
        doc_data["upload_timestamp"] = datetime.utcnow()

        result = self.collection.insert_one(doc_data)

        # link document to user
        self.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"documents": result.inserted_id}}
        )

        return str(result.inserted_id)

    def get_user_documents(self, user_id):
        docs = self.collection.find({"user_id": ObjectId(user_id)})
        return list(docs)

    def get_document(self, doc_id, user_id):
        return self.collection.find_one({
            "_id": ObjectId(doc_id),
            "user_id": ObjectId(user_id)
        })

    def delete_document(self, doc_id, user_id):
        self.collection.delete_one({
            "_id": ObjectId(doc_id),
            "user_id": ObjectId(user_id)
        })

        self.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$pull": {"documents": ObjectId(doc_id)}}
        )