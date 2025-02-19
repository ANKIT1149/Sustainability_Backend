import os
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS

dotenv_path = "backend/.env"
load_dotenv(dotenv_path=dotenv_path)


MONGO_URI = os.getenv("MONGO_DB_URI")
print(MONGO_URI)

client = MongoClient(MONGO_URI)
db = client["Sustainability"]
print("database connected", db.name)
fs = GridFS(db)
collections = db["predictions"]
user_collections = db['users']
report_collections = db["reports"]
