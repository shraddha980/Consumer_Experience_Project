import pymongo
from dataclasses import dataclass
import os

@dataclass
class EnvironmentVariable:
    mongo_db_url:str = os.getenv("mongodb+srv://nlpprojects809:Crt9VnDR6LTzjhqm@cluster0.wuw5etg.mongodb.net/?retryWrites=true&w=majority")


env_var = EnvironmentVariable()
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
TARGET_COLUMN = "sentiment"
print(env_var.mongo_db_url)

