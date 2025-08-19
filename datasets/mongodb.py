import os
import sys

# Add the src directory to the Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import dotenv
import pymongo

from utils.logger import get_logger

dotenv.load_dotenv()
logger = get_logger(__name__)


class ImageDB:
    def __init__(self, connection_url=os.environ["MONGO_URI"]):
        try:
            self.connection_url = connection_url.split("@")[-1]
            self.connection = pymongo.MongoClient(connection_url)
            self._db = self.connection[os.environ["DB_NAME"]]
            self._collection = self._db[os.environ["COLLECTION_NAME"]]
            self.connection_url_not_split = connection_url
        except pymongo.errors.ConnectionFailure as e:
            logger.info(f"Connection failed: {e}")
            return None
        
    def get_all(self):
        return list(self._collection.find({}))

    def insert_one(self, document):
        self._collection.insert_one(document)

    def insert_many(self, documents):
        self._collection.insert_many(documents)

    def __del__(self):
        if self.connection:
            self.connection.close()
