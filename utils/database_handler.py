import sqlite3
from utils.config import Config

class DatabaseHandler:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH)
        self.cursor = self.conn.cursor()
        self.initialize_db()

    def initialize_db(self):
        """Ensure the OCR results table exists."""
        self.cursor.execute(Config.TABLE_CREATION_QUERY)
        self.conn.commit()

    def store_ocr_result(self, bbox, raw_text):
        """Store OCR result (bounding box and extracted text) in the database."""
        self.cursor.execute("INSERT INTO ocr_results (bbox, text) VALUES (?, ?)", (str(bbox), raw_text))
        self.conn.commit()

    def fetch_ocr_results(self):
        """Retrieve all OCR results from the database."""
        self.cursor.execute("SELECT * FROM ocr_results")
        return self.cursor.fetchall()

    def close_connection(self):
        """Close the database connection."""
        self.conn.close()
