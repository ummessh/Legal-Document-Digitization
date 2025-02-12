import sqlite3
from utils.config import Config

# Ensure table exists
def initialize_db():
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(Config.TABLE_CREATION_QUERY)
    conn.commit()
    conn.close()

# Store OCR result in database
def store_ocr_result(bbox, raw_text):
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO ocr_results (bbox, text) VALUES (?, ?)", (str(bbox), raw_text))
    conn.commit()
    conn.close()
