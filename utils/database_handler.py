import sqlite3
from utils.config import Config
import logging  # Import logging

class DatabaseHandler:
    def __init__(self):
        try:
            self.conn = sqlite3.connect(Config.DB_PATH)
            self.cursor = self.conn.cursor()
            self.initialize_db()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")  # Log the error
            self.conn = None  # Set connection to None if it fails
            self.cursor = None

    def initialize_db(self):
        if self.cursor is None: # Check if connection was successful
            return
        try:
            self.cursor.execute(Config.TABLE_CREATION_QUERY)
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error initializing database: {e}")


    def store_ocr_result(self, bbox, raw_text, corrected_text=None): # Added corrected_text
        """Store OCR result in the database. Includes corrected text."""
        if self.cursor is None: # Check if connection was successful
            return
        try:
            if corrected_text is not None:
                self.cursor.execute("INSERT INTO ocr_results (bbox, raw_text, corrected_text) VALUES (?, ?, ?)", (str(bbox), raw_text, corrected_text))
            else:
                self.cursor.execute("INSERT INTO ocr_results (bbox, raw_text) VALUES (?, ?)", (str(bbox), raw_text)) # Handle case when corrected text is not available
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error storing OCR result: {e}")

    def fetch_ocr_results(self):
        if self.cursor is None: # Check if connection was successful
            return [] # Return empty list if no connection
        try:
            self.cursor.execute("SELECT * FROM ocr_results")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error fetching OCR results: {e}")
            return []

    def close_connection(self):
        if self.conn: # Check if connection exists
            self.conn.close()
