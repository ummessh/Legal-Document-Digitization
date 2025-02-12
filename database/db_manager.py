import sqlite3
from datetime import datetime
import logging

class DBManager:
    def __init__(self, db_path: str = "ocr_documents.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.setup_logging()
        self.create_tables()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_tables(self):
        """Creates necessary tables in the database."""
        self.cursor.executescript('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT UNIQUE,
            mime_type TEXT,
            file_size INTEGER,
            processed_date DATETIME,
            last_modified DATETIME,
            status TEXT,
            total_pages INTEGER,
            ocr_engine TEXT,
            ocr_version TEXT,
            language TEXT,
            processing_time FLOAT,
            error_message TEXT
        );
        
        CREATE TABLE IF NOT EXISTS text_elements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            page_number INTEGER,
            content TEXT,
            content_type TEXT,
            language TEXT,
            confidence_score FLOAT,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            font_size FLOAT,
            font_family TEXT,
            is_bold BOOLEAN,
            is_italic BOOLEAN,
            rotation_angle FLOAT,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS table_elements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            page_number INTEGER,
            table_data TEXT,
            table_structure TEXT,
            header_rows INTEGER,
            rows INTEGER,
            columns INTEGER,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            confidence_score FLOAT,
            has_borders BOOLEAN,
            detected_currency BOOLEAN,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_doc_file_hash ON documents(file_hash);
        CREATE INDEX IF NOT EXISTS idx_text_doc_page ON text_elements(document_id, page_number);
        CREATE INDEX IF NOT EXISTS idx_table_doc_page ON table_elements(document_id, page_number);
        ''')
        self.conn.commit()
        self.logger.info("Database tables created successfully.")

    def close_connection(self):
        """Closes the database connection."""
        self.conn.close()

# Singleton instance to avoid multiple connections
DB_INSTANCE = DBManager()
