import sqlite3
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import logging

class EnhancedOCRProcessor:
    def __init__(self, db_path: str = "ocr_documents.db"):
        self.conn = sqlite3.connect(db_path)
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
        # Enhanced documents table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT UNIQUE,  -- For deduplication
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
        )''')

        # Enhanced text elements table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_elements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            page_number INTEGER,
            content TEXT,
            content_type TEXT,  -- header, paragraph, footer, etc.
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
        )''')

        # Enhanced table elements
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_elements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            page_number INTEGER,
            table_data TEXT,
            table_structure TEXT,  -- JSON format for merged cells info
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
        )''')

        # Enhanced stamps
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS stamps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            page_number INTEGER,
            stamp_type TEXT,
            stamp_text TEXT,  -- Extracted text from stamp
            color TEXT,  -- RGB value
            shape TEXT,  -- circular, rectangular, etc.
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            confidence_score FLOAT,
            image_path TEXT,
            rotation_angle FLOAT,
            is_official BOOLEAN,
            verification_status TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )''')

        # Enhanced signatures
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            page_number INTEGER,
            signature_type TEXT,
            signer_name TEXT,
            signing_date TEXT,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            confidence_score FLOAT,
            image_path TEXT,
            stroke_count INTEGER,
            is_verified BOOLEAN,
            verification_method TEXT,
            metadata TEXT,  -- JSON for additional signature metadata
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )''')

        # Create indices for better query performance
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_file_hash ON documents(file_hash)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_doc_page ON text_elements(document_id, page_number)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_table_doc_page ON table_elements(document_id, page_number)')
        
        self.conn.commit()

    def add_document(self, file_info: Dict) -> int:
        try:
            self.cursor.execute('''
            INSERT INTO documents (
                file_name, file_path, file_hash, mime_type, file_size,
                processed_date, status, total_pages, ocr_engine,
                ocr_version, language, processing_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_info['file_name'], file_info['file_path'],
                file_info.get('file_hash'), file_info.get('mime_type'),
                file_info.get('file_size'), datetime.now(),
                'processing', file_info['total_pages'],
                file_info.get('ocr_engine'), file_info.get('ocr_version'),
                file_info.get('language'), file_info.get('processing_time')
            ))
            doc_id = self.cursor.lastrowid
            self.conn.commit()
            self.logger.info(f"Added document {file_info['file_name']} with ID {doc_id}")
            return doc_id
        except sqlite3.Error as e:
            self.logger.error(f"Error adding document: {e}")
            self.conn.rollback()
            raise

    def update_document_status(self, doc_id: int, status: str, error_message: Optional[str] = None):
        try:
            self.cursor.execute('''
            UPDATE documents
            SET status = ?, error_message = ?, last_modified = ?
            WHERE id = ?
            ''', (status, error_message, datetime.now(), doc_id))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Error updating document status: {e}")
            self.conn.rollback()
            raise

    def get_document_summary(self, doc_id: int) -> Dict:
        """Get a complete summary of all elements in a document."""
        try:
            document = self.cursor.execute(
                'SELECT * FROM documents WHERE id = ?', (doc_id,)
            ).fetchone()
            
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")

            text_elements = self.get_document_text(doc_id)
            tables = self.get_document_tables(doc_id)
            stamps = self.get_document_stamps(doc_id)
            signatures = self.get_document_signatures(doc_id)

            return {
                'document_info': dict(zip([d[0] for d in self.cursor.description], document)),
                'text_elements': text_elements,
                'tables': tables,
                'stamps': stamps,
                'signatures': signatures
            }
        except sqlite3.Error as e:
            self.logger.error(f"Error getting document summary: {e}")
            raise

    def __del__(self):
        self.conn.close()
