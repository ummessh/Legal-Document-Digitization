# 📄 Legal Document Digitization System

<a href="https://universe.roboflow.com/major-a0zsb/documents-dataset-yygxz">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>


 You can try it out [here.](https://legal-document-digitization.streamlit.app/#legal-document-digitizer)
 
 A Document digitization system designed specifically for the legal sector, capable of converting scanned legal documents into searchable, structured digital formats using computer vision, Optical Character Recognition and error correction techniques.

## 🎯 Project Overview

This system automates the digitization of legal documents through a sophisticated pipeline that includes document preprocessing, text region detection, OCR, domain-specific error correction, and named entity recognition. The processed data is stored in a structured database, making it easily accessible and searchable.

### Key Features

- **Intelligent Document Preprocessing**
  - Automatic deskewing and rotation correction
  - Advanced binarization for enhanced text clarity
  - Optimized image enhancement for better OCR results

- **Custom YOLOv8-based Text Region Detection**
  - Trained on legal documents for specific layout understanding
  - Detection of four key elements:
    - Text blocks
    - Tables
    - Stamps
    - Signatures

- **Advanced OCR Pipeline**
  - Region-specific text extraction using YOLO detections.
  - Legal domain-specific error correction

---

## 🚀 Getting Started

### Prerequisites

To run this project, you need the following dependencies:

```bash
# Core Requirements
- Python 3.x
- PyTorch
- YOLOv8
- OpenCV
- PyTesseract
- Streamlit
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ummessh/Legal-Document-Digitization.git
   cd Legal-Document-Digitization
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained YOLOv8 model weights (if available) or train your own model using the provided dataset.

4. Configure the OCR and database settings in the `config.yaml` file.

5. Run the preprocessing and digitization pipeline:
   ```bash
   python main.py --input_path /path/to/documents
   ```

---

## 🛠️ Technical Architecture

The system follows a modular pipeline architecture:

1. **Document Input**: Accepts scanned legal documents in various formats (PDF, JPG, JPEG, etc.).
2. **Preprocessing Module**:
   - Image enhancement
   - Deskewing and rotation correction
   - Binarization for improved OCR accuracy
3. **Text Region Detection**:
   - Custom YOLOv8 model trained on legal documents
   - Region classification and localization (text, tables, stamps, signatures)
4. **OCR Processing**:
   - Region-wise text extraction using PyTesseract
   - Legal domain-specific error correction
5. **Data Structuring**:
   - Named Entity Recognition (NER) for extracting key legal entities
   - Database storage in SQLite
   - Indexing for efficient search and retrieval
6. **Query Interface**:
   - Support for SQL queries
   - Integration with RAG for advanced querying

---

Here’s the updated performance metrics section with the new data you provided, formatted consistently with the README:

---

## 📊 Performance Metrics

### Current YOLOv8 Model Performance

| Class      | Precision (P) | Recall (R) | mAP50  | mAP50-95 |
|------------|---------------|------------|--------|----------|
| **All**    | 0.641         | 0.503      | 0.481  | 0.324    |
| Signature  | 0.429         | 0.370      | 0.305  | 0.171    |
| Stamp      | 0.898         | 0.594      | 0.718  | 0.598    |
| Table      | 0.762         | 0.529      | 0.528  | 0.359    |
| Text       | 0.476         | 0.519      | 0.372  | 0.169    |

### Training Details
- **Epochs**: 255
- **Training Time**: 1.452 hours
- **Model Size**: 22.5MB (optimizer stripped)
- **Hardware**: Tesla T4 GPU (15GB VRAM)
- **Framework**: Ultralytics YOLOv8, Python 3.10.12, PyTorch 2.5.1+cu121
- **Speed**:
  - Preprocess: 0.2ms per image
  - Inference: 5.2ms per image
  - Postprocess: 0.8ms per image

---
## 🔄 Current Status

- **Completed**:
  - Preprocessing pipeline implementation
  - YOLOv8 model training
  - OCR integration with PyTesseract
  - Database schema design
- **In Progress**:
  - Error correction model
  - NER implementation
  - Web interface development using Django
  - Integration with RAG for advanced querying
  - Performance optimization for large-scale document processing

---

## 📝 License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](https://github.com/ummessh/Legal-Document-Digitization/blob/main/LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Roboflow** for dataset annotation tools.
- **IL-TUR Benchmark** for providing domain context for NER in Indian legal documents.
- **YOLOv8** and **PyTesseract** communities for their excellent tools and documentation.

