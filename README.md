# Legal Document Digitization System

An advanced document processing system designed specifically for the legal sector, capable of converting scanned legal documents into searchable, structured digital formats using computer vision and machine learning techniques.

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
  - Region-specific text extraction
  - Legal domain-specific error correction
  - Named Entity Recognition (NER) for structured data extraction

- **Structured Data Storage**
  - SQLite database integration
  - Efficient document retrieval system
  - Full-text search capabilities

## 🚀 Getting Started

### Prerequisites

```bash
# Requirements
- Python 3.x
- PyTorch
- YOLOv8n
- OpenCV
- Django
```
📖 Usage
[Coming Soon]

🛠️ Technical Architecture
The system follows a modular pipeline architecture:

Document Input → Accepts scanned legal documents in various formats

**Preprocessing Module**
Image enhancement
Deskewing
Binarization


**Text Region Detection**

Custom YOLOv8 model trained on legal documents
Region classification and localization


**OCR Processing**

Region-wise text extraction
Legal domain-specific error correction


**Data Structuring**

Named Entity Recognition
Database storage
Indexing for search



**🔄 Current Status**

 Preprocessing pipeline implementation
 YOLOv8 model training (Accuracy metrics available)
 OCR integration
 Error correction model
 NER implementation
 Database schema design
 Web interface development

**📊 Performance Metrics**
Current YOLOv8 Model Performance:

mAP50: 0.496
mAP50-95: 0.368

Class-wise Performance:
```
CopyClass      P     R    mAP50  mAP50-95
All      0.549 0.568  0.496   0.368
Signature 0.381 0.414  0.296   0.160
Stamp    0.932 0.575  0.736   0.591
Table    0.620 0.780  0.706   0.624
Text     0.261 0.504  0.244   0.097
```
🤝 Contributing
[Coming Soon]

📝 License
[Apache 2.0](https://github.com/PhoenixAlpha23/Legal-Document-Digitization/blob/main/LICENSE)

🙏 Acknowledgments
Thank you to Roboflow for Dataset annotation.
We used the Signatures and stamps dataset and custom annotated it for the YOLO model, for it to detect the text regions better.

