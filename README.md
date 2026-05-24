🧠 FinOCR AI — Intelligent BFSI Document Intelligence Platform

**FinOCR AI** is an advanced Streamlit-based AI-powered document intelligence system designed for the Banking, Financial Services, and Insurance (BFSI) domain. It combines OCR, machine learning heuristics, fraud detection, multilingual processing, and financial analytics into a single unified platform.



🚀 Features

### 📄 OCR & Document Intelligence
- Extract text from images and PDFs using **Tesseract OCR**
- Preprocessing pipeline:
  - Grayscale conversion
  - Denoising (OpenCV)
  - Adaptive thresholding
  - Sharpening & contrast enhancement
- Confidence scoring for OCR output
- Word and line-level analytics

---

### 🧾 Document Classification
Automatically classifies BFSI documents into:
- Bank Statements
- Invoices
- Payslips
- Loan Documents
- Insurance Documents
- Tax Documents

Uses keyword-based intelligent scoring.

---

### 🌍 Multilingual OCR
Supports multiple languages:
- English
- Hindi + English
- Marathi + English
- Tamil + English
- Telugu + English
- Arabic + English

---

### 🤖 AI Assistant
A built-in intelligent assistant that can:
- Summarize extracted documents
- Explain financial content
- Detect fraud signals
- Guide OCR workflows
- Assist with loan and credit understanding

---

### 💰 Financial Insights Dashboard
- Income vs expense visualization
- Transaction trend analysis
- Risk event tracking
- Extracted monetary value analytics
- Interactive Plotly charts

---

### 🛡 Fraud Detection Engine
Rule-based fraud scoring system:
- Low OCR confidence detection
- Suspicious language detection
- Missing banking identifiers
- Unusual transaction patterns
- Round-number anomaly detection

Outputs a **0–100 risk score**.

---

### 🏦 Loan Recommendation System
Evaluates loan eligibility based on:
- CIBIL score
- Income & obligations
- Debt-to-income ratio
- Applicant type (Student, MSME, Salaried)

Provides:
- Eligibility score
- EMI capacity estimate
- Loan product recommendations

---

### 📊 PDF Analysis Engine
- PDF page rendering (PyMuPDF)
- Embedded text extraction (pypdf)
- OCR fallback for scanned PDFs
- Page-wise document classification

---

### 🎥 Real-time OCR
- Webcam-based document scanning
- Voice note capture (optional speech-to-text)
- Field verification support

---

### 👤 Profile & Access System
- Login / Signup UI
- Workspace profile management
- User preferences (dark mode, notifications)

---

## 🏗 Tech Stack

- **Frontend/UI:** Streamlit
- **OCR Engine:** Tesseract OCR
- **Image Processing:** OpenCV, PIL
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Plotly
- **PDF Processing:** PyMuPDF, pypdf
- **Speech Recognition (optional):** SpeechRecognition
- **Language:** Python 3.9+

