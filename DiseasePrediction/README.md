# 🏥 Dual Disease Prediction System
### Machine Learning CA2 Project | Lovely Professional University

---

## 👥 Team Details

| Field | Details |
|---|---|
| **Student 1** | Adil Khan |
| **Reg. No.** | 12510526 |
| **Student 2** | Bhuvan Pratap Singh |
| **Reg. No.** | 12515658 |
| **Subject** | Machine Learning |
| **Assessment** | CA2 Project |
| **University** | Lovely Professional University, Jalandhar, Punjab |
| **Guide** | Manpreet Kaur Mam, Assistant Professor, School of Computer Application |

---

## 📌 Project Overview

An AI-powered **Dual Disease Prediction System** that uses **6 Machine Learning algorithms** to predict:
1. **Diabetes** (Diabetic / Not Diabetic)
2. **Breast Cancer** (Malignant / Benign)

Best model is automatically selected for each disease and deployed in a real-time Streamlit web app.

---

## 📁 Folder Structure

```
DualDiseasePrediction/
│
├── notebooks/
│   └── Dual_Disease_Prediction_Colab.py    ← Google Colab code (copy to .ipynb)
│
├── src/
│   ├── main.py          ← Entry point (VS Code)
│   ├── preprocess.py    ← Data loading & cleaning
│   ├── train_models.py  ← Train all 6 ML models
│   ├── evaluate.py      ← Graphs, confusion matrix, ROC
│   └── predict.py       ← Prediction system
│
├── app/
│   └── streamlit_app.py ← Web application
│
├── data/                ← Auto-created: cleaned CSVs
├── models/              ← Auto-created: saved .pkl models
├── outputs/             ← Auto-created: all graphs
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for submission)
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy code from `notebooks/Dual_Disease_Prediction_Colab.py` cell by cell
4. Run all cells — outputs appear automatically

### Option 2: VS Code (Local)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python src/main.py

# 3. Or run specific modules
python src/preprocess.py
python src/train_models.py
python src/evaluate.py
python src/predict.py
```

### Option 3: Streamlit Web App
```bash
# Install dependencies first
pip install -r requirements.txt

# Launch web app
streamlit run app/streamlit_app.py

# Open browser at http://localhost:8501
```

---

## 📦 Dependencies

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
streamlit>=1.20
joblib>=1.1
```

Install all: `pip install -r requirements.txt`

---

## 📊 Datasets

| Dataset | Source | Samples | Features |
|---|---|---|---|
| Pima Indians Diabetes | UCI / NIDDK | 768 | 8 |
| Wisconsin Breast Cancer | UCI ML Repo | 569 | 30 |

Both datasets are loaded automatically — no manual download needed.

---

## 🤖 ML Models Used

| Model | Type |
|---|---|
| Logistic Regression | Linear Classifier |
| Decision Tree | Tree-Based |
| Random Forest | Ensemble (Bagging) |
| K-Nearest Neighbors | Instance-Based |
| Naive Bayes | Probabilistic |
| Support Vector Machine | Kernel-Based |

---

## 📈 Results (Typical)

| Model | Diabetes Accuracy | Cancer Accuracy |
|---|---|---|
| Logistic Regression | ~77% | ~95% |
| Decision Tree | ~74% | ~94% |
| **Random Forest** | **~78%** | **~97%** |
| KNN | ~74% | ~95% |
| Naive Bayes | ~75% | ~93% |
| SVM | ~77% | ~97% |

---

## 💡 Project Novelty

- **Dual disease system** in one unified platform
- **6-algorithm comparison** under identical conditions  
- **Auto best-model selection** by accuracy
- **Real-time web predictions** via Streamlit
- **PCA visualization** for dimensionality reduction
- **Comprehensive EDA** with 8+ graph types

---

## ⚠️ Disclaimer

This project is developed for **educational/academic purposes only**.  
Predictions must **never substitute professional medical diagnosis**.

---

*Lovely Professional University, Jalandhar, Punjab*
