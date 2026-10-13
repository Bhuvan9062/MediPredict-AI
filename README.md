# 🏥 Dual Disease Prediction System | MediPredict-AI

An AI/ML-based project that predicts two health conditions — **Diabetes and Breast Cancer** — using six machine learning classification algorithms.

## 📌 Project Overview

MediPredict-AI is a machine learning project developed to demonstrate how classification algorithms can be used for disease prediction. It compares multiple models, evaluates their performance, and selects the best-performing model for each dataset based on accuracy.

The project includes data preprocessing, exploratory data analysis (EDA), dimensionality reduction using PCA, model evaluation, and prediction demonstrations.

## ✨ Key Features

- Prediction of Diabetes and Breast Cancer
- Comparison of six machine learning algorithms
- Exploratory Data Analysis with visualizations
- Data preprocessing and feature scaling
- PCA-based dimensionality reduction
- Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation
- Confusion matrices and ROC curves
- Best-model selection and model serialization using Joblib
- Sample disease prediction demonstrations

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib
- **Development Environment:** Google Colab / Jupyter Notebook

## 🤖 Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Gaussian Naive Bayes
6. Support Vector Machine (SVM)

## 📊 Datasets

| Dataset | Samples | Features |
|---|---:|---:|
| Pima Indians Diabetes Dataset | 768 | 8 |
| Wisconsin Breast Cancer Dataset | 569 | 30 |

The Diabetes dataset is loaded from a public CSV source, while the Breast Cancer dataset is loaded using Scikit-learn.

## ⚙️ Project Workflow

1. Load the datasets
2. Perform exploratory data analysis
3. Handle zero values in selected Diabetes features using median imputation
4. Split data into training and testing sets
5. Apply StandardScaler
6. Perform PCA for visualization
7. Train and compare six classification algorithms for each disease
8. Evaluate model performance
9. Select and save the best-performing models
10. Run sample disease predictions

## 📁 Repository Structure

```text
MediPredict-AI/
│
├── Dual_Disease_Prediction_Colab.py
├── Final_Report_DualDiseaseML.docx
├── VIVA_QUESTIONS_ANSWERS.md
├── requirements.txt
└── README.md
```

The Python script creates `models/` and `outputs/` directories when the relevant code executes successfully.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/MediPredict-AI.git
cd MediPredict-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

Open `Dual_Disease_Prediction_Colab.py` in Google Colab or Jupyter Notebook and execute the code in sequence.

Alternatively, after installing the dependencies, run the Python script from a terminal:

```bash
python Dual_Disease_Prediction_Colab.py
```

**Note:** Ensure the `outputs` directory exists before executing the plotting sections, because the script saves visualizations there before creating that directory later in the workflow.

## 📈 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Five-fold cross-validation
- Confusion matrices and ROC curves

The script prints the evaluation results and identifies the highest-accuracy model for each disease.

## 📚 Documentation

- **Final_Report_DualDiseaseML.docx:** Detailed project report, methodology, datasets, and analysis.
- **VIVA_QUESTIONS_ANSWERS.md:** Project viva questions and answers for academic preparation.
- **requirements.txt:** Python library dependencies.

## 🔮 Future Improvements

- Develop an interactive Streamlit web application
- Add more disease prediction models
- Improve model interpretability
- Deploy the application for online access
- Evaluate models on additional clinical datasets

## ⚠️ Disclaimer

This project is intended for educational and academic purposes only. Its predictions are experimental and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 👨‍💻 Authors

**Bhuvan Pratap Singh**  
**Adil Khan**

Lovely Professional University, Punjab, India
