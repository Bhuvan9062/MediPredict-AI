"""
predict.py — Interactive prediction system (CLI)
Adil Khan (12510526) | Bhuvan Pratap Singh (12515658)
"""

import numpy as np
import joblib
import os


def predict_diabetes(values: list) -> dict:
    """
    Predict diabetes from 8 input values.
    values = [Pregnancies, Glucose, BloodPressure, SkinThickness,
              Insulin, BMI, DiabetesPedigreeFunction, Age]
    """
    model  = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/diabetes_scaler.pkl')

    X = np.array([values])
    X_sc = scaler.transform(X)
    pred = model.predict(X_sc)[0]
    prob = model.predict_proba(X_sc)[0]

    return {
        'prediction': int(pred),
        'label': 'DIABETIC' if pred == 1 else 'NOT DIABETIC',
        'prob_not_diabetic': float(prob[0]),
        'prob_diabetic': float(prob[1]),
        'risk_level': 'HIGH' if pred == 1 else 'LOW',
    }


def predict_cancer(values: list) -> dict:
    """
    Predict breast cancer from 30 feature values.
    Returns: 0 = Malignant, 1 = Benign
    """
    model  = joblib.load('models/cancer_model.pkl')
    scaler = joblib.load('models/cancer_scaler.pkl')

    X = np.array([values])
    X_sc = scaler.transform(X)
    pred = model.predict(X_sc)[0]
    prob = model.predict_proba(X_sc)[0]

    return {
        'prediction': int(pred),
        'label': 'BENIGN' if pred == 1 else 'MALIGNANT',
        'prob_malignant': float(prob[0]),
        'prob_benign': float(prob[1]),
        'risk_level': 'HIGH (Malignant)' if pred == 0 else 'LOW (Benign)',
    }


def interactive_diabetes_prediction():
    """CLI: collect diabetes inputs from user."""
    print("\n" + "─"*50)
    print("  🩺 DIABETES PREDICTION")
    print("─"*50)
    fields = [
        ("Pregnancies (0-20)", 0, 20),
        ("Glucose (mg/dL, 50-220)", 50, 220),
        ("Blood Pressure (mm Hg, 20-130)", 20, 130),
        ("Skin Thickness (mm, 0-99)", 0, 99),
        ("Insulin (μU/ml, 0-850)", 0, 850),
        ("BMI (10-70)", 10.0, 70.0),
        ("Diabetes Pedigree Function (0.08-2.5)", 0.08, 2.5),
        ("Age (18-100)", 18, 100),
    ]
    values = []
    for label, lo, hi in fields:
        while True:
            try:
                val = float(input(f"  Enter {label}: "))
                if lo <= val <= hi:
                    values.append(val)
                    break
                else:
                    print(f"    ⚠️  Please enter a value between {lo} and {hi}")
            except ValueError:
                print("    ⚠️  Please enter a valid number")

    result = predict_diabetes(values)
    print("\n" + "="*50)
    print(f"  🎯 RESULT: {result['label']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Probability — Not Diabetic: {result['prob_not_diabetic']*100:.1f}%")
    print(f"  Probability — Diabetic:     {result['prob_diabetic']*100:.1f}%")
    print("="*50)
    print("  ⚠️  Consult a medical professional for diagnosis.")
    return result


def run_prediction_demo():
    """Run automated demo predictions without user input."""
    print("\n" + "="*60)
    print("  PREDICTION SYSTEM DEMO")
    print("="*60)

    if not os.path.exists('models/diabetes_model.pkl'):
        print("⚠️  Models not found. Please run training first.")
        return

    # ── Demo: Diabetes ──
    print("\n🩺 Diabetes Demo Predictions:")
    samples_d = [
        [6, 148, 72, 35, 0, 33.6, 0.627, 50],   # High risk
        [1,  85, 66, 29, 0, 26.6, 0.351, 31],   # Low risk
        [2, 197, 70, 45, 543, 30.5, 0.158, 53], # High risk
    ]
    d_labels = ["Sample 1 (High Risk)", "Sample 2 (Low Risk)", "Sample 3 (High Risk)"]

    for label, sample in zip(d_labels, samples_d):
        result = predict_diabetes(sample)
        icon = "🔴" if result['prediction'] == 1 else "🟢"
        print(f"  {icon} {label}: {result['label']} "
              f"({result['prob_diabetic']*100:.1f}% diabetic)")

    # ── Demo: Cancer ──
    print("\n🔬 Breast Cancer Demo Predictions:")
    from sklearn.datasets import load_breast_cancer
    cancer_data = load_breast_cancer()

    for idx, label in [(0, "Sample 1 (Malignant)"), (100, "Sample 2 (Benign)"), (200, "Sample 3 (Benign)")]:
        sample = list(cancer_data.data[idx])
        result = predict_cancer(sample)
        icon = "🔴" if result['prediction'] == 0 else "🟢"
        print(f"  {icon} {label}: {result['label']} "
              f"({result['prob_benign']*100:.1f}% benign)")

    print("\n✅ Prediction demo complete.")
    print("   Run 'streamlit run app/streamlit_app.py' for web UI.")


if __name__ == "__main__":
    # Interactive prediction
    choice = input("Predict: (1) Diabetes  (2) Cancer  (3) Demo All\nChoice: ").strip()
    if choice == "1":
        interactive_diabetes_prediction()
    elif choice == "2":
        print("For cancer prediction, please use the Streamlit web app.")
    else:
        run_prediction_demo()
