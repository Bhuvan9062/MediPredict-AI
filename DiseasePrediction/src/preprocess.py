"""
preprocess.py — Data loading and preprocessing
Adil Khan (12510526) | Bhuvan Pratap Singh (12515658)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── Ensure output directories exist ──────────────────────────
os.makedirs('outputs', exist_ok=True)
os.makedirs('models',  exist_ok=True)
os.makedirs('data',    exist_ok=True)


def load_diabetes_data():
    """Load Pima Indians Diabetes Dataset."""
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.data.csv")
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    try:
        df = pd.read_csv(url, names=cols)
        print("✅ Diabetes dataset loaded from URL.")
    except Exception as e:
        print(f"⚠️  URL failed ({e}). Generating synthetic diabetes data...")
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            'Pregnancies': np.random.randint(0, 17, n),
            'Glucose': np.clip(np.random.normal(120, 32, n), 50, 200).astype(int),
            'BloodPressure': np.clip(np.random.normal(69, 19, n), 24, 130).astype(int),
            'SkinThickness': np.clip(np.random.normal(20, 16, n), 0, 99).astype(int),
            'Insulin': np.clip(np.random.normal(79, 115, n), 0, 846).astype(int),
            'BMI': np.clip(np.random.normal(32, 8, n), 0, 67.1).round(1),
            'DiabetesPedigreeFunction': np.clip(np.random.normal(0.47, 0.33, n), 0.08, 2.42).round(3),
            'Age': np.clip(np.random.normal(33, 12, n), 21, 81).astype(int),
            'Outcome': np.random.choice([0, 1], n, p=[0.65, 0.35])
        })
    return df


def preprocess_diabetes(df):
    """Clean and split diabetes data."""
    print("\n🔧 Preprocessing Diabetes Dataset...")

    # Replace zero values (biologically impossible) with median
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        n_zeros = (df[col] == 0).sum()
        median_val = df[col].replace(0, np.nan).median()
        df[col] = df[col].replace(0, median_val)
        print(f"   {col}: {n_zeros} zeros → median ({median_val:.2f})")

    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save
    joblib.dump(scaler, 'models/diabetes_scaler.pkl')
    df.to_csv('data/diabetes_clean.csv', index=False)

    print(f"   Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, df.columns.tolist()[:-1]


def load_cancer_data():
    """Load Wisconsin Breast Cancer Dataset."""
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    print("✅ Breast Cancer dataset loaded from sklearn.")
    return df, cancer.feature_names.tolist()


def preprocess_cancer(df, feature_names):
    """Scale and split breast cancer data."""
    print("\n🔧 Preprocessing Breast Cancer Dataset...")

    X = df.drop('target', axis=1).values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    joblib.dump(scaler, 'models/cancer_scaler.pkl')
    df.to_csv('data/cancer_clean.csv', index=False)

    print(f"   Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


def run_preprocessing():
    """Run full preprocessing pipeline and return data splits."""
    print("\n" + "="*60)
    print("  DATA LOADING & PREPROCESSING")
    print("="*60)

    df_d = load_diabetes_data()
    X_train_d, X_test_d, y_train_d, y_test_d, scaler_d, d_features = preprocess_diabetes(df_d)

    df_c, c_features = load_cancer_data()
    X_train_c, X_test_c, y_train_c, y_test_c, scaler_c = preprocess_cancer(df_c, c_features)

    print("\n✅ Preprocessing complete!")
    return {
        'diabetes': (X_train_d, X_test_d, y_train_d, y_test_d, scaler_d, d_features, df_d),
        'cancer':   (X_train_c, X_test_c, y_train_c, y_test_c, scaler_c, c_features, df_c),
    }


if __name__ == "__main__":
    run_preprocessing()
