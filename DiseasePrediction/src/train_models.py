"""
train_models.py — Train all ML models and save best ones
Adil Khan (12510526) | Bhuvan Pratap Singh (12515658)
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from src.preprocess import run_preprocessing


def get_models():
    """Return dictionary of all ML models."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        'KNN':                 KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'Naive Bayes':         GaussianNB(),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42, C=1.0),
    }


def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    """Train model and return comprehensive metrics."""
    print(f"  🔄 Training {name}...", end=" ", flush=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    rep = classification_report(y_test, y_pred, output_dict=True)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"✅  Accuracy: {acc*100:.2f}%  CV: {cv.mean()*100:.2f}±{cv.std()*100:.2f}%"
          + (f"  AUC: {auc:.4f}" if auc else ""))

    return {
        'name': name, 'model': model,
        'accuracy': acc, 'cv_mean': cv.mean(), 'cv_std': cv.std(),
        'auc': auc, 'report': rep, 'cm': cm,
        'precision': rep['weighted avg']['precision'],
        'recall':    rep['weighted avg']['recall'],
        'f1':        rep['weighted avg']['f1-score'],
        'y_pred': y_pred, 'y_prob': y_prob,
    }


def run_training(data=None):
    """Train all models, save best, return results."""
    print("\n" + "="*60)
    print("  TRAINING MACHINE LEARNING MODELS")
    print("="*60)

    if data is None:
        data = run_preprocessing()

    X_train_d, X_test_d, y_train_d, y_test_d, scaler_d, d_feats, df_d = data['diabetes']
    X_train_c, X_test_c, y_train_c, y_test_c, scaler_c, c_feats, df_c = data['cancer']

    # ── Train on Diabetes ──
    print("\n📌 Training on Diabetes Dataset:")
    d_results = []
    for name, model in get_models().items():
        res = train_and_evaluate(model, X_train_d, X_test_d, y_train_d, y_test_d, name)
        d_results.append(res)

    best_d = max(d_results, key=lambda x: x['accuracy'])
    joblib.dump(best_d['model'], 'models/diabetes_model.pkl')
    print(f"\n  🏆 BEST: {best_d['name']} ({best_d['accuracy']*100:.2f}%) → saved as diabetes_model.pkl")

    # ── Train on Cancer ──
    print("\n📌 Training on Breast Cancer Dataset:")
    c_results = []
    for name, model in get_models().items():
        res = train_and_evaluate(model, X_train_c, X_test_c, y_train_c, y_test_c, name)
        c_results.append(res)

    best_c = max(c_results, key=lambda x: x['accuracy'])
    joblib.dump(best_c['model'], 'models/cancer_model.pkl')
    print(f"\n  🏆 BEST: {best_c['name']} ({best_c['accuracy']*100:.2f}%) → saved as cancer_model.pkl")

    return d_results, c_results, best_d, best_c, data


if __name__ == "__main__":
    run_training()
