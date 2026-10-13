# CELL 1: Install & Import Libraries
# !pip install scikit-learn pandas numpy matplotlib seaborn streamlit  # Uncomment in Colab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import joblib
import os

print("✅ All libraries imported successfully!")
print(f"   Scikit-learn version: {__import__('sklearn').__version__}")
print(f"   Pandas version: {pd.__version__}")
print(f"   NumPy version: {np.__version__}")


# ─────────────────────────────────────────────
# CELL 2: Load Datasets
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  LOADING DATASETS")
print("="*60)

# ── DIABETES DATASET ──
# Source: Pima Indians Diabetes Dataset (from sklearn-compatible CSV)
# Features: 8 medical measurements + 1 target
diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
diabetes_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df_diabetes = pd.read_csv(diabetes_url, names=diabetes_columns)
print(f"\n📊 Diabetes Dataset loaded!")
print(f"   Shape: {df_diabetes.shape}")
print(f"   Target: 0 = Not Diabetic, 1 = Diabetic")
print(f"   Class Distribution:\n{df_diabetes['Outcome'].value_counts().to_string()}")

# ── BREAST CANCER DATASET ──
# Source: sklearn built-in Wisconsin Breast Cancer Dataset
cancer_data = load_breast_cancer()
df_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df_cancer['target'] = cancer_data.target
# target: 0 = Malignant, 1 = Benign
print(f"\n📊 Breast Cancer Dataset loaded!")
print(f"   Shape: {df_cancer.shape}")
print(f"   Target: 0 = Malignant, 1 = Benign")
print(f"   Class Distribution:\n{df_cancer['target'].value_counts().to_string()}")


# ─────────────────────────────────────────────
# CELL 3: Exploratory Data Analysis - DIABETES
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  EDA: DIABETES DATASET")
print("="*60)

print("\n📋 First 5 rows:")
print(df_diabetes.head())
print("\n📋 Data Info:")
print(df_diabetes.info())
print("\n📋 Statistical Summary:")
print(df_diabetes.describe().round(2))
print("\n📋 Missing Values:", df_diabetes.isnull().sum().sum())

# ── EDA PLOTS: DIABETES ──
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('🩺 Diabetes Dataset - EDA Visualizations', fontsize=16, fontweight='bold', y=1.02)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

for i, (feat, col) in enumerate(zip(features, colors)):
    ax = axes[i // 3][i % 3]
    ax.hist(df_diabetes[feat], bins=25, color=col, alpha=0.8, edgecolor='white')
    ax.set_title(f'{feat}', fontsize=11, fontweight='bold')
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Last subplot: Outcome distribution
ax = axes[2][2]
counts = df_diabetes['Outcome'].value_counts()
ax.bar(['Not Diabetic (0)', 'Diabetic (1)'], counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title('Target Distribution', fontsize=11, fontweight='bold')
ax.set_ylabel('Count', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/diabetes_eda_histograms.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Histogram plot saved.")

# ── COUNT PLOT ──
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
colors_count = ['#27ae60', '#c0392b']
bars = ax.bar(['Not Diabetic', 'Diabetic'], df_diabetes['Outcome'].value_counts().values,
              color=colors_count, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, df_diabetes['Outcome'].value_counts().values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val),
            ha='center', fontweight='bold', fontsize=12)
ax.set_title('Diabetes: Class Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/diabetes_count_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Count plot saved.")

# ── BOXPLOT: Glucose by Outcome ──
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Diabetes: Boxplots by Outcome', fontsize=14, fontweight='bold')

key_features = ['Glucose', 'BMI', 'Age']
for i, feat in enumerate(key_features):
    df_diabetes.boxplot(column=feat, by='Outcome', ax=axes[i],
                        boxprops=dict(color='#2980b9'),
                        medianprops=dict(color='#e74c3c', linewidth=2),
                        whiskerprops=dict(color='#2980b9'),
                        capprops=dict(color='#2980b9'))
    axes[i].set_title(f'{feat} by Outcome', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Outcome (0=No Diabetes, 1=Diabetes)', fontsize=9)
    axes[i].set_ylabel(feat, fontsize=9)
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('')
plt.tight_layout()
plt.savefig('outputs/diabetes_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Boxplot saved.")

# ── CORRELATION HEATMAP ──
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_diabetes.corr()
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, mask=mask,
            annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
ax.set_title('Diabetes Dataset - Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/diabetes_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Correlation heatmap saved.")


# ─────────────────────────────────────────────
# CELL 4: EDA - BREAST CANCER
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  EDA: BREAST CANCER DATASET")
print("="*60)

print(f"\n📋 First 5 rows (first 10 features shown):")
print(df_cancer.iloc[:, :10].head())
print(f"\n📋 Shape: {df_cancer.shape}")
print(f"\n📋 Missing Values: {df_cancer.isnull().sum().sum()}")

# ── COUNT PLOT ──
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
bars = ax.bar(['Malignant (0)', 'Benign (1)'], df_cancer['target'].value_counts().sort_index().values,
              color=['#e74c3c', '#27ae60'], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, df_cancer['target'].value_counts().sort_index().values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val),
            ha='center', fontweight='bold', fontsize=12)
ax.set_title('Breast Cancer: Class Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/cancer_count_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# ── HISTOGRAMS FOR TOP FEATURES ──
top_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                'mean smoothness', 'mean compactness', 'mean concavity',
                'mean concave points', 'mean symmetry']

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
fig.suptitle('🔬 Breast Cancer Dataset - Feature Distributions', fontsize=15, fontweight='bold')

palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
           '#1abc9c', '#e67e22', '#34495e', '#d35400']
for i, (feat, col) in enumerate(zip(top_features, palette)):
    ax = axes[i // 3][i % 3]
    for cls, lbl, clr in zip([0, 1], ['Malignant', 'Benign'], ['#e74c3c', '#27ae60']):
        ax.hist(df_cancer[df_cancer['target'] == cls][feat], bins=20,
                alpha=0.65, label=lbl, color=clr, edgecolor='white')
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/cancer_eda_histograms.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Cancer EDA histograms saved.")

# ── CORRELATION HEATMAP ──
fig, ax = plt.subplots(figsize=(14, 12))
top_10_corr = df_cancer[list(cancer_data.feature_names[:15]) + ['target']].corr()
sns.heatmap(top_10_corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.4, ax=ax, annot_kws={'size': 7}, cbar_kws={'shrink': 0.8})
ax.set_title('Breast Cancer - Correlation Heatmap (Top 15 Features)', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('outputs/cancer_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Cancer correlation heatmap saved.")


# ─────────────────────────────────────────────
# CELL 5: Data Preprocessing
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  DATA PREPROCESSING")
print("="*60)

# ── DIABETES ──
print("\n🔧 Preprocessing Diabetes Dataset...")

# Replace zeros with median for medical features (zeros are biologically impossible)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    median_val = df_diabetes[col].median()
    zeros_count = (df_diabetes[col] == 0).sum()
    df_diabetes[col] = df_diabetes[col].replace(0, median_val)
    print(f"   ✓ {col}: Replaced {zeros_count} zeros with median ({median_val:.2f})")

X_diabetes = df_diabetes.drop('Outcome', axis=1).values
y_diabetes = df_diabetes['Outcome'].values

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes)

scaler_d = StandardScaler()
X_train_d_scaled = scaler_d.fit_transform(X_train_d)
X_test_d_scaled  = scaler_d.transform(X_test_d)

print(f"\n   Training set: {X_train_d_scaled.shape}")
print(f"   Test set:     {X_test_d_scaled.shape}")
print(f"   Features scaled using StandardScaler ✅")

# ── BREAST CANCER ──
print("\n🔧 Preprocessing Breast Cancer Dataset...")

X_cancer = cancer_data.data
y_cancer = cancer_data.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled  = scaler_c.transform(X_test_c)

print(f"   Training set: {X_train_c_scaled.shape}")
print(f"   Test set:     {X_test_c_scaled.shape}")
print(f"   Features scaled using StandardScaler ✅")


# ─────────────────────────────────────────────
# CELL 6: PCA - Dimensionality Reduction
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  PCA VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('PCA - 2D Projection of Datasets', fontsize=14, fontweight='bold')

# ── PCA: Diabetes ──
pca_d = PCA(n_components=2, random_state=42)
X_pca_d = pca_d.fit_transform(X_train_d_scaled)

scatter = axes[0].scatter(X_pca_d[:, 0], X_pca_d[:, 1],
                          c=y_train_d, cmap='bwr', alpha=0.7, edgecolors='white', linewidth=0.3, s=50)
axes[0].set_title('Diabetes - PCA 2D Projection', fontsize=12, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_d.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=10)
axes[0].set_ylabel(f'PC2 ({pca_d.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=10)
legend1 = axes[0].legend(*scatter.legend_elements(), labels=['Not Diabetic', 'Diabetic'], fontsize=9)
axes[0].add_artist(legend1)
axes[0].grid(alpha=0.3)
var_explained_d = sum(pca_d.explained_variance_ratio_) * 100
axes[0].text(0.02, 0.98, f'Total Variance Explained: {var_explained_d:.1f}%',
             transform=axes[0].transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ── PCA: Breast Cancer ──
pca_c = PCA(n_components=2, random_state=42)
X_pca_c = pca_c.fit_transform(X_train_c_scaled)

scatter2 = axes[1].scatter(X_pca_c[:, 0], X_pca_c[:, 1],
                           c=y_train_c, cmap='RdYlGn', alpha=0.7, edgecolors='white', linewidth=0.3, s=50)
axes[1].set_title('Breast Cancer - PCA 2D Projection', fontsize=12, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca_c.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=10)
axes[1].set_ylabel(f'PC2 ({pca_c.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=10)
legend2 = axes[1].legend(*scatter2.legend_elements(), labels=['Malignant', 'Benign'], fontsize=9)
axes[1].add_artist(legend2)
axes[1].grid(alpha=0.3)
var_explained_c = sum(pca_c.explained_variance_ratio_) * 100
axes[1].text(0.02, 0.98, f'Total Variance Explained: {var_explained_c:.1f}%',
             transform=axes[1].transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/pca_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ PCA: Diabetes explains {var_explained_d:.1f}% | Cancer explains {var_explained_c:.1f}%")


# ─────────────────────────────────────────────
# CELL 7: Train All Models - DIABETES
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  TRAINING MODELS: DIABETES DATASET")
print("="*60)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train, evaluate and return metrics for a model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  📌 {model_name}")
    print(f"     Accuracy:  {acc*100:.2f}%")
    print(f"     CV Score:  {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    if auc: print(f"     ROC-AUC:   {auc:.4f}")
    print(f"     Precision: {report['weighted avg']['precision']:.4f}")
    print(f"     Recall:    {report['weighted avg']['recall']:.4f}")
    print(f"     F1-Score:  {report['weighted avg']['f1-score']:.4f}")

    return {
        'name': model_name, 'model': model, 'accuracy': acc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'auc': auc, 'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'y_pred': y_pred, 'y_prob': y_prob, 'cm': cm, 'report': report
    }


diabetes_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
}

diabetes_results = []
for name, model in diabetes_models.items():
    result = evaluate_model(model, X_train_d_scaled, X_test_d_scaled,
                            y_train_d, y_test_d, name)
    diabetes_results.append(result)

# Best model
best_d = max(diabetes_results, key=lambda x: x['accuracy'])
print(f"\n🏆 BEST MODEL FOR DIABETES: {best_d['name']} ({best_d['accuracy']*100:.2f}%)")


# ─────────────────────────────────────────────
# CELL 8: Train All Models - BREAST CANCER
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  TRAINING MODELS: BREAST CANCER DATASET")
print("="*60)

cancer_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
}

cancer_results = []
for name, model in cancer_models.items():
    result = evaluate_model(model, X_train_c_scaled, X_test_c_scaled,
                            y_train_c, y_test_c, name)
    cancer_results.append(result)

best_c = max(cancer_results, key=lambda x: x['accuracy'])
print(f"\n🏆 BEST MODEL FOR BREAST CANCER: {best_c['name']} ({best_c['accuracy']*100:.2f}%)")


# ─────────────────────────────────────────────
# CELL 9: Accuracy Comparison Charts
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  ACCURACY COMPARISON GRAPHS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Accuracy Comparison', fontsize=15, fontweight='bold')

bar_colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

for ax, results, title in zip(axes,
                               [diabetes_results, cancer_results],
                               ['Diabetes Dataset', 'Breast Cancer Dataset']):
    names = [r['name'] for r in results]
    accs  = [r['accuracy']*100 for r in results]
    bars  = ax.bar(names, accs, color=bar_colors, edgecolor='white', linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(50, 105)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=max(accs), color='red', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('outputs/accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Accuracy comparison chart saved.")


# ─────────────────────────────────────────────
# CELL 10: Confusion Matrices
# ─────────────────────────────────────────────

def plot_confusion_matrix(result, ax, title, labels):
    cm = result['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'{title}\n{result["name"]} ({result["accuracy"]*100:.1f}%)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrix - Best Models', fontsize=14, fontweight='bold')

plot_confusion_matrix(best_d, axes[0], 'Diabetes', ['Not Diabetic', 'Diabetic'])
plot_confusion_matrix(best_c, axes[1], 'Breast Cancer', ['Malignant', 'Benign'])

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved.")


# ─────────────────────────────────────────────
# CELL 11: ROC Curves
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ROC Curves - All Models', fontsize=14, fontweight='bold')

colors_roc = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

for ax, results, y_test, title in zip(axes,
                                      [diabetes_results, cancer_results],
                                      [y_test_d, y_test_c],
                                      ['Diabetes', 'Breast Cancer']):
    for result, color in zip(results, colors_roc):
        if result['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
            auc_score = result['auc']
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{result['name']} (AUC={auc_score:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_title(f'ROC Curve - {title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ROC curves saved.")


# ─────────────────────────────────────────────
# CELL 12: Comprehensive Results Table
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)

print("\n📊 DIABETES - All Models Performance:")
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'AUC':>8}")
print("-" * 72)
for r in sorted(diabetes_results, key=lambda x: x['accuracy'], reverse=True):
    star = " ★ BEST" if r['name'] == best_d['name'] else ""
    auc_str = f"{r['auc']:.4f}" if r['auc'] else "  N/A  "
    print(f"{r['name']:<22} {r['accuracy']*100:>9.2f}% {r['precision']:>10.4f} "
          f"{r['recall']:>8.4f} {r['f1']:>10.4f} {auc_str:>8}{star}")

print("\n📊 BREAST CANCER - All Models Performance:")
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'AUC':>8}")
print("-" * 72)
for r in sorted(cancer_results, key=lambda x: x['accuracy'], reverse=True):
    star = " ★ BEST" if r['name'] == best_c['name'] else ""
    auc_str = f"{r['auc']:.4f}" if r['auc'] else "  N/A  "
    print(f"{r['name']:<22} {r['accuracy']*100:>9.2f}% {r['precision']:>10.4f} "
          f"{r['recall']:>8.4f} {r['f1']:>10.4f} {auc_str:>8}{star}")


# ─────────────────────────────────────────────
# CELL 13: Save Best Models
# ─────────────────────────────────────────────

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Save best models and scalers
joblib.dump(best_d['model'], 'models/diabetes_model.pkl')
joblib.dump(best_c['model'], 'models/cancer_model.pkl')
joblib.dump(scaler_d,        'models/diabetes_scaler.pkl')
joblib.dump(scaler_c,        'models/cancer_scaler.pkl')

print("\n✅ Models saved:")
print(f"   models/diabetes_model.pkl  → {best_d['name']}")
print(f"   models/cancer_model.pkl    → {best_c['name']}")
print(f"   models/diabetes_scaler.pkl → StandardScaler")
print(f"   models/cancer_scaler.pkl   → StandardScaler")


# ─────────────────────────────────────────────
# CELL 14: Command-Line Prediction System
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  PREDICTION SYSTEM DEMO")
print("="*60)

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, diabetes_pedigree, age):
    """Predict diabetes for given inputs."""
    model = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/diabetes_scaler.pkl')
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    label = "🔴 DIABETIC" if prediction == 1 else "🟢 NOT DIABETIC"
    return prediction, probability, label

def predict_cancer(features_30):
    """Predict breast cancer for given 30 features."""
    model = joblib.load('models/cancer_model.pkl')
    scaler = joblib.load('models/cancer_scaler.pkl')
    input_data = np.array([features_30])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    label = "🔴 MALIGNANT" if prediction == 0 else "🟢 BENIGN"
    return prediction, probability, label

# Demo predictions
print("\n🧪 Sample Diabetes Prediction:")
pred, prob, label = predict_diabetes(
    pregnancies=6, glucose=148, blood_pressure=72, skin_thickness=35,
    insulin=0, bmi=33.6, diabetes_pedigree=0.627, age=50)
print(f"   Input: Glucose=148, BMI=33.6, Age=50")
print(f"   Result: {label}")
print(f"   Confidence: Not Diabetic={prob[0]*100:.1f}% | Diabetic={prob[1]*100:.1f}%")

print("\n🧪 Sample Cancer Prediction:")
sample_cancer = list(cancer_data.data[0])  # First sample from dataset
pred_c, prob_c, label_c = predict_cancer(sample_cancer)
print(f"   Input: Sample from dataset (30 features)")
print(f"   Result: {label_c}")
print(f"   Confidence: Malignant={prob_c[0]*100:.1f}% | Benign={prob_c[1]*100:.1f}%")

print("\n" + "="*60)
print("  ✅ PROJECT COMPLETE! All outputs saved in 'outputs/' folder.")
print("="*60)
