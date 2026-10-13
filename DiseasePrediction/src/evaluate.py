"""
evaluate.py — Model evaluation, graphs, confusion matrices, ROC curves
Adil Khan (12510526) | Bhuvan Pratap Singh (12515658)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
import os

from src.preprocess import run_preprocessing
from src.train_models import run_training

os.makedirs('outputs', exist_ok=True)


def plot_accuracy_comparison(d_results, c_results):
    """Bar chart: model accuracy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('📊 Model Accuracy Comparison', fontsize=16, fontweight='bold', y=1.01)

    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

    for ax, results, title in zip(axes,
                                   [d_results, c_results],
                                   ['Diabetes Dataset', 'Breast Cancer Dataset']):
        names = [r['name'] for r in results]
        accs  = [r['accuracy'] * 100 for r in results]
        best_acc = max(accs)

        bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=1.5, zorder=3)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10, zorder=4)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_ylim(50, 108)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.axhline(y=best_acc, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5, zorder=2)
        ax.text(0.01, best_acc + 0.3, f'Best: {best_acc:.1f}%', color='#e74c3c',
                fontsize=9, transform=ax.get_yaxis_transform())

    plt.tight_layout()
    plt.savefig('outputs/accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Accuracy comparison → outputs/accuracy_comparison.png")


def plot_confusion_matrices(best_d, best_c):
    """Heatmap: confusion matrices for best models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('🎯 Confusion Matrices — Best Models', fontsize=15, fontweight='bold')

    configs = [
        (best_d, axes[0], 'Diabetes', ['Not Diabetic', 'Diabetic'],   'Blues'),
        (best_c, axes[1], 'Cancer',   ['Malignant', 'Benign'],        'Greens'),
    ]
    for result, ax, disease, labels, cmap in configs:
        cm = result['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=2, linecolor='white',
                    annot_kws={'size': 15, 'weight': 'bold'})
        ax.set_title(f'{disease} — {result["name"]}\nAccuracy: {result["accuracy"]*100:.2f}%',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrices → outputs/confusion_matrices.png")


def plot_roc_curves(d_results, c_results, y_test_d, y_test_c):
    """ROC curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('📈 ROC Curves — All Models', fontsize=15, fontweight='bold')
    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c']

    for ax, results, y_test, title in zip(
            axes, [d_results, c_results], [y_test_d, y_test_c],
            ['Diabetes', 'Breast Cancer']):
        for result, color in zip(results, colors):
            if result['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
                auc_val = result['auc'] or roc_auc_score(y_test, result['y_prob'])
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f"{result['name']} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random (AUC=0.500)')
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
        ax.set_title(f'ROC — {title}', fontsize=13, fontweight='bold')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ ROC curves → outputs/roc_curves.png")


def plot_pca(X_train_d, y_train_d, X_train_c, y_train_c):
    """2D PCA projection of both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('🔍 PCA — 2D Dimensionality Reduction', fontsize=15, fontweight='bold')

    datasets = [
        (X_train_d, y_train_d, axes[0], 'Diabetes', ['Not Diabetic', 'Diabetic'], 'bwr'),
        (X_train_c, y_train_c, axes[1], 'Breast Cancer', ['Malignant', 'Benign'], 'RdYlGn'),
    ]
    for X, y, ax, title, labels, cmap in datasets:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        var  = pca.explained_variance_ratio_ * 100

        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap,
                        alpha=0.7, edgecolors='white', linewidth=0.3, s=50)
        ax.set_title(f'{title} — PCA 2D Projection', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({var[0]:.1f}% variance)', fontsize=10)
        ax.set_ylabel(f'PC2 ({var[1]:.1f}% variance)', fontsize=10)
        ax.legend(handles=sc.legend_elements()[0], labels=labels, fontsize=9)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.97, f'Total variance explained: {var.sum():.1f}%',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig('outputs/pca_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ PCA visualization → outputs/pca_visualization.png")


def plot_eda_diabetes(df):
    """EDA plots for diabetes dataset."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('🩺 Diabetes Dataset — EDA Histograms', fontsize=15, fontweight='bold')
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    for i, (feat, col) in enumerate(zip(features, colors)):
        ax = axes[i // 3][i % 3]
        ax.hist(df[feat], bins=25, color=col, alpha=0.85, edgecolor='white')
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    # Last: class distribution
    ax = axes[2][2]
    counts = df['Outcome'].value_counts()
    ax.bar(['Not Diabetic', 'Diabetic'], counts.values, color=['#27ae60', '#e74c3c'], edgecolor='white')
    ax.set_title('Class Distribution', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/diabetes_eda.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=0.5, ax=ax2, annot_kws={'size': 9})
    ax2.set_title('Diabetes — Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/diabetes_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Diabetes EDA → outputs/diabetes_eda.png, diabetes_heatmap.png")


def print_results_table(d_results, c_results, best_d, best_c):
    """Print formatted comparison table."""
    import pandas as pd
    print("\n" + "="*75)
    print("  FINAL MODEL COMPARISON TABLE")
    print("="*75)

    for results, disease, best in [(d_results, "DIABETES", best_d), (c_results, "BREAST CANCER", best_c)]:
        print(f"\n📊 {disease}:")
        rows = []
        for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            rows.append({
                'Model': ('★ ' if r['name'] == best['name'] else '  ') + r['name'],
                'Accuracy': f"{r['accuracy']*100:.2f}%",
                'Precision': f"{r['precision']:.4f}",
                'Recall':    f"{r['recall']:.4f}",
                'F1-Score':  f"{r['f1']:.4f}",
                'AUC':       f"{r['auc']:.4f}" if r['auc'] else "N/A",
            })
        df_out = pd.DataFrame(rows)
        print(df_out.to_string(index=False))


def run_evaluation(training_output=None):
    """Run complete evaluation pipeline."""
    print("\n" + "="*60)
    print("  EVALUATION & VISUALIZATION")
    print("="*60)

    if training_output is None:
        data = run_preprocessing()
        d_results, c_results, best_d, best_c, data = run_training(data)
    else:
        d_results, c_results, best_d, best_c, data = training_output

    X_train_d, X_test_d, y_train_d, y_test_d = data['diabetes'][:4]
    X_train_c, X_test_c, y_train_c, y_test_c = data['cancer'][:4]
    df_d = data['diabetes'][6]

    plot_eda_diabetes(df_d)
    plot_accuracy_comparison(d_results, c_results)
    plot_confusion_matrices(best_d, best_c)
    plot_roc_curves(d_results, c_results, y_test_d, y_test_c)
    plot_pca(X_train_d, y_train_d, X_train_c, y_train_c)
    print_results_table(d_results, c_results, best_d, best_c)

    print("\n✅ All evaluations complete! Check outputs/ folder.")


if __name__ == "__main__":
    run_evaluation()
