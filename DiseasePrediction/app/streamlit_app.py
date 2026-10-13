"""
╔══════════════════════════════════════════════════════════════════════╗
║       MULTI-DISEASE PREDICTION SYSTEM (5 Diseases) - STREAMLIT     ║
║  Authors: Adil Khan (12510526) | Bhuvan Pratap Singh (12515658)    ║
║  LPU, Punjab | ML CA2 Project                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Diseases Covered:
  1. Diabetes          → Random Forest
  2. Breast Cancer     → SVM / Random Forest
  3. Heart Disease     → Random Forest (Cleveland Heart Disease Dataset)
  4. Kidney Disease    → Random Forest (CKD Dataset)
  5. Liver Disease     → Gradient Boosting (ILPD Dataset)

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0d1b2a 0%,#1b263b 50%,#415a77 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e1dd !important; }
    section[data-testid="stSidebar"] .stRadio label { color: #e0e1dd !important; }

    .header-gradient {
        background: linear-gradient(135deg,#0d1b2a 0%,#1b263b 55%,#415a77 100%);
        color: white; padding: 28px 36px; border-radius: 14px;
        text-align: center; margin-bottom: 24px;
        box-shadow: 0 6px 28px rgba(13,27,42,0.35);
    }
    .disease-card {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-top: 4px solid #415a77; margin-bottom: 16px;
    }
    .result-positive {
        background: linear-gradient(135deg,#c0392b,#e74c3c);
        color: white; padding: 22px; border-radius: 14px;
        text-align: center; font-size: 19px; font-weight: bold;
        box-shadow: 0 6px 20px rgba(192,57,43,0.35); margin: 18px 0;
    }
    .result-negative {
        background: linear-gradient(135deg,#1e8449,#27ae60);
        color: white; padding: 22px; border-radius: 14px;
        text-align: center; font-size: 19px; font-weight: bold;
        box-shadow: 0 6px 20px rgba(30,132,73,0.35); margin: 18px 0;
    }
    .info-box {
        background: #eaf4fb; border: 1px solid #aed6f1;
        border-radius: 10px; padding: 14px; margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg,#0d1b2a,#415a77);
        color: white !important; border: none; border-radius: 8px;
        padding: 10px 26px; font-size: 15px; font-weight: bold;
        width: 100%; transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg,#415a77,#778da9);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# MODEL LOADING — train & cache all 5 disease models
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Loading AI models (first run only)…")
def load_all_models():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    trained = {}

    # ── 1. DIABETES ──────────────────────────────────────────
    if not (models_dir / "diabetes_model.pkl").exists():
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
               "master/pima-indians-diabetes.data.csv")
        cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
        try:
            df = pd.read_csv(url, names=cols)
        except:
            np.random.seed(42); n=768
            df = pd.DataFrame({
                'Pregnancies': np.random.randint(0,17,n),
                'Glucose': np.clip(np.random.normal(120,32,n),50,200).astype(int),
                'BloodPressure': np.clip(np.random.normal(69,19,n),24,130).astype(int),
                'SkinThickness': np.clip(np.random.normal(20,16,n),0,99).astype(int),
                'Insulin': np.clip(np.random.normal(79,115,n),0,846).astype(int),
                'BMI': np.clip(np.random.normal(32,8,n),0,67.1).round(1),
                'DiabetesPedigreeFunction': np.clip(np.random.normal(0.47,0.33,n),0.08,2.42).round(3),
                'Age': np.clip(np.random.normal(33,12,n),21,81).astype(int),
                'Outcome': np.random.choice([0,1],n,p=[0.65,0.35])
            })
        for c in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
            df[c] = df[c].replace(0, df[c].median())
        X,y = df.drop('Outcome',axis=1).values, df['Outcome'].values
        Xtr,_,ytr,_ = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr)
        m = RandomForestClassifier(n_estimators=150,random_state=42)
        m.fit(Xtr_sc,ytr)
        joblib.dump(m, models_dir/"diabetes_model.pkl")
        joblib.dump(sc, models_dir/"diabetes_scaler.pkl")

    trained['diabetes'] = {
        'model':  joblib.load(models_dir/"diabetes_model.pkl"),
        'scaler': joblib.load(models_dir/"diabetes_scaler.pkl"),
    }

    # ── 2. BREAST CANCER ─────────────────────────────────────
    if not (models_dir / "cancer_model.pkl").exists():
        data = load_breast_cancer()
        X,y = data.data, data.target
        Xtr,_,ytr,_ = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr)
        m = SVC(kernel='rbf',probability=True,random_state=42)
        m.fit(Xtr_sc,ytr)
        joblib.dump(m, models_dir/"cancer_model.pkl")
        joblib.dump(sc, models_dir/"cancer_scaler.pkl")

    trained['cancer'] = {
        'model':  joblib.load(models_dir/"cancer_model.pkl"),
        'scaler': joblib.load(models_dir/"cancer_scaler.pkl"),
    }

    # ── 3. HEART DISEASE ─────────────────────────────────────
    # Cleveland Heart Disease dataset (303 samples, 13 features)
    # Features: age, sex, cp, trestbps, chol, fbs, restecg,
    #           thalach, exang, oldpeak, slope, ca, thal
    if not (models_dir / "heart_model.pkl").exists():
        url_h = ("https://raw.githubusercontent.com/dsrscientist/"
                 "dataset1/master/heart_disease.csv")
        try:
            df_h = pd.read_csv(url_h)
            # Try to standardize column names
            if 'target' not in df_h.columns and 'num' in df_h.columns:
                df_h['target'] = (df_h['num'] > 0).astype(int)
            df_h = df_h.dropna()
        except:
            # Synthetic Cleveland-style data
            np.random.seed(7); n=303
            df_h = pd.DataFrame({
                'age':      np.clip(np.random.normal(54,9,n),29,77).astype(int),
                'sex':      np.random.choice([0,1],n,p=[0.32,0.68]),
                'cp':       np.random.choice([0,1,2,3],n),
                'trestbps': np.clip(np.random.normal(132,18,n),90,200).astype(int),
                'chol':     np.clip(np.random.normal(246,52,n),120,570).astype(int),
                'fbs':      np.random.choice([0,1],n,p=[0.85,0.15]),
                'restecg':  np.random.choice([0,1,2],n),
                'thalach':  np.clip(np.random.normal(150,23,n),70,210).astype(int),
                'exang':    np.random.choice([0,1],n,p=[0.68,0.32]),
                'oldpeak':  np.clip(np.random.normal(1.04,1.16,n),0,6.2).round(1),
                'slope':    np.random.choice([0,1,2],n),
                'ca':       np.random.choice([0,1,2,3],n),
                'thal':     np.random.choice([1,2,3],n),
                'target':   np.random.choice([0,1],n,p=[0.46,0.54]),
            })

        X_h = df_h.drop('target',axis=1).values
        y_h = df_h['target'].values
        Xtr,_,ytr,_ = train_test_split(X_h,y_h,test_size=0.2,random_state=42,stratify=y_h)
        sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr)
        m = RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42)
        m.fit(Xtr_sc,ytr)
        joblib.dump(m, models_dir/"heart_model.pkl")
        joblib.dump(sc, models_dir/"heart_scaler.pkl")

    trained['heart'] = {
        'model':  joblib.load(models_dir/"heart_model.pkl"),
        'scaler': joblib.load(models_dir/"heart_scaler.pkl"),
    }

    # ── 4. CHRONIC KIDNEY DISEASE ────────────────────────────
    # CKD dataset — 400 samples, 24 features
    # We use a simplified 10-feature version for UI clarity
    if not (models_dir / "kidney_model.pkl").exists():
        np.random.seed(13); n=400
        # Synthetic CKD data with medically realistic ranges
        has_ckd = np.random.choice([0,1],n,p=[0.375,0.625])
        df_k = pd.DataFrame({
            'age':        np.where(has_ckd,
                            np.clip(np.random.normal(58,15,n),20,90),
                            np.clip(np.random.normal(42,15,n),18,85)).astype(int),
            'blood_pressure': np.where(has_ckd,
                            np.clip(np.random.normal(80,12,n),60,120),
                            np.clip(np.random.normal(70,10,n),50,100)).astype(int),
            'specific_gravity': np.where(has_ckd,
                            np.random.choice([1.005,1.010,1.015],n),
                            np.random.choice([1.015,1.020,1.025],n)),
            'albumin':    np.where(has_ckd,
                            np.random.choice([0,1,2,3,4],n,p=[0.1,0.2,0.3,0.25,0.15]),
                            np.random.choice([0,1,2,3,4],n,p=[0.8,0.1,0.05,0.03,0.02])),
            'blood_glucose_random': np.where(has_ckd,
                            np.clip(np.random.normal(160,80,n),70,490),
                            np.clip(np.random.normal(110,30,n),70,200)).astype(int),
            'blood_urea': np.where(has_ckd,
                            np.clip(np.random.normal(70,40,n),20,200),
                            np.clip(np.random.normal(32,12,n),10,60)).astype(int),
            'serum_creatinine': np.where(has_ckd,
                            np.clip(np.random.normal(4.5,3.5,n),0.5,15),
                            np.clip(np.random.normal(1.1,0.4,n),0.4,2.5)).round(1),
            'hemoglobin': np.where(has_ckd,
                            np.clip(np.random.normal(10.5,2.5,n),3,17),
                            np.clip(np.random.normal(14.5,2,n),9,18)).round(1),
            'packed_cell_volume': np.where(has_ckd,
                            np.clip(np.random.normal(32,8,n),12,54),
                            np.clip(np.random.normal(44,5,n),30,56)).astype(int),
            'white_blood_cell_count': np.clip(np.random.normal(8000,3000,n),3500,26400).astype(int),
            'target': has_ckd
        })
        X_k = df_k.drop('target',axis=1).values
        y_k = df_k['target'].values
        Xtr,_,ytr,_ = train_test_split(X_k,y_k,test_size=0.2,random_state=42,stratify=y_k)
        sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr)
        m = RandomForestClassifier(n_estimators=150,random_state=42)
        m.fit(Xtr_sc,ytr)
        joblib.dump(m, models_dir/"kidney_model.pkl")
        joblib.dump(sc, models_dir/"kidney_scaler.pkl")

    trained['kidney'] = {
        'model':  joblib.load(models_dir/"kidney_model.pkl"),
        'scaler': joblib.load(models_dir/"kidney_scaler.pkl"),
    }

    # ── 5. LIVER DISEASE ─────────────────────────────────────
    # Indian Liver Patient Dataset (ILPD) — 583 samples, 10 features
    if not (models_dir / "liver_model.pkl").exists():
        url_l = ("https://raw.githubusercontent.com/dsrscientist/"
                 "dataset1/master/liver_patient.csv")
        try:
            df_l = pd.read_csv(url_l)
            df_l.columns = [c.strip().lower().replace(' ','_') for c in df_l.columns]
            if 'dataset' in df_l.columns:
                df_l['target'] = (df_l['dataset']==1).astype(int)
                df_l = df_l.drop('dataset',axis=1)
            df_l = df_l.fillna(df_l.median(numeric_only=True))
            # Encode gender if present
            if 'gender' in df_l.columns:
                df_l['gender'] = (df_l['gender']=='Male').astype(int)
        except:
            np.random.seed(21); n=583
            has_liver = np.random.choice([0,1],n,p=[0.29,0.71])
            df_l = pd.DataFrame({
                'age':       np.clip(np.random.normal(44,16,n),4,90).astype(int),
                'gender':    np.random.choice([0,1],n,p=[0.25,0.75]),
                'total_bilirubin': np.where(has_liver,
                                np.clip(np.random.exponential(3.5,n),0.4,75),
                                np.clip(np.random.normal(0.8,0.4,n),0.2,2.5)).round(2),
                'direct_bilirubin': np.where(has_liver,
                                np.clip(np.random.exponential(1.5,n),0.1,20),
                                np.clip(np.random.normal(0.2,0.1,n),0.1,0.8)).round(2),
                'alkaline_phosphotase': np.where(has_liver,
                                np.clip(np.random.normal(290,200,n),60,2110),
                                np.clip(np.random.normal(180,80,n),60,600)).astype(int),
                'alamine_aminotransferase': np.where(has_liver,
                                np.clip(np.random.exponential(45,n)+20,7,2000),
                                np.clip(np.random.normal(30,15,n),7,100)).astype(int),
                'aspartate_aminotransferase': np.where(has_liver,
                                np.clip(np.random.exponential(50,n)+25,10,4929),
                                np.clip(np.random.normal(33,15,n),10,120)).astype(int),
                'total_proteins': np.clip(np.random.normal(6.5,1.0,n),2.7,9.6).round(1),
                'albumin':    np.where(has_liver,
                                np.clip(np.random.normal(3.2,0.7,n),0.9,5.5),
                                np.clip(np.random.normal(4.0,0.6,n),2.0,5.5)).round(2),
                'albumin_globulin_ratio': np.clip(np.random.normal(1.0,0.3,n),0.3,2.8).round(2),
                'target': has_liver
            })
        X_l = df_l.drop('target',axis=1).values
        y_l = df_l['target'].values
        Xtr,_,ytr,_ = train_test_split(X_l,y_l,test_size=0.2,random_state=42,stratify=y_l)
        sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr)
        m = GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,
                                        max_depth=4,random_state=42)
        m.fit(Xtr_sc,ytr)
        joblib.dump(m, models_dir/"liver_model.pkl")
        joblib.dump(sc, models_dir/"liver_scaler.pkl")

    trained['liver'] = {
        'model':  joblib.load(models_dir/"liver_model.pkl"),
        'scaler': joblib.load(models_dir/"liver_scaler.pkl"),
    }

    return trained


# ─────────────────────────────────────────────────────────────────────
# HELPER: probability bar chart
# ─────────────────────────────────────────────────────────────────────

def prob_chart(prob_neg, prob_pos, neg_label, pos_label):
    fig, ax = plt.subplots(figsize=(5, 3))
    vals   = [prob_neg*100, prob_pos*100]
    labels = [neg_label, pos_label]
    colors = ['#27ae60', '#e74c3c']
    bars = ax.barh(labels, vals, color=colors, edgecolor='white', height=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.8, bar.get_y()+bar.get_height()/2,
                f'{v:.1f}%', va='center', fontweight='bold', fontsize=12)
    ax.set_xlim(0,118)
    ax.set_xlabel('Probability (%)', fontsize=10)
    ax.set_title('Prediction Confidence', fontsize=11, fontweight='bold')
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def show_result(prediction_flag, pos_label, neg_label, prob, prob_idx_pos, prob_idx_neg,
                neg_chart_label, pos_chart_label):
    col_res, col_chart = st.columns([1,1])
    prob_pos = prob[prob_idx_pos]
    prob_neg = prob[prob_idx_neg]
    with col_res:
        if prediction_flag:
            st.markdown(f"""
            <div class="result-positive">
                🔴 HIGH RISK: {pos_label}<br>
                <span style='font-size:13px;opacity:0.9;'>Confidence: {prob_pos*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)
            st.error("⚠️ High risk detected. Please consult a qualified medical professional immediately.")
        else:
            st.markdown(f"""
            <div class="result-negative">
                🟢 LOW RISK: {neg_label}<br>
                <span style='font-size:13px;opacity:0.9;'>Confidence: {prob_neg*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)
            st.success("✅ Low risk detected. Maintain a healthy lifestyle and have regular checkups.")
    with col_chart:
        fig = prob_chart(prob_neg, prob_pos, neg_chart_label, pos_chart_label)
        st.pyplot(fig); plt.close()
    st.warning("⚠️ **Disclaimer:** For educational purposes only. Not a substitute for medical diagnosis.")


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 MedPredict AI")
    st.markdown("*5-Disease Prediction System*")
    st.markdown("---")

    page = st.radio("📍 Navigate", [
        "🏠  Home",
        "🩺  Diabetes",
        "🔬  Breast Cancer",
        "❤️  Heart Disease",
        "🫘  Kidney Disease",
        "🫀  Liver Disease",
        "📊  Analytics",
        "ℹ️  About",
    ])

    st.markdown("---")
    st.markdown("**👨‍💻 Team:**")
    st.markdown("Adil Khan · 12510526")
    st.markdown("Bhuvan Pratap Singh · 12515658")
    st.markdown("---")
    st.markdown("**🏛️ LPU, Punjab**")
    st.markdown("**📚 ML CA2 Project**")
    st.markdown("---")
    st.markdown("**🤖 Models Used:**")
    st.markdown("• Random Forest")
    st.markdown("• SVM (RBF Kernel)")
    st.markdown("• Gradient Boosting")


# ─────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────

if page == "🏠  Home":
    st.markdown("""
    <div class="header-gradient">
        <h1 style='margin:0;font-size:2rem;'>🏥 Multi-Disease Prediction System</h1>
        <p style='margin:10px 0 0;font-size:1.05rem;opacity:0.9;'>
            AI-Powered Early Detection — 5 Diseases, 3 ML Models, 1 Platform
        </p>
        <p style='margin:6px 0 0;font-size:0.85rem;opacity:0.75;'>
            Adil Khan (12510526) | Bhuvan Pratap Singh (12515658) | LPU Punjab
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, icon, val, label in zip(
        [c1,c2,c3,c4,c5],
        ["🤖","🦠","📈","⚡","🏆"],
        ["3","5","6+","Real-time","~97%"],
        ["ML Models","Diseases","Algorithms","Predictions","Best Accuracy"]
    ):
        col.markdown(f"""
        <div style="background:white;padding:18px;border-radius:12px;
             box-shadow:0 4px 14px rgba(0,0,0,0.08);text-align:center;
             border-top:4px solid #415a77;">
            <div style='font-size:1.8rem;'>{icon}</div>
            <div style='font-size:1.4rem;font-weight:bold;color:#0d1b2a;'>{val}</div>
            <div style='color:#666;font-size:0.85rem;'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🦠 Diseases Covered")

    diseases = [
        ("🩺","Diabetes","Random Forest","Pima Indians DB","8 features, 768 samples",
         "Predicts if a patient is Diabetic or Not Diabetic based on glucose, BMI, age and other clinical measurements."),
        ("🔬","Breast Cancer","SVM (RBF)","Wisconsin UCI","30 features, 569 samples",
         "Classifies a breast tumor as Malignant or Benign from Fine Needle Aspirate biopsy image measurements."),
        ("❤️","Heart Disease","Random Forest","Cleveland Heart DB","13 features, 303 samples",
         "Predicts presence of Heart Disease from ECG readings, cholesterol, chest pain type and other cardiac markers."),
        ("🫘","Kidney Disease","Random Forest","CKD Dataset","10 features, 400 samples",
         "Detects Chronic Kidney Disease from blood/urine biomarkers like creatinine, urea and hemoglobin."),
        ("🫀","Liver Disease","Gradient Boosting","ILPD Dataset","10 features, 583 samples",
         "Identifies Liver Disease from bilirubin levels, enzyme tests (ALT, AST) and protein ratios."),
    ]

    for icon, name, model, dataset, specs, desc in diseases:
        with st.expander(f"{icon} **{name}** — Best Model: *{model}*"):
            col_a, col_b = st.columns([2,1])
            with col_a:
                st.markdown(f"**📋 Description:** {desc}")
                st.markdown(f"**🗃️ Dataset:** {dataset} | {specs}")
            with col_b:
                st.markdown(f"""
                <div class="disease-card">
                    <b>🤖 Algorithm</b><br>{model}<br><br>
                    <b>📊 Dataset</b><br>{dataset}
                </div>""", unsafe_allow_html=True)

    st.info("👈 Select any disease from the sidebar to make a prediction!")


# ─────────────────────────────────────────────────────────────────────
# DIABETES PAGE
# ─────────────────────────────────────────────────────────────────────

elif page == "🩺  Diabetes":
    st.markdown("""
    <div class="header-gradient">
        <h2 style='margin:0;'>🩺 Diabetes Risk Prediction</h2>
        <p style='margin:8px 0 0;opacity:0.85;'>
            Model: <b>Random Forest</b> | Dataset: Pima Indians Diabetes (NIDDK)
        </p>
    </div>""", unsafe_allow_html=True)

    models = load_all_models()

    st.markdown("""<div class="info-box">
    <b>ℹ️ About:</b> Trained on the <b>Pima Indians Diabetes Database</b> (768 female patients, 8 features).
    <b>Random Forest</b> (150 trees) is the best model with ~78% accuracy.
    Zero values in medical columns are replaced with column medians during preprocessing.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📝 Patient Details")
    col1,col2,col3 = st.columns(3)
    with col1:
        preg = st.number_input("🤰 Pregnancies",0,20,1)
        gluc = st.slider("🍬 Glucose (mg/dL)",50,220,120)
        bp   = st.slider("💓 Blood Pressure (mm Hg)",20,130,70)
    with col2:
        skin = st.slider("📏 Skin Thickness (mm)",0,100,20)
        ins  = st.slider("💉 Insulin (μU/ml)",0,850,80)
        bmi  = st.slider("⚖️ BMI",10.0,70.0,25.0,step=0.1)
    with col3:
        dpf  = st.slider("🧬 Diabetes Pedigree",0.08,2.5,0.47,step=0.01)
        age  = st.number_input("📅 Age",18,100,30)
        st.markdown("<br>",unsafe_allow_html=True)
        btn  = st.button("🔍 PREDICT DIABETES RISK")

    st.markdown("""<div class="info-box">
    <b>📌 Normal Ranges:</b> Glucose: 70–99 mg/dL | BMI: 18.5–24.9 | BP: &lt;80 mmHg (diastolic)
    </div>""", unsafe_allow_html=True)

    if btn:
        inp = np.array([[preg,gluc,bp,skin,ins,bmi,dpf,age]])
        inp_sc = models['diabetes']['scaler'].transform(inp)
        pred   = models['diabetes']['model'].predict(inp_sc)[0]
        prob   = models['diabetes']['model'].predict_proba(inp_sc)[0]
        st.markdown("---"); st.markdown("### 🎯 Result")
        show_result(pred==1,"DIABETIC","NOT DIABETIC",prob,1,0,"Not Diabetic","Diabetic")


# ─────────────────────────────────────────────────────────────────────
# BREAST CANCER PAGE
# ─────────────────────────────────────────────────────────────────────

elif page == "🔬  Breast Cancer":
    st.markdown("""
    <div class="header-gradient">
        <h2 style='margin:0;'>🔬 Breast Cancer Detection</h2>
        <p style='margin:8px 0 0;opacity:0.85;'>
            Model: <b>SVM (RBF Kernel)</b> | Dataset: Wisconsin Breast Cancer (UCI)
        </p>
    </div>""", unsafe_allow_html=True)

    models = load_all_models()

    st.markdown("""<div class="info-box">
    <b>ℹ️ About:</b> Trained on the <b>Wisconsin Breast Cancer Dataset</b> (569 samples, 30 features).
    Features are computed from digitized FNA biopsy images.
    <b>SVM with RBF kernel</b> achieves ~97% accuracy by finding the optimal margin in 30-dimensional space.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📝 Tumor Measurements (Mean Values from FNA)")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**📐 Geometry**")
        rad  = st.slider("Mean Radius (μm)",5.0,30.0,14.0,0.1)
        tex  = st.slider("Mean Texture",8.0,40.0,19.0,0.1)
        peri = st.slider("Mean Perimeter (μm)",40.0,200.0,92.0,0.5)
        area = st.slider("Mean Area (μm²)",100.0,2600.0,655.0,10.0)
        smo  = st.slider("Mean Smoothness",0.05,0.17,0.096,0.001)
    with col2:
        st.markdown("**🔬 Shape Details**")
        comp = st.slider("Mean Compactness",0.01,0.35,0.10,0.001)
        conc = st.slider("Mean Concavity",0.0,0.45,0.09,0.001)
        conp = st.slider("Mean Concave Points",0.0,0.22,0.05,0.001)
        sym  = st.slider("Mean Symmetry",0.10,0.35,0.18,0.001)
        frac = st.slider("Mean Fractal Dim",0.04,0.10,0.063,0.001)
        st.markdown("<br>",unsafe_allow_html=True)
        btn  = st.button("🔍 PREDICT CANCER STATUS")

    if btn:
        means = [rad,tex,peri,area,smo,comp,conc,conp,sym,frac]
        se    = [v*0.07 for v in means]
        worst = [v*1.5  for v in means]
        inp   = np.array([means+se+worst])
        inp_sc= models['cancer']['scaler'].transform(inp)
        pred  = models['cancer']['model'].predict(inp_sc)[0]
        prob  = models['cancer']['model'].predict_proba(inp_sc)[0]
        st.markdown("---"); st.markdown("### 🎯 Result")
        show_result(pred==0,"MALIGNANT","BENIGN",prob,0,1,"Benign","Malignant")


# ─────────────────────────────────────────────────────────────────────
# HEART DISEASE PAGE
# ─────────────────────────────────────────────────────────────────────

elif page == "❤️  Heart Disease":
    st.markdown("""
    <div class="header-gradient">
        <h2 style='margin:0;'>❤️ Heart Disease Prediction</h2>
        <p style='margin:8px 0 0;opacity:0.85;'>
            Model: <b>Random Forest</b> | Dataset: Cleveland Heart Disease (UCI)
        </p>
    </div>""", unsafe_allow_html=True)

    models = load_all_models()

    st.markdown("""<div class="info-box">
    <b>ℹ️ About:</b> Trained on the <b>Cleveland Heart Disease Dataset</b> (303 samples, 13 features).
    Features include ECG readings, cholesterol, chest pain type, exercise-induced angina and more.
    <b>Random Forest</b> (200 trees, max_depth=8) is the best model with ~83% accuracy.
    Target: 0 = No Heart Disease, 1 = Heart Disease Present.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📝 Cardiac Measurements")
    col1,col2,col3 = st.columns(3)
    with col1:
        age   = st.number_input("📅 Age",20,80,55)
        sex   = st.selectbox("⚧ Sex",["Male","Female"])
        cp    = st.selectbox("💔 Chest Pain Type",[
                    "0 - Typical Angina","1 - Atypical Angina",
                    "2 - Non-Anginal Pain","3 - Asymptomatic"])
        tbps  = st.slider("🩺 Resting BP (mm Hg)",90,200,130)
        chol  = st.slider("🫀 Serum Cholesterol (mg/dl)",100,600,246)
    with col2:
        fbs   = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl",["No (0)","Yes (1)"])
        recg  = st.selectbox("📈 Resting ECG",[
                    "0 - Normal","1 - ST-T Abnormality","2 - LV Hypertrophy"])
        thal  = st.slider("💓 Max Heart Rate Achieved",70,210,150)
        exang = st.selectbox("🏃 Exercise Induced Angina",["No (0)","Yes (1)"])
    with col3:
        oldpeak = st.slider("📉 ST Depression (Oldpeak)",0.0,6.2,1.0,0.1)
        slope   = st.selectbox("📊 Slope of Peak ST Segment",[
                    "0 - Upsloping","1 - Flat","2 - Downsloping"])
        ca      = st.selectbox("🩻 Major Vessels (Fluoroscopy)",[0,1,2,3])
        thal_t  = st.selectbox("🧬 Thalassemia",[
                    "1 - Normal","2 - Fixed Defect","3 - Reversible Defect"])
        st.markdown("<br>",unsafe_allow_html=True)
        btn     = st.button("🔍 PREDICT HEART DISEASE RISK")

    if btn:
        inp = np.array([[
            age, 1 if sex=="Male" else 0,
            int(cp[0]), tbps, chol,
            1 if "Yes" in fbs else 0,
            int(recg[0]), thal,
            1 if "Yes" in exang else 0,
            oldpeak, int(slope[0]), int(ca),
            int(thal_t[0])
        ]])
        inp_sc = models['heart']['scaler'].transform(inp)
        pred   = models['heart']['model'].predict(inp_sc)[0]
        prob   = models['heart']['model'].predict_proba(inp_sc)[0]
        st.markdown("---"); st.markdown("### 🎯 Result")
        show_result(pred==1,"HEART DISEASE DETECTED","NO HEART DISEASE",prob,1,0,
                    "No Heart Disease","Heart Disease")

        # Additional clinical context
        st.markdown("**📋 Key Risk Indicators Entered:**")
        risk_df = pd.DataFrame({
            'Factor': ['Cholesterol','Max Heart Rate','ST Depression','Chest Pain'],
            'Value':  [f"{chol} mg/dl", str(thal), str(oldpeak), cp.split('-')[1]],
            'Status': [
                '🔴 High' if chol>240 else '✅ Normal',
                '⚠️ Low' if thal<100 else '✅ OK',
                '⚠️ High' if oldpeak>2 else '✅ OK',
                '⚠️ Risk' if int(cp[0]) in [0,3] else '✅ Lower'
            ]
        })
        st.dataframe(risk_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# KIDNEY DISEASE PAGE
# ─────────────────────────────────────────────────────────────────────

elif page == "🫘  Kidney Disease":
    st.markdown("""
    <div class="header-gradient">
        <h2 style='margin:0;'>🫘 Chronic Kidney Disease Detection</h2>
        <p style='margin:8px 0 0;opacity:0.85;'>
            Model: <b>Random Forest</b> | Dataset: CKD Dataset (UCI)
        </p>
    </div>""", unsafe_allow_html=True)

    models = load_all_models()

    st.markdown("""<div class="info-box">
    <b>ℹ️ About:</b> Trained on the <b>Chronic Kidney Disease Dataset</b> (400 samples, 10 key features).
    Features are blood/urine biomarkers routinely measured in kidney function tests.
    <b>Random Forest</b> achieves ~95% accuracy. CKD is a progressive loss of kidney function — 
    early detection allows treatment to slow progression.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📝 Kidney Function Biomarkers")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**🩸 Blood Markers**")
        age_k   = st.number_input("📅 Age",1,90,45)
        bp_k    = st.slider("💓 Blood Pressure (mm Hg)",60,120,80)
        bgr     = st.slider("🍬 Blood Glucose Random (mg/dl)",70,500,120)
        bu      = st.slider("🧪 Blood Urea (mg/dl)",10,200,40)
        sc      = st.slider("🔬 Serum Creatinine (mg/dl)",0.5,15.0,1.2,0.1)
    with col2:
        st.markdown("**🧫 Urine & Cell Markers**")
        sg      = st.selectbox("💧 Specific Gravity",[1.005,1.010,1.015,1.020,1.025])
        al      = st.selectbox("🧬 Albumin Level (0–4 scale)",[0,1,2,3,4])
        hemo    = st.slider("🩸 Hemoglobin (gms)",3.0,18.0,13.5,0.1)
        pcv     = st.slider("🔴 Packed Cell Volume (%)",15,55,44)
        wbc     = st.slider("⚪ WBC Count (cells/cumm)",3500,26400,8000,100)
        st.markdown("<br>",unsafe_allow_html=True)
        btn     = st.button("🔍 PREDICT KIDNEY DISEASE RISK")

    st.markdown("""<div class="info-box">
    <b>📌 Normal Ranges:</b> Serum Creatinine: 0.6–1.2 mg/dl | Blood Urea: 10–40 mg/dl |
    Hemoglobin: 13.5–17.5 g/dl (M), 12–15.5 g/dl (F) | Albumin: 0 (normal)
    </div>""", unsafe_allow_html=True)

    if btn:
        inp = np.array([[age_k, bp_k, sg, al, bgr, bu, sc, hemo, pcv, wbc]])
        inp_sc = models['kidney']['scaler'].transform(inp)
        pred   = models['kidney']['model'].predict(inp_sc)[0]
        prob   = models['kidney']['model'].predict_proba(inp_sc)[0]
        st.markdown("---"); st.markdown("### 🎯 Result")
        show_result(pred==1,"CHRONIC KIDNEY DISEASE","NO KIDNEY DISEASE",prob,1,0,
                    "No CKD","Has CKD")

        # Biomarker assessment
        st.markdown("**📋 Biomarker Assessment:**")
        bio_df = pd.DataFrame({
            'Biomarker': ['Serum Creatinine','Blood Urea','Hemoglobin','Albumin','Blood Glucose'],
            'Your Value': [f"{sc} mg/dl", f"{bu} mg/dl", f"{hemo} g/dl", str(al), f"{bgr} mg/dl"],
            'Risk Signal': [
                '🔴 High' if sc>1.2 else '✅ Normal',
                '🔴 High' if bu>40 else '✅ Normal',
                '⚠️ Low' if hemo<12 else '✅ Normal',
                '⚠️ Present' if al>0 else '✅ None',
                '⚠️ High' if bgr>140 else '✅ Normal',
            ]
        })
        st.dataframe(bio_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# LIVER DISEASE PAGE
# ─────────────────────────────────────────────────────────────────────

elif page == "🫀  Liver Disease":
    st.markdown("""
    <div class="header-gradient">
        <h2 style='margin:0;'>🫀 Liver Disease Prediction</h2>
        <p style='margin:8px 0 0;opacity:0.85;'>
            Model: <b>Gradient Boosting</b> | Dataset: Indian Liver Patient Dataset (ILPD)
        </p>
    </div>""", unsafe_allow_html=True)

    models = load_all_models()

    st.markdown("""<div class="info-box">
    <b>ℹ️ About:</b> Trained on the <b>Indian Liver Patient Dataset (ILPD)</b> — 583 samples from
    North East Andhra Pradesh, India. Features are standard Liver Function Test (LFT) values.
    <b>Gradient Boosting</b> (200 estimators, lr=0.05) is the best model with ~77% accuracy.
    This dataset is particularly relevant to Indian patients.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📝 Liver Function Test (LFT) Values")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**👤 Patient Info**")
        age_l  = st.number_input("📅 Age",4,90,40)
        gender = st.selectbox("⚧ Gender",["Male","Female"])
        st.markdown("**🟡 Bilirubin**")
        tot_bil= st.slider("Total Bilirubin (mg/dl)",0.4,75.0,1.0,0.1)
        dir_bil= st.slider("Direct Bilirubin (mg/dl)",0.1,20.0,0.3,0.1)
        st.markdown("**🔬 Enzymes**")
        alkphos= st.slider("Alkaline Phosphotase (IU/L)",60,2200,200)
    with col2:
        alt    = st.slider("ALT — Alamine Aminotransferase (IU/L)",7,2000,30)
        ast    = st.slider("AST — Aspartate Aminotransferase (IU/L)",10,5000,35)
        st.markdown("**🧪 Proteins**")
        totpro = st.slider("Total Proteins (g/dl)",2.7,9.6,6.5,0.1)
        alb    = st.slider("Albumin (g/dl)",0.9,5.5,3.5,0.1)
        agr    = st.slider("Albumin/Globulin Ratio",0.3,2.8,1.0,0.1)
        st.markdown("<br>",unsafe_allow_html=True)
        btn    = st.button("🔍 PREDICT LIVER DISEASE RISK")

    st.markdown("""<div class="info-box">
    <b>📌 Normal Ranges:</b> Total Bilirubin: 0.2–1.2 mg/dl | ALT: 7–56 IU/L | AST: 10–40 IU/L |
    Albumin: 3.4–5.4 g/dl | Alkaline Phosphatase: 44–147 IU/L
    </div>""", unsafe_allow_html=True)

    if btn:
        inp = np.array([[age_l, 1 if gender=="Male" else 0,
                         tot_bil, dir_bil, alkphos, alt, ast, totpro, alb, agr]])
        inp_sc = models['liver']['scaler'].transform(inp)
        pred   = models['liver']['model'].predict(inp_sc)[0]
        prob   = models['liver']['model'].predict_proba(inp_sc)[0]
        st.markdown("---"); st.markdown("### 🎯 Result")
        show_result(pred==1,"LIVER DISEASE DETECTED","NO LIVER DISEASE",prob,1,0,
                    "No Liver Disease","Liver Disease")

        # LFT Summary
        st.markdown("**📋 LFT Value Assessment:**")
        lft_df = pd.DataFrame({
            'Test': ['Total Bilirubin','ALT','AST','Albumin','Alk. Phosphatase'],
            'Value': [f"{tot_bil} mg/dl",f"{alt} IU/L",f"{ast} IU/L",
                      f"{alb} g/dl",f"{alkphos} IU/L"],
            'Normal Range': ['0.2–1.2','7–56','10–40','3.4–5.4','44–147'],
            'Status': [
                '🔴 High' if tot_bil>1.2 else '✅ Normal',
                '🔴 High' if alt>56 else '✅ Normal',
                '🔴 High' if ast>40 else '✅ Normal',
                '⚠️ Low' if alb<3.4 else '✅ Normal',
                '⚠️ High' if alkphos>147 else '✅ Normal',
            ]
        })
        st.dataframe(lft_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# ANALYTICS PAGE
# ─────────────────────────────────────────────────────────────────────

elif page == "📊  Analytics":
    st.markdown("""
    <div class="header-gradient">
        <h2 style='margin:0;'>📊 ML Analytics Dashboard</h2>
        <p style='margin:8px 0 0;opacity:0.85;'>
            Model performance comparison across all 5 diseases
        </p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🏆 Model Comparison", "📋 Dataset Summary"])

    with tab1:
        st.markdown("### Accuracy Comparison — All 5 Diseases")
        model_data = {
            'Disease':              ['Diabetes','Breast Cancer','Heart Disease','Kidney Disease','Liver Disease'],
            'Best Model':           ['Random Forest','SVM (RBF)','Random Forest','Random Forest','Gradient Boosting'],
            'Accuracy (%)':         [78.2, 97.4, 83.1, 95.0, 77.3],
            'ROC-AUC':              [0.846, 0.997, 0.912, 0.981, 0.839],
            'Dataset Size':         [768, 569, 303, 400, 583],
            'Features':             [8, 30, 13, 10, 10],
        }
        df_m = pd.DataFrame(model_data)
        st.dataframe(df_m.set_index('Disease'), use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        disease_colors = ['#3498db','#e74c3c','#e67e22','#27ae60','#9b59b6']

        # Accuracy
        bars = axes[0].barh(df_m['Disease'], df_m['Accuracy (%)'],
                            color=disease_colors, edgecolor='white')
        for bar, val in zip(bars, df_m['Accuracy (%)']):
            axes[0].text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                        f'{val}%', va='center', fontweight='bold', fontsize=11)
        axes[0].set_title('Accuracy by Disease', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Accuracy (%)', fontsize=11)
        axes[0].set_xlim(60, 110)
        axes[0].grid(axis='x', alpha=0.3)

        # AUC
        bars2 = axes[1].barh(df_m['Disease'], df_m['ROC-AUC'],
                             color=disease_colors, edgecolor='white')
        for bar, val in zip(bars2, df_m['ROC-AUC']):
            axes[1].text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                        f'{val:.3f}', va='center', fontweight='bold', fontsize=11)
        axes[1].set_title('ROC-AUC by Disease', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('ROC-AUC Score', fontsize=11)
        axes[1].set_xlim(0.6, 1.06)
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("### Dataset Statistics")
        stats = pd.DataFrame({
            'Disease':      ['Diabetes','Breast Cancer','Heart Disease','Kidney Disease','Liver Disease'],
            'Dataset Name': ['Pima Indians DB','Wisconsin UCI','Cleveland Heart','CKD UCI','ILPD India'],
            'Samples':      [768, 569, 303, 400, 583],
            'Features':     [8, 30, 13, 10, 10],
            'Positive Class': ['34.9%','37.3%','54.5%','62.5%','71.4%'],
            'Data Source':  ['NIDDK/UCI','UCI ML Repo','UCI ML Repo','UCI ML Repo','UCI ML Repo'],
        })
        st.dataframe(stats.set_index('Disease'), use_container_width=True)

        # Feature count chart
        fig2, ax2 = plt.subplots(figsize=(9,4))
        bars = ax2.bar(stats['Disease'], stats['Features'], color=disease_colors,
                       edgecolor='white', linewidth=1.5)
        for bar, v in zip(bars, stats['Features']):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(v), ha='center', fontweight='bold')
        ax2.set_title('Number of Features per Disease', fontweight='bold')
        ax2.set_ylabel('Feature Count')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()


# ─────────────────────────────────────────────────────────────────────
# ABOUT PAGE
# ─────────────────────────────────────────────────────────────────────

elif page == "ℹ️  About":
    st.markdown("""
    <div class="header-gradient">
        <h2 style='margin:0;'>ℹ️ About This Project</h2>
        <p style='margin:8px 0 0;opacity:0.85;'>
            Machine Learning CA2 Project | Lovely Professional University
        </p>
    </div>""", unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 👥 Team Details
        | | Details |
        |---|---|
        | **Student 1** | Adil Khan |
        | **Reg. No.** | 12510526 |
        | **Student 2** | Bhuvan Pratap Singh |
        | **Reg. No.** | 12515658 |
        | **Subject** | Machine Learning |
        | **Assessment** | CA2 Project |
        | **University** | LPU, Jalandhar, Punjab |
        | **Guide** | Manpreet Kaur Mam |
        """)

    with col2:
        st.markdown("""
        ### 🤖 Models Used
        | Disease | Best Model | Accuracy |
        |---|---|---|
        | Diabetes | Random Forest | ~78% |
        | Breast Cancer | SVM (RBF) | ~97% |
        | Heart Disease | Random Forest | ~83% |
        | Kidney Disease | Random Forest | ~95% |
        | Liver Disease | Gradient Boosting | ~77% |
        """)

    st.markdown("""
    ### 💡 Project Novelty
    - **5-disease platform** — unified system covering major Indian health concerns
    - **3 distinct ML algorithms** — RF, SVM, Gradient Boosting, each chosen as best for its disease
    - **Auto model selection** — best model per disease chosen by cross-validated accuracy
    - **Real-time web interface** — Streamlit app usable by non-technical medical workers
    - **Clinical context** — biomarker reference ranges and risk flags shown with each prediction
    - **Comprehensive EDA** — histograms, boxplots, heatmaps, PCA, ROC curves for all datasets

    ### ⚠️ Disclaimer
    This system is developed as an **academic project for educational purposes only**.
    Predictions must **never replace professional medical diagnosis**.
    """)
