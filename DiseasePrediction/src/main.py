"""
╔══════════════════════════════════════════════════════════════╗
║       DUAL DISEASE PREDICTION SYSTEM — VS CODE MAIN         ║
║  Adil Khan (12510526) | Bhuvan Pratap Singh (12515658)      ║
║  LPU Punjab | ML CA2 Project                               ║
╚══════════════════════════════════════════════════════════════╝

Run this file from project root:
    python src/main.py

Or run individual modules:
    python src/preprocess.py
    python src/train_models.py
    python src/evaluate.py
    python src/predict.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import run_preprocessing
from src.train_models import run_training
from src.evaluate import run_evaluation
from src.predict import run_prediction_demo


def print_banner():
    print("\n" + "╔" + "═"*62 + "╗")
    print("║" + " "*15 + "DUAL DISEASE PREDICTION SYSTEM" + " "*16 + "║")
    print("║" + " "*10 + "Adil Khan | Bhuvan Pratap Singh" + " "*10 + "║")
    print("║" + " "*12 + "LPU Punjab | ML CA2 Project" + " "*12 + "║")
    print("╚" + "═"*62 + "╝\n")


def main():
    print_banner()

    print("Choose what to run:")
    print("  1. Full Pipeline (Preprocess + Train + Evaluate + Demo)")
    print("  2. Preprocessing only")
    print("  3. Training only (requires preprocessing)")
    print("  4. Evaluation only (requires training)")
    print("  5. Prediction demo (requires training)")
    print("  6. Launch Streamlit Web App")
    print()

    try:
        choice = input("Enter choice (1-6) [default=1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "1":
        print("\n🚀 Running Full Pipeline...\n")
        run_preprocessing()
        run_training()
        run_evaluation()
        run_prediction_demo()
        print("\n✅ Full pipeline complete!")
        print("📁 Check outputs/ folder for all graphs and results.")

    elif choice == "2":
        run_preprocessing()

    elif choice == "3":
        run_training()

    elif choice == "4":
        run_evaluation()

    elif choice == "5":
        run_prediction_demo()

    elif choice == "6":
        print("\n🌐 Launching Streamlit Web App...")
        print("   URL: http://localhost:8501")
        os.system("streamlit run app/streamlit_app.py")

    else:
        print("❌ Invalid choice. Running full pipeline...")
        run_preprocessing()
        run_training()
        run_evaluation()
        run_prediction_demo()


if __name__ == "__main__":
    main()
