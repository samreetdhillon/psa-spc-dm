import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def stress_test_report():
    # Load data from the noisy run (after you rerun main.py with noise=0.25)
    df = pd.read_csv("data/processed/pulse_features.csv")
    X = df[['peak', 'rise_time', 'area', 'fwhm', 'chi_sq']]
    y = df['label']
    
    # Load your existing trained model
    model = joblib.load("results/models/psa_random_forest.pkl")
    
    # Check how it performs on the noisy data
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    print(f"--- STRESS TEST REPORT ---")
    print(f"Detector Noise Level: 0.25 (High)")
    print(f"System Accuracy: {acc*100:.2f}%")
    
    if acc < 0.95:
        print("Conclusion: Discrimination is significantly degraded by electronic noise.")
        print("Recommendation: Implement a Band-pass filter in Phase 2 feature extraction.")
    else:
        print("Conclusion: The classifier is robust even at high noise levels.")

if __name__ == "__main__":
    stress_test_report()