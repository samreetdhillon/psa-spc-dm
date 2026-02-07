import pandas as pd
import numpy as np
import joblib

def run_physical_analysis():
    df = pd.read_csv("data/processed/pulse_features.csv")

    FEATURES = ['peak', 'rise_time', 'area', 'fwhm', 'chi_sq']
    X = df[FEATURES]
    y = df['label']  # 0 = ER (background), 1 = NR (signal)

    model = joblib.load("results/models/psa_random_forest.pkl")
    signal_prob = model.predict_proba(X)[:, 1]

    SIGNAL_THRESHOLD = 0.995  # ultra-conservative discovery cut
    is_signal_candidate = signal_prob > SIGNAL_THRESHOLD

    n_total = len(df)
    n_signal_candidates = is_signal_candidate.sum()
    n_background_leakage = ((y == 0) & is_signal_candidate).sum()

    mu_b = n_background_leakage
    p_false = 1 - np.exp(-mu_b)

    print("\n========== PHYSICAL ANALYSIS ==========")
    print(f"Total triggered events        : {n_total}")
    print(f"Signal candidates after PSA  : {n_signal_candidates}")
    print(f"Background leakage (ER→NR)   : {n_background_leakage}")
    print("--------------------------------------")
    print(f"Expected false discovery μ_b : {mu_b:.2f}")
    print(f"P(≥1 false discovery)        : {100*p_false:.2f}%")
    print("======================================\n")

if __name__ == "__main__":
    run_physical_analysis()


