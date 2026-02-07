from ..simulation.generate_pulses import simulate_pulse, simulate_pileup
from ..features.extraction import extract_features
import joblib
import numpy as np
import pandas as pd

np.random.seed(42)

def run_blind_test():
    model = joblib.load("results/models/psa_random_forest.pkl")
    t = np.linspace(0, 100, 500)
    
    # Generate 50 mystery events
    mystery_data = []
    true_labels = []
    
    for i in range(50):
        choice = np.random.choice(['signal', 'bg', 'pileup'])
        if choice == 'signal':
            p = simulate_pulse(t, 20, 1.0, 2.0, 0.25)
            true_labels.append(1)
        elif choice == 'bg':
            p = simulate_pulse(t, 20, 0.8, 8.0, 0.25)
            true_labels.append(0)
        else: # Pileup - two overlapping pulses
            p = simulate_pileup(t, 20, 0.25)
            true_labels.append(0) 
            
        feat = extract_features(t, p)
        mystery_data.append(feat)

    # Predict
    df_mystery = pd.DataFrame(mystery_data, columns=['peak', 'rise_time', 'area', 'fwhm', 'chi_sq'])
    predictions = model.predict(df_mystery)
    
    proba = model.predict_proba(df_mystery)[:, 1]
    SIGNAL_THRESHOLD = 0.995
    predictions = proba > SIGNAL_THRESHOLD


    # Calculate success against Discovery criteria
    fake_discoveries = np.sum((predictions == 1) & (np.array(true_labels) == 0))


    print("\n--- BLIND PHYSICAL TEST ---")
    print(f"Total mystery events     : 50")
    print(f"Signal candidates found : {predictions.sum()}")
    print(f"Background leakage      : {fake_discoveries}")

    if fake_discoveries > 0:
        print("STATUS: Discovery NOT safe (background leakage present)")
    else:
        print("STATUS: Discovery SAFE (0 expected background events)")
    

if __name__ == "__main__":
    run_blind_test()