from ..simulation.generate_pulses import simulate_pulse, simulate_pileup
from ..features.extraction import extract_features
import joblib
import numpy as np
import pandas as pd

np.random.seed(42)

def run_blind_test():
    model = joblib.load("results/models/psa_random_forest.pkl")
    t = np.linspace(0, 100, 500)
    
    # Generate 50 'mystery' events
    mystery_data = []
    true_labels = [] # We keep these for the final grade
    
    for i in range(50):
        choice = np.random.choice(['signal', 'bg', 'pileup'])
        if choice == 'signal':
            p = simulate_pulse(t, 20, 1.0, 2.0, 0.25)
            true_labels.append(1)
        elif choice == 'bg':
            p = simulate_pulse(t, 20, 0.8, 8.0, 0.25)
            true_labels.append(0)
        else: # Pileup
            p = simulate_pileup(t, 20, 0.25)
            true_labels.append(0) # Pileup is technically background (noise)
            
        feat = extract_features(t, p)
        mystery_data.append(feat)

    # Predict
    df_mystery = pd.DataFrame(mystery_data, columns=['peak', 'rise_time', 'area', 'fwhm', 'chi_sq'])
    predictions = model.predict(df_mystery)
    
    # Calculate success against "Discovery"
    # In DM physics, a 'False Positive' (predicting 1 when it's 0) is a fake discovery.
    fake_discoveries = np.sum((predictions == 1) & (np.array(true_labels) == 0))
    
    print(f"--- BLIND TEST RESULTS ---")
    print(f"Total mystery events: 50")
    print(f"Fake Discoveries (False Positives): {fake_discoveries}")
    
    if fake_discoveries > 0:
        print("ALERT: The model is mistaking pile-up/noise for Dark Matter signals!")
    else:
        print("SUCCESS: The model is robust against event pile-up.")

if __name__ == "__main__":
    run_blind_test()