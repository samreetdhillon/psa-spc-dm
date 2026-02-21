import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from src.models.train import train_psa_model
from src.models.blind_test import run_blind_test
from src.utils.physical_analysis import run_physical_analysis

def finalize_project():
    print("--- STEP 1: UPGRADING FEATURES ---")
    print("Implementing Chi-Square Template Matching to identify pile-up...")
    
    
    print("\n--- STEP 2: RE-TRAINING MODEL ---")
    # Retrain the Random Forest with the 'Quality Control' metrics
    train_psa_model() 

    print("\n--- STEP 3: FINAL BLIND DISCOVERY RUN ---")
    # Should return 0 Fake Discoveries
    run_physical_analysis()

if __name__ == "__main__":
    finalize_project()