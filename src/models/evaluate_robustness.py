"""
FOR DIAGNOSTIC PURPOSES ONLY — NOT A PHYSICAL RESULT
Used for internal tuning of PSA features.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib

def plot_roc_curve():
    # 1. Load data and model
    df = pd.read_csv("data/processed/pulse_features.csv")
    X = df[['peak', 'rise_time', 'area', 'fwhm', 'chi_sq']]
    y = df['label']
    
    model = joblib.load("results/models/psa_random_forest.pkl")
    
    # 2. Get probability scores
    y_probs = model.predict_proba(X)[:, 1]
    
    # 3. Calculate ROC
    fpr, tpr, thresholds = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # 4. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Leakage)')
    plt.ylabel('True Positive Rate (Signal Efficiency)')
    plt.title('PSA Performance: Signal Efficiency vs. Background Rejection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("results/figures/roc_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_roc_curve()